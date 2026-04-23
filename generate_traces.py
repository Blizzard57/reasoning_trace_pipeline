"""
generate_traces.py
==================
Generate reasoning traces for AIME24, AIME25, HumanEval, and GPQA using
DeepSeek-R1-Distill models, Gemma 4, or any HF-compatible model.
"""

from __future__ import annotations
import argparse, json, logging, re
from pathlib import Path
from typing import Any, Dict, List

LOGGER = logging.getLogger(__name__)

# ── prompt templates ─────────────────────────────────────────────────────────

DEEPSEEK_MATH = (
    "<\uFF5Cbegin\u2581of\u2581sentence\uFF5C>Please reason step by step, "
    "and put your final answer within \\boxed{{}}."
    "<\uFF5CUser\uFF5C>{input}<\uFF5CAssistant\uFF5C><think>\n"
)

DEEPSEEK_CODE = (
    "<\uFF5Cbegin\u2581of\u2581sentence\uFF5C>Please reason step by step. "
    "Write your final solution inside <code>...</code> tags.\n"
    "<\uFF5CUser\uFF5C>{input}<\uFF5CAssistant\uFF5C><think>\n"
)

GEMMA_TEMPLATE = (
    "<start_of_turn>user\nThink step by step.\n\n"
    "{input}<end_of_turn>\n<start_of_turn>model\n"
)

GENERIC_TEMPLATE = "Think step by step.\n\n{input}\n\nAnswer:"

TASK_TYPE_MAP: Dict[str, str] = {
    "aime24": "math", "aime25": "math", "hmmt": "math",
    "math500": "math", "gpqa": "mcq", "humaneval": "code",
}

# ── prompt helpers ───────────────────────────────────────────────────────────

def _template(model_name: str, task_type: str) -> str:
    n = model_name.lower()
    if "deepseek" in n:
        return DEEPSEEK_CODE if task_type == "code" else DEEPSEEK_MATH
    if "gemma" in n:
        return GEMMA_TEMPLATE
    return GENERIC_TEMPLATE


def build_prompt(question: str, model_name: str, task_type: str = "math") -> str:
    return _template(model_name, task_type).format(input=question)

# ── dataset loading ──────────────────────────────────────────────────────────

def load_dataset(data_dir: Path, name: str, split: str = "test") -> List[Dict[str, Any]]:
    path = data_dir / name / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    examples = [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]
    if not examples:
        return []
    if "idx" not in examples[0]:
        examples = [{"idx": i, **ex} for i, ex in enumerate(examples)]
    return sorted(examples, key=lambda x: x["idx"])


def extract_question(ex: Dict[str, Any], task_type: str) -> str:
    if task_type == "mcq":
        q = str(ex.get("question", "")).strip()
        choices = ex.get("choices", [])
        if choices:
            opts = "\n".join(f"{lbl}. {opt}" for lbl, opt in zip("ABCD", choices))
            return f"{q}\n\n{opts}"
        return q
    for key in ("question", "problem", "prompt", "input", "Question"):
        if ex.get(key):
            return str(ex[key]).strip()
    return ""


def extract_think_text(full_text: str) -> str:
    m = re.search(r"<think>(.*?)</think>", full_text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else full_text.strip()

# ── inference backends ───────────────────────────────────────────────────────

def _load_torch(model_path: str, use_4bit: bool, dtype: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kw: Dict[str, Any] = {"trust_remote_code": True}
    dmap = {"float16": torch.float16, "bfloat16": torch.bfloat16,
            "float32": torch.float32}
    if dtype.lower() in dmap:
        kw["torch_dtype"] = dmap[dtype.lower()]
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16)
            kw["device_map"] = "auto"
        except Exception as e:
            LOGGER.warning("4-bit unavailable: %s", e)
            use_4bit = False
    if not use_4bit:
        kw["device_map"] = None
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
    if not use_4bit:
        dev = (torch.device("mps") if torch.backends.mps.is_available() else
               torch.device("cuda") if torch.cuda.is_available() else
               torch.device("cpu"))
        model = model.to(dev)
    model.eval()
    return model, tok


def _gen_torch(model, tok, prompt: str, max_new_tokens: int,
               temperature: float, top_p: float) -> str:
    import torch
    dev = next(model.parameters()).device
    inp = tok(prompt, return_tensors="pt")
    ids = inp["input_ids"].to(dev)
    mask = inp.get("attention_mask")
    if mask is not None:
        mask = mask.to(dev)
    do_sample = temperature > 0
    with torch.no_grad():
        out = model.generate(
            input_ids=ids, attention_mask=mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            do_sample=do_sample, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True)


def _gen_mlx(mlx_model, mlx_tokenizer, prompt: str, max_new_tokens: int,
             temperature: float, top_p: float) -> str:
    from mlx_lm import generate
    try:
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=temperature, top_p=top_p)
        # ADDED verbose=True HERE
        return generate(mlx_model, mlx_tokenizer, prompt=prompt,
                        max_tokens=max_new_tokens, sampler=sampler, verbose=True) 
    except ImportError:
        # ADDED verbose=True HERE
        return generate(mlx_model, mlx_tokenizer, prompt=prompt,
                        max_tokens=max_new_tokens, verbose=True)

# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate reasoning traces.")
    p.add_argument("--datasets", default="aime24,aime25",
                   help="Comma-separated dataset names (aime24,aime25,gpqa,humaneval).")
    p.add_argument("--data-dir", default="",
                   help="Root data directory. Default: <repo>/data.")
    p.add_argument("--model",
                   default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    p.add_argument("--backend", default="mlx", choices=["torch", "mlx"])
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--use-4bit", action="store_true")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--num-samples", type=int, default=-1,
                   help="-1 = all examples.")
    p.add_argument("--output-dir", default="outputs/traces")
    p.add_argument("--split", default="test")
    p.add_argument("--resume", action="store_true",
                   help="Skip already-generated indices.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    repo_root = Path(__file__).parent
    data_dir = Path(args.data_dir) if args.data_dir else repo_root / "data"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading model: %s  backend=%s", args.model, args.backend)
    if args.backend == "mlx":
        from mlx_lm import load as mlx_load
        _mlx_model, _mlx_tok = mlx_load(args.model)
    else:
        _torch_model, _torch_tok = _load_torch(
            args.model, use_4bit=args.use_4bit, dtype=args.dtype)

    model_slug = args.model.replace("/", "_")

    for dataset_name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        LOGGER.info("Dataset: %s", dataset_name)
        try:
            examples = load_dataset(data_dir, dataset_name, args.split)
        except FileNotFoundError as e:
            LOGGER.error("%s — skipping.", e); continue

        if args.num_samples > 0:
            examples = examples[:args.num_samples]
        task_type = TASK_TYPE_MAP.get(dataset_name, "math")
        out_file = output_dir / f"{dataset_name}_{model_slug}.jsonl"

        done_ids: set = set()
        if args.resume and out_file.exists():
            for line in out_file.read_text("utf-8").splitlines():
                try: done_ids.add(json.loads(line)["idx"])
                except Exception: pass
            LOGGER.info("Resuming %s — %d done.", dataset_name, len(done_ids))

        mode = "a" if args.resume else "w"
        with out_file.open(mode, encoding="utf-8") as fout:
            for ex in examples:
                idx = ex.get("idx", 0)
                if idx in done_ids: continue
                question = extract_question(ex, task_type)
                if not question: continue
                prompt = build_prompt(question, args.model, task_type)
                try:
                    if args.backend == "mlx":
                        full = _gen_mlx(_mlx_model, _mlx_tok, prompt,
                                        args.max_new_tokens, args.temperature, args.top_p)
                    else:
                        full = _gen_torch(_torch_model, _torch_tok, prompt,
                                          args.max_new_tokens, args.temperature, args.top_p)
                except Exception as e:
                    LOGGER.error("idx=%s failed: %s", idx, e); continue
                think = extract_think_text(full)
                record = {"idx": idx, "dataset": dataset_name, "model": args.model,
                          "question": question, "full_text": full, "think_text": think,
                          "ground_truth": str(ex.get("answer", ex.get("gt", ""))),
                          "task_type": task_type}
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                LOGGER.info("idx=%s  think_len=%d", idx, len(think))
        LOGGER.info("Done %s -> %s", dataset_name, out_file)


if __name__ == "__main__":
    main()