from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)


@dataclass
class ReasoningModelConfig:
    model_name_or_path: str
    max_new_tokens: int = 768
    temperature: float = 0.6
    top_p: float = 0.95
    do_sample: bool = True
    trust_remote_code: bool = True
    use_4bit: bool = True
    backend: str = "torch"  # torch | mlx (mlx is reserved for extension)
    device_preference: str = "mps"
    dtype: str = "float16"


@dataclass
class GenerationResult:
    full_text: str
    think_text: str
    hidden_states: Optional[torch.Tensor] = None
    hidden_state_series: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None


class ReasoningModel:
    """Wrapper for local reasoning-model inference with optional latent hooks."""

    def __init__(self, config: ReasoningModelConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.device_preference)
        self._is_mlx_backend = config.backend.lower() == "mlx"
        self.tokenizer = None
        self.model = None
        self._mlx_generate = None
        self._mlx_make_sampler = None

        if self._is_mlx_backend:
            self._load_mlx_backend()
        else:
            self._load_torch_backend()

    @staticmethod
    def extract_text_between_tags(text: str, start_tag: str, end_tag: str) -> str:
        pattern = re.compile(
            rf"{re.escape(start_tag)}(.*?){re.escape(end_tag)}",
            re.DOTALL | re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
        return text.strip()

    @classmethod
    def extract_think_text(cls, text: str) -> str:
        return cls.extract_text_between_tags(text, "<think>", "</think>")

    def generate(
        self,
        prompt: str,
        return_hidden_states: bool = False,
        logits_processor: Any | None = None,
    ) -> GenerationResult:
        if self._is_mlx_backend:
            return self._generate_mlx(prompt, return_hidden_states=return_hidden_states)

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded.")

        tokenized = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            generate_kwargs: Dict[str, Any] = {}
            if logits_processor is not None:
                generate_kwargs["logits_processor"] = logits_processor
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
                return_dict_in_generate=True,
                output_hidden_states=return_hidden_states,
                pad_token_id=self.tokenizer.eos_token_id,
                **generate_kwargs,
            )

        generated_ids = outputs.sequences[0]
        full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        think_text = self.extract_think_text(full_text)

        last_layer_hidden: Optional[torch.Tensor] = None
        hidden_state_series: Optional[torch.Tensor] = None
        if return_hidden_states:
            # Hidden states are returned per generated token step.
            # Capture the final step's final layer trajectory for future latent analysis.
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states and len(hidden_states) > 0:
                # Each element corresponds to a generation step; each step is a tuple of layers.
                per_step_last_token: List[torch.Tensor] = []
                for step in hidden_states:
                    if not step:
                        continue
                    last_layer = step[-1]  # (batch, seq_len, hidden)
                    if last_layer is None:
                        continue
                    per_step_last_token.append(last_layer[:, -1, :].detach().cpu())

                if per_step_last_token:
                    hidden_state_series = torch.cat(per_step_last_token, dim=0)  # (steps, hidden)

                final_step = hidden_states[-1]
                if final_step and len(final_step) > 0 and final_step[-1] is not None:
                    last_layer_hidden = final_step[-1].detach().cpu()
            LOGGER.info(
                "Hidden states requested. Available=%s",
                last_layer_hidden is not None,
            )

        return GenerationResult(
            full_text=full_text,
            think_text=think_text,
            hidden_states=last_layer_hidden,
            hidden_state_series=hidden_state_series,
            metadata={
                "device": str(self.device),
                "backend": self.config.backend,
                "use_4bit": self.config.use_4bit,
            },
        )

    def _generate_mlx(
        self,
        prompt: str,
        return_hidden_states: bool = False,
    ) -> GenerationResult:
        if self.model is None or self.tokenizer is None or self._mlx_generate is None:
            raise RuntimeError("MLX backend is not loaded.")

        if return_hidden_states:
            LOGGER.info(
                "Hidden states are not currently exposed via MLX path; returning None."
            )

        generate_kwargs: Dict[str, Any] = {
            "max_tokens": self.config.max_new_tokens,
        }
        if self._mlx_make_sampler is not None:
            generate_kwargs["sampler"] = self._mlx_make_sampler(
                temp=self.config.temperature,
                top_p=self.config.top_p,
            )
        full_text = self._mlx_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            **generate_kwargs,
        )
        think_text = self.extract_think_text(full_text)
        return GenerationResult(
            full_text=full_text,
            think_text=think_text,
            hidden_states=None,
            metadata={
                "device": "mlx",
                "backend": "mlx",
                "use_4bit": self.config.use_4bit,
            },
        )

    def _load_torch_backend(self) -> None:
        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
        }

        torch_dtype = self._resolve_dtype(self.config.dtype)
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

        quantization_enabled = False
        if self.config.use_4bit:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                load_kwargs["device_map"] = "auto"
                quantization_enabled = True
                LOGGER.info("4-bit quantization enabled via bitsandbytes.")
            except Exception as exc:  # pragma: no cover - environment-specific
                LOGGER.warning(
                    "4-bit quantization unavailable; falling back. reason=%s",
                    exc,
                )

        if not quantization_enabled:
            load_kwargs["device_map"] = None

        LOGGER.info(
            "Loading model=%s device=%s backend=%s quantized=%s",
            self.config.model_name_or_path,
            self.device,
            self.config.backend,
            quantization_enabled,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            **load_kwargs,
        )
        if not quantization_enabled:
            self.model = self.model.to(self.device)
        self.model.eval()

    def _load_mlx_backend(self) -> None:
        try:
            from mlx_lm import generate, load
        except Exception as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "backend='mlx' requested but mlx-lm is not available. "
                "Install mlx-lm or use backend='torch'."
            ) from exc

        LOGGER.info("Loading MLX model=%s", self.config.model_name_or_path)
        self.model, self.tokenizer = load(self.config.model_name_or_path)
        self._mlx_generate = generate

        try:
            from mlx_lm.sample_utils import make_sampler as _make_sampler
            self._mlx_make_sampler = _make_sampler
        except ImportError:
            self._mlx_make_sampler = None

    @staticmethod
    def _resolve_device(preferred: str) -> torch.device:
        if preferred == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def _resolve_dtype(dtype_name: str) -> Optional[torch.dtype]:
        normalized = dtype_name.lower()
        if normalized == "float16":
            return torch.float16
        if normalized == "bfloat16":
            return torch.bfloat16
        if normalized == "float32":
            return torch.float32
        return None
