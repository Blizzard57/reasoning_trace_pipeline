# Reasoning Trace Pattern Report

**Source:** aime24_deepseek-ai_DeepSeek-R1-Distill-Qwen-7B_analysis.jsonl, aime24_mlx-community_DeepSeek-R1-Distill-Qwen-7B-4bit_analysis.jsonl

## 1. Where uninformative steps appear
| Position | Count | Fraction |
|----------|-------|----------|
| Early (0–33%) | 148 | 32.0% |
| Middle (33–67%) | 160 | 34.6% |
| Late (67–100%) | 155 | 33.5% |

**Interpretation:** If early >> late, models front-load hedging.
If late >> early, models over-verify near the answer.

## 2. Redundant step gap distribution
| Gap (|i-j|) | Count |
|------------|-------|
| 1 | 85 |
| 2 | 84 |
| 3 | 66 |
| 4 | 87 |
| 5 | 63 |
| 6 | 63 |
| 7 | 60 |
| 8 | 50 |
| 9 | 34 |
| 10 | 44 |
| 11 | 43 |
| 12 | 33 |
| 13 | 40 |
| 14 | 30 |
| 15 | 29 |
| 16 | 21 |
| 17 | 22 |
| 18 | 24 |
| 19 | 22 |
| 20 | 16 |
| 21 | 20 |
| 22 | 20 |
| 23 | 18 |
| 24 | 19 |
| 25 | 17 |
| 26 | 15 |
| 27 | 11 |
| 28 | 8 |
| 29 | 7 |
| 30 | 8 |
| 31 | 7 |
| 32 | 12 |
| 33 | 7 |
| 34 | 7 |
| 35 | 9 |
| 36 | 6 |
| 37 | 8 |
| 38 | 5 |
| 39 | 7 |
| 40 | 7 |
| 41 | 11 |
| 42 | 5 |
| 43 | 6 |
| 44 | 5 |
| 45 | 6 |
| 46 | 7 |
| 47 | 4 |
| 48 | 5 |
| 49 | 8 |
| 50 | 4 |
| 51 | 3 |
| 52 | 3 |
| 53 | 3 |
| 54 | 3 |
| 55 | 1 |
| 56 | 4 |
| 57 | 3 |
| 58 | 3 |
| 59 | 4 |
| 60 | 2 |
| 61 | 1 |
| 62 | 2 |
| 63 | 1 |
| 64 | 2 |
| 66 | 1 |
| 68 | 1 |
| 71 | 1 |
| 72 | 1 |
| 73 | 1 |
| 75 | 1 |
| 77 | 1 |
| 82 | 1 |
| 83 | 1 |
| 84 | 1 |
| 85 | 1 |
| 86 | 1 |
| 89 | 3 |
| 90 | 1 |
| 95 | 1 |
| 96 | 1 |

**Adjacent (gap=1):** 85  |  **Non-adjacent:** 1163

If gap=1 dominates: adjacent step pairs are often paraphrases.
If gap>1 is common: the model revisits earlier ideas much later.

## 3. Step-label transition bigrams (top 20)
| Transition | Count |
|------------|-------|
| uninformative->uninformative | 394 |
| uninformative->informative | 37 |
| informative->uninformative | 31 |
| marginal->uninformative | 26 |
| uninformative->marginal | 23 |
| informative->informative | 14 |
| informative->marginal | 6 |
| marginal->marginal | 3 |
| marginal->informative | 3 |

**uninformative→informative:** 37
**informative→uninformative:** 31

High uninformative→informative count suggests brief "pivot" phrases
before substantive reasoning steps — a common model habit.
