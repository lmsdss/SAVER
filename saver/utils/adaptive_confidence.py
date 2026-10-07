"""
Adaptive FPS gate confidence: full-sequence vs last \\boxed{...} inner span (aligned with accuracy_boxed_reward).
"""

from __future__ import annotations

import re
from typing import Literal, Optional, Sequence, Union

import torch
import torch.nn.functional as F

# Extract the final boxed answer span (last box wins).
_PATTERN_BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")

_CHOICE_PAREN = re.compile(r"""^\s*[\(\[\{]\s*([A-Za-z])\s*[\)\]\}]\s*(?:[.)/:;\-]\s*)?""", re.X)
_CHOICE_BARE_WITH_DELIM = re.compile(r"""^\s*([A-Za-z])\s*[.)/:;\-]\s*""", re.X)
_CHOICE_SINGLE_LETTER = re.compile(r"""^\s*([A-Za-z])\s*[.]?\s*$""", re.X)


def normalize_text(s: str) -> str:
    """Normalize multiple-choice style answers (same spirit as lmms_eval early_exit)."""
    m = _CHOICE_PAREN.match(s) or _CHOICE_BARE_WITH_DELIM.match(s) or _CHOICE_SINGLE_LETTER.match(s)
    if m:
        return m.group(1)
    return s


def _find_subsequence(haystack_ids: Sequence[int], needle_ids: Sequence[int]):
    if not needle_ids:
        return None
    n = len(needle_ids)
    limit = len(haystack_ids) - n + 1
    for i in range(max(0, limit)):
        if list(haystack_ids[i : i + n]) == list(needle_ids):
            return i, i + n
    if limit <= 0 and list(haystack_ids) == list(needle_ids):
        return 0, len(haystack_ids)
    return None


def _first_nonempty_find(text: str, variants):
    for v in variants:
        if not v:
            continue
        pos = text.find(v)
        if pos != -1:
            return v, pos
    return None, -1


def find_token_span_for_text(
    gen_ids: Union[torch.Tensor, Sequence[int]],
    text_piece: str,
    tokenizer,
    decoded_answer: str,
) -> Optional[tuple[int, int]]:
    """
    Map a text fragment to token indices within `gen_ids` (completion tokens only).
    """
    if isinstance(gen_ids, torch.Tensor):
        gen_ids_list = gen_ids.detach().cpu().tolist()
    else:
        gen_ids_list = list(gen_ids)

    candidates_text = [
        text_piece,
        text_piece.strip(),
        text_piece.lstrip(),
        (" " + text_piece) if not text_piece.startswith(" ") else text_piece,
    ]
    for cand in candidates_text:
        cand_ids = tokenizer.encode(cand, add_special_tokens=False)
        if not cand_ids:
            continue
        span = _find_subsequence(gen_ids_list, cand_ids)
        if span is not None:
            return span

    chosen, pos = _first_nonempty_find(decoded_answer, candidates_text)
    if chosen is not None:
        prefix_ids = tokenizer.encode(decoded_answer[:pos], add_special_tokens=False)
        chosen_ids = tokenizer.encode(chosen, add_special_tokens=False)
        start = len(prefix_ids)
        end = start + len(chosen_ids)
        if end <= len(gen_ids_list):
            return (start, end)

    return None


def extract_last_boxed_inner(text: str) -> Optional[str]:
    """Return inner text of the last \\boxed{...}, or None if none matched."""
    matches = _PATTERN_BOXED.findall(text)
    if not matches:
        return None
    return matches[-1]


def compute_confidence_full(
    generated_ids: torch.Tensor,
    generation_scores: tuple[torch.Tensor, ...],
    sample_idx: int,
) -> float:
    """Mean exp(log p(token)) over the full generated continuation."""
    token_logps = []
    for t, tok_id in enumerate(generated_ids):
        if t >= len(generation_scores):
            break
        step_scores = generation_scores[t][sample_idx]
        step_logprobs = F.log_softmax(step_scores, dim=-1)
        token_logps.append(step_logprobs[int(tok_id.item())])
    if not token_logps:
        return 0.0
    return torch.stack(token_logps).mean().exp().item()


def compute_confidence_boxed(
    generated_ids: torch.Tensor,
    generation_scores: tuple[torch.Tensor, ...],
    sample_idx: int,
    tokenizer,
    decoded_text: str,
) -> float:
    """
    Mean exp(log p) over tokens that correspond to the last \\boxed{...} inner content.
    Falls back to full-sequence confidence if no box or span cannot be aligned.
    """
    inner = extract_last_boxed_inner(decoded_text)
    if inner is None or not inner.strip():
        return compute_confidence_full(generated_ids, generation_scores, sample_idx)

    piece = normalize_text(inner.strip())
    if not piece:
        return compute_confidence_full(generated_ids, generation_scores, sample_idx)

    span = find_token_span_for_text(generated_ids, piece, tokenizer, decoded_text)
    if span is None:
        return compute_confidence_full(generated_ids, generation_scores, sample_idx)

    s, e = span
    s = max(0, min(s, len(generated_ids)))
    e = max(s, min(e, len(generated_ids)))
    if e <= s:
        return compute_confidence_full(generated_ids, generation_scores, sample_idx)

    token_logps = []
    for t in range(s, e):
        if t >= len(generation_scores):
            break
        tok_id = generated_ids[t]
        step_scores = generation_scores[t][sample_idx]
        step_logprobs = F.log_softmax(step_scores, dim=-1)
        token_logps.append(step_logprobs[int(tok_id.item())])
    if not token_logps:
        return compute_confidence_full(generated_ids, generation_scores, sample_idx)
    return torch.stack(token_logps).mean().exp().item()


ConfidenceMode = Literal["full", "boxed"]


def mean_exp_logps(token_logps: torch.Tensor) -> float:
    """Geometric mean probability as exp(mean log p) over a 1D logprob tensor."""
    if token_logps.numel() == 0:
        return 0.0
    return token_logps.float().mean().exp().item()


def compute_confidence_full_from_logps(token_logps: torch.Tensor) -> float:
    return mean_exp_logps(token_logps)


def compute_confidence_boxed_from_logps(
    generated_ids: torch.Tensor,
    token_logps: torch.Tensor,
    tokenizer,
    decoded_text: str,
) -> float:
    """Boxed span confidence when per-step sampled log p is already known (e.g. vLLM)."""
    inner = extract_last_boxed_inner(decoded_text)
    if inner is None or not inner.strip():
        return compute_confidence_full_from_logps(token_logps)

    piece = normalize_text(inner.strip())
    if not piece:
        return compute_confidence_full_from_logps(token_logps)

    span = find_token_span_for_text(generated_ids, piece, tokenizer, decoded_text)
    if span is None:
        return compute_confidence_full_from_logps(token_logps)

    s, e = span
    n = min(len(generated_ids), token_logps.shape[0])
    s = max(0, min(s, n))
    e = max(s, min(e, n))
    if e <= s:
        return compute_confidence_full_from_logps(token_logps)
    sl = token_logps[s:e]
    if sl.numel() == 0:
        return compute_confidence_full_from_logps(token_logps)
    return mean_exp_logps(sl)


def compute_adaptive_confidence(
    mode: ConfidenceMode,
    generated_ids: torch.Tensor,
    generation_scores: tuple[torch.Tensor, ...],
    sample_idx: int,
    tokenizer,
    decoded_text: str,
    token_logps_completion: Optional[torch.Tensor] = None,
) -> float:
    """
    If ``token_logps_completion`` is set (vLLM path), confidence uses sampled-token log p per step
    and ignores ``generation_scores``.
    """
    if token_logps_completion is not None and token_logps_completion.numel() > 0:
        gen = generated_ids
        tlp = token_logps_completion
        n = min(int(gen.shape[0]), int(tlp.shape[0]))
        gen = gen[:n]
        tlp = tlp[:n]
        if mode == "boxed":
            return compute_confidence_boxed_from_logps(gen, tlp, tokenizer, decoded_text)
        return compute_confidence_full_from_logps(tlp)

    if mode == "boxed":
        return compute_confidence_boxed(
            generated_ids, generation_scores, sample_idx, tokenizer, decoded_text
        )
    return compute_confidence_full(generated_ids, generation_scores, sample_idx)
