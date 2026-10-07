import re
from typing import Any, List, Union

import torch
import torch.nn.functional as F


def _mutable_scores_list(scores_or_gen_out: Any) -> List[torch.Tensor]:
    """Build a list of per-step logit tensors; supports HF generate output or a pre-built list."""
    if scores_or_gen_out is None:
        return []
    if hasattr(scores_or_gen_out, "scores"):
        if getattr(scores_or_gen_out, "scores", None) is None:
            return []
        return list(scores_or_gen_out.scores)
    # Caller may pass the same list we mutate; do not copy or pops would not free their refs.
    if isinstance(scores_or_gen_out, list):
        return scores_or_gen_out
    return list(scores_or_gen_out)


def compute_first_boxed_answer_probs(b, gen_ids, gen_out, ans, task, tokenizer):
    # extract logprobs for each step based on the gen_ids
    scores = _mutable_scores_list(gen_out)
    cur_lp = []
    try:
        with torch.inference_mode():
            for t, tok_id in enumerate(gen_ids):
                if t >= len(scores):
                    break
                step_tensor = scores[t]
                if step_tensor is None:
                    break
                scores[t] = None
                step_scores = step_tensor[b]  # [V]
                step_logprobs = F.log_softmax(step_scores, dim=-1)
                cur_lp.append(step_logprobs[tok_id.item()].unsqueeze(0))
                del step_tensor
    finally:
        while scores:
            st = scores.pop()
            del st
    lp_vec = torch.cat(cur_lp, dim=0) if cur_lp else torch.empty(0)

    if task.startswith("mvp_"):
        ans = f"\\boxed{{Answer:{ans}"
        _prefix_ids_tensor = tokenizer(
            "\\boxed{Answer:",
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]
        prefix_ids = _prefix_ids_tensor.to(
            device=gen_ids.device,
            dtype=gen_ids.dtype,
        )
        fake_lp = torch.zeros(
            prefix_ids.shape[0],
            device=lp_vec.device,
            dtype=lp_vec.dtype,
        )
        gen_ids = torch.cat([prefix_ids, gen_ids], dim=0)
        lp_vec = torch.cat([fake_lp, lp_vec], dim=0)

    first = extract_first_boxed_content(ans)
    first_box_tok_logprobs = None

    if first:
        first_content, _, _ = first  # content only
        # Exclude cases starting with "Let's analyze" (ignoring leading whitespace).
        if not first_content.lstrip().startswith("Let's analyze"):
            # Normalize (normalize_text may clean up choices/letters to forms like 'H', etc.)
            first_box_norm = normalize_text(first_content)

            # Locate the normalized text within the token sequence
            span = find_token_span_for_text(
                gen_ids=gen_ids,
                text_piece=first_box_norm,
                tokenizer=tokenizer,
                decoded_answer=ans,
            )
            if span is not None and lp_vec.numel() > 0:
                s, e = span
                # Defensive clipping (shouldn't be necessary in theory)
                s = max(0, min(s, lp_vec.shape[0]))
                e = max(0, min(e, lp_vec.shape[0]))
                if e > s:
                    first_box_tok_logprobs = lp_vec[s:e]

                    # for mvp, the answer is like "Answer: A", so we use the last token
                    if task.startswith("mvp_"):
                        first_box_tok_logprobs = first_box_tok_logprobs[-1]

    if first_box_tok_logprobs is None:
        first_box_probs = -1
    else:
        first_box_probs = first_box_tok_logprobs.mean().exp().item()

    return first_box_probs
              
_PATTERN_BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{(?:[^{}]+|\{[^{}]*\})*\}[^{}]*)*)\}")


def extract_first_boxed_content(text: str):
    """
    Returns:
        (content, inner_start, inner_end)
        - content: inner text of the first \\boxed{...} (group 1)
        - inner_start, inner_end: character indices of that inner content in `text` (end is exclusive)

    Requirement: the text must contain at least two \\boxed{...} occurrences; otherwise return False.
    """
    it = _PATTERN_BOXED.finditer(text)
    m1 = next(it, None)
    if m1 is None:
        return False
    if next(it, None) is None:  # require at least two boxed occurrences
        return False
    content = m1.group(1)
    inner_start, inner_end = m1.span(1)  # return the span of the *inner* content only
    return content, inner_start, inner_end


def _find_subsequence(haystack_ids, needle_ids):
    """
    Return (start_idx, end_idx); return None if not found.
    """
    if not needle_ids:
        return None
    n = len(needle_ids)
    limit = len(haystack_ids) - n + 1
    for i in range(max(0, 0), max(0, limit)):
        if haystack_ids[i : i + n] == needle_ids:
            return i, i + n
    # Edge case: if the needle is longer than the haystack, fail directly
    if limit <= 0 and haystack_ids == needle_ids:
        return 0, len(haystack_ids)
    return None


def _first_nonempty_find(text, variants):
    """Find the first occurring variant in `text` (in order). Return (variant, char_pos) or (None, -1)."""
    for v in variants:
        if not v:
            continue
        pos = text.find(v)
        if pos != -1:
            return v, pos
    return None, -1


def find_token_span_for_text(gen_ids, text_piece, tokenizer, decoded_answer):
    """
    Goal: Given the decoded complete answer string `decoded_answer`, its generated token sequence `gen_ids`,
          and a text fragment `text_piece`, find the corresponding token span for that fragment.

    Strategy:
      A) Encode `text_piece` into tokens and search it as a subsequence in `gen_ids`
         using multiple textual variants: original / stripped / lstrip / prefixed with a space.
      B) If (A) fails: locate the fragment via `str.find()` in `decoded_answer`, then
         re-encode `decoded_answer[:pos]` and the chosen fragment to infer the token span by length.

    Returns: (tok_start, tok_end) or None
    """
    # Common variants: original, strip, lstrip, prefixed space
    candidates_text = [
        text_piece,
        text_piece.strip(),
        text_piece.lstrip(),
        (" " + text_piece) if not text_piece.startswith(" ") else text_piece,
    ]
    # (A) Direct token subsequence match
    for cand in candidates_text:
        cand_ids = tokenizer.encode(cand, add_special_tokens=False)
        if not cand_ids:
            continue
        span = _find_subsequence(gen_ids, cand_ids)
        if span is not None:
            return span

    # (B) Fallback: use character position + re-encoding to estimate the token span
    chosen, pos = _first_nonempty_find(decoded_answer, candidates_text)
    if chosen is not None:
        prefix_ids = tokenizer.encode(decoded_answer[:pos], add_special_tokens=False)
        chosen_ids = tokenizer.encode(chosen, add_special_tokens=False)
        start = len(prefix_ids)
        end = start + len(chosen_ids)
        if end <= len(gen_ids):
            return (start, end)

    return None


_CHOICE_PAREN = re.compile(r"""^\s*[\(\[\{]\s*([A-Za-z])\s*[\)\]\}]\s*(?:[.)/:;\-]\s*)?""", re.X)
_CHOICE_BARE_WITH_DELIM = re.compile(r"""^\s*([A-Za-z])\s*[.)/:;\-]\s*""", re.X)
_CHOICE_SINGLE_LETTER = re.compile(r"""^\s*([A-Za-z])\s*[.]?\s*$""", re.X)


def normalize_text(s):
    m = _CHOICE_PAREN.match(s) or _CHOICE_BARE_WITH_DELIM.match(s) or _CHOICE_SINGLE_LETTER.match(s)
    if m:
        return m.group(1)
    else:
        return s


def extract_first_boxed_content_single(text: str):
    """First ``\\boxed{...}`` inner content; single occurrence is enough (unlike ``extract_first_boxed_content``)."""
    m = _PATTERN_BOXED.search(text)
    if m is None:
        return False
    content = m.group(1)
    inner_start, inner_end = m.span(1)
    return content, inner_start, inner_end


def compute_answer_span_confidence(
    b,
    gen_ids,
    scores_or_gen_out: Union[Any, List[torch.Tensor]],
    ans,
    task,
    tokenizer,
    confidence_mode: str = "boxed",
):
    """
    Confidence proxy: mean exp(log p(token)) over a chosen span.

    ``scores_or_gen_out``: HF ``generate`` return (with ``.scores``) **or** a ``list`` of per-step
    logit tensors.     When a list is passed, each step's tensor is cleared after use so GPU memory for that
    timestep can be freed before the next (critical for long generations with ``output_scores=True``).

    ``confidence_mode`` (aligned with training ``adaptive_confidence_mode``):

    - ``full``: mean exp(log p) over the entire generated continuation (after optional mvp prefix handling).
    - ``boxed`` (default): mean exp(log p) over tokens aligned to the **last** ``\\boxed{...}`` inner span
      (same as ``saver.utils.adaptive_confidence`` / reward), with fallback to full-sequence
      when no box or span cannot be aligned.
    """
    if confidence_mode not in ("full", "boxed"):
        raise ValueError(f"confidence_mode must be 'full' or 'boxed', got {confidence_mode!r}")

    scores = _mutable_scores_list(scores_or_gen_out)
    cur_lp = []
    try:
        with torch.inference_mode():
            for t, tok_id in enumerate(gen_ids):
                if t >= len(scores):
                    break
                step_tensor = scores[t]
                if step_tensor is None:
                    break
                scores[t] = None
                step_scores = step_tensor[b]
                step_logprobs = F.log_softmax(step_scores, dim=-1)
                cur_lp.append(step_logprobs[tok_id.item()].unsqueeze(0))
                del step_tensor
    finally:
        for j in range(len(scores)):
            scores[j] = None

    lp_vec = torch.cat(cur_lp, dim=0) if cur_lp else torch.empty(0)

    mvp_prefix_len = 0
    if task.startswith("mvp_"):
        ans = f"\\boxed{{Answer:{ans}"
        _prefix_ids_tensor = tokenizer(
            "\\boxed{Answer:",
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]
        prefix_ids = _prefix_ids_tensor.to(device=gen_ids.device, dtype=gen_ids.dtype)
        fake_lp = torch.zeros(
            prefix_ids.shape[0],
            device=lp_vec.device,
            dtype=lp_vec.dtype,
        )
        gen_ids = torch.cat([prefix_ids, gen_ids], dim=0)
        lp_vec = torch.cat([fake_lp, lp_vec], dim=0)
        mvp_prefix_len = prefix_ids.shape[0]

    def _full_tail_confidence() -> float:
        if lp_vec.numel() <= mvp_prefix_len:
            return -1.0
        tail = lp_vec[mvp_prefix_len:]
        return tail.mean().exp().item()

    if confidence_mode == "full":
        return _full_tail_confidence()

    # boxed: last \\boxed{...} inner span, then full-sequence fallback (training-aligned).
    boxed_lp = None
    try:
        from utils.adaptive_confidence import (
            extract_last_boxed_inner,
            find_token_span_for_text,
            normalize_text as _norm_train,
        )

        inner = extract_last_boxed_inner(ans)
        if inner is not None and inner.strip() and not inner.lstrip().startswith("Let's analyze"):
            piece = _norm_train(inner.strip())
            if piece:
                span = find_token_span_for_text(
                    gen_ids=gen_ids,
                    text_piece=piece,
                    tokenizer=tokenizer,
                    decoded_answer=ans,
                )
                if span is not None and lp_vec.numel() > 0:
                    s, e = span
                    s = max(0, min(s, lp_vec.shape[0]))
                    e = max(0, min(e, lp_vec.shape[0]))
                    if e > s:
                        seg = lp_vec[s:e]
                        if task.startswith("mvp_"):
                            seg = seg[-1]
                        boxed_lp = seg

    except ImportError:
        boxed_lp = None

    # Legacy fallback when ``utils.adaptive_confidence`` is unavailable (e.g. minimal PYTHONPATH):
    # first ``\\boxed{...}`` inner span, same as older eval behavior.
    if boxed_lp is None:
        first = extract_first_boxed_content_single(ans)
        if first:
            first_content, _, _ = first
            if not first_content.lstrip().startswith("Let's analyze"):
                first_box_norm = normalize_text(first_content)
                span = find_token_span_for_text(
                    gen_ids=gen_ids,
                    text_piece=first_box_norm,
                    tokenizer=tokenizer,
                    decoded_answer=ans,
                )
                if span is not None and lp_vec.numel() > 0:
                    s, e = span
                    s = max(0, min(s, lp_vec.shape[0]))
                    e = max(0, min(e, lp_vec.shape[0]))
                    if e > s:
                        seg = lp_vec[s:e]
                        if task.startswith("mvp_") and seg.numel() > 0:
                            seg = seg[-1]
                        boxed_lp = seg

    if boxed_lp is not None and boxed_lp.numel() > 0:
        return boxed_lp.mean().exp().item()

    return _full_tail_confidence()