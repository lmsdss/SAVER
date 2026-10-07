import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import resolve_qwen_vl_process_vision_info

process_vision_info, _has_qwen_vl = resolve_qwen_vl_process_vision_info()
if not _has_qwen_vl:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


def _resolve_model_class(pretrained: str, is_moe: bool):
    """Auto-detect and return the appropriate HF model class for a Qwen3 variant.

    Returns (model_class, dtype_kwarg_name) where dtype_kwarg_name is the
    keyword argument name for specifying dtype in from_pretrained().
    """
    config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")

    if "qwen3_5" in model_type:
        from transformers import (
            Qwen3_5ForConditionalGeneration,
            Qwen3_5MoeForConditionalGeneration,
        )

        model_cls = Qwen3_5MoeForConditionalGeneration if is_moe else Qwen3_5ForConditionalGeneration
        dtype_key = "torch_dtype"
    else:
        from transformers import (
            Qwen3VLForConditionalGeneration,
            Qwen3VLMoeForConditionalGeneration,
        )

        model_cls = Qwen3VLMoeForConditionalGeneration if is_moe else Qwen3VLForConditionalGeneration
        dtype_key = "dtype"

    return model_cls, dtype_key


def _video_frame_counts_per_conversation(
    batched_messages: List[List[Dict[str, Any]]],
    video_inputs_raw: Optional[List[Any]],
) -> List[int]:
    """Match `process_vision_info` order: sum temporal length (T) of each video per conversation."""
    n = len(batched_messages)
    if video_inputs_raw is None:
        return [0] * n
    frame_sizes: List[int] = []
    for vi in video_inputs_raw:
        tensor = vi[0] if isinstance(vi, (tuple, list)) else vi
        frame_sizes.append(int(tensor.shape[0]))
    counts = [0] * n
    cursor = 0
    for i, conv in enumerate(batched_messages):
        for message in conv:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for ele in content:
                if "video" in ele:
                    if cursor >= len(frame_sizes):
                        raise RuntimeError("video_inputs and batched_messages video count mismatch")
                    counts[i] += frame_sizes[cursor]
                    cursor += 1
    if cursor != len(frame_sizes):
        raise RuntimeError(
            f"video_inputs and batched_messages video count mismatch: {cursor} vs {len(frame_sizes)}"
        )
    return counts


def _merge_and_log_video_frame_stats(lm: Any, frame_by_doc: Dict[Tuple[str, str], int]) -> None:
    merged: Dict[Tuple[str, str], int] = dict(frame_by_doc)
    if getattr(lm, "_world_size", 1) > 1:
        from accelerate.utils import gather_object

        gathered = gather_object([frame_by_doc])
        if lm.rank != 0:
            return
        merged = {}
        for d in gathered:
            merged.update(d)
    elif lm.rank != 0:
        return

    txt_path = os.environ.get("LMMS_VIDEO_FRAME_STATS_TXT", "").strip()
    current_task = os.environ.get("LMMS_CURRENT_TASK", "").strip()

    if not merged:
        eval_logger.info("[video_frame_stats] No video tensors in this run (skipped or image-only).")
        if txt_path:
            label = current_task or "unknown_task"
            parent = os.path.dirname(txt_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(txt_path, "a", encoding="utf-8") as f:
                f.write(f"{'=' * 60}\n")
                f.write(f"task: {label}\n")
                f.write("  num_video_samples: 0\n")
                f.write("  note: no video tensors in this run\n\n")
        return

    vals = list(merged.values())
    task_names = {k[0] for k in merged.keys()}
    task_str = ",".join(sorted(task_names)) if len(task_names) <= 3 else f"{len(task_names)}_tasks"

    stats = {
        "tasks": sorted(task_names),
        "num_video_samples": len(vals),
        "total_frames": int(sum(vals)),
        "mean_frames_per_sample": float(sum(vals) / len(vals)),
        "min_frames_per_sample": int(min(vals)),
        "max_frames_per_sample": int(max(vals)),
    }
    eval_logger.info(
        "[video_frame_stats] "
        f"tasks={task_str} | samples={stats['num_video_samples']} | "
        f"total_frames={stats['total_frames']} | "
        f"mean={stats['mean_frames_per_sample']:.2f} | "
        f"min={stats['min_frames_per_sample']} | max={stats['max_frames_per_sample']}"
    )

    if txt_path:
        parent = os.path.dirname(txt_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        label = current_task or (stats["tasks"][0] if len(stats["tasks"]) == 1 else task_str)
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write(f"{'=' * 60}\n")
            f.write(f"task: {label}\n")
            f.write(f"  num_video_samples: {stats['num_video_samples']}\n")
            f.write(f"  total_frames: {stats['total_frames']}\n")
            f.write(f"  mean_frames_per_sample: {stats['mean_frames_per_sample']:.4f}\n")
            f.write(f"  min_frames_per_sample: {stats['min_frames_per_sample']}\n")
            f.write(f"  max_frames_per_sample: {stats['max_frames_per_sample']}\n\n")

    out = os.environ.get("LMMS_VIDEO_FRAME_STATS_JSONL", "").strip()
    if out:
        record = {**stats, "tasks": list(stats["tasks"])}
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


@register_model("qwen3_vl")
class Qwen3_VL(lmms):
    """
    Unified model class for the Qwen3-VL and Qwen3.5 model families.

    Auto-detects the model variant from the HuggingFace config and loads
    the appropriate model class. Supports both dense and MoE variants.

    - Qwen3-VL: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
    - Qwen3.5:  https://huggingface.co/Qwen/Qwen3.5-4B
    """

    DEFAULT_GEN_KWARGS = {
        "max_new_tokens": 128,
        "temperature": 0.0,
        "top_p": None,
        "num_beams": 1,
    }

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        total_pixels: Optional[int] = None,
        max_num_frames: int = 32,
        fps: Optional[float] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        enable_thinking: Optional[bool] = None,
        reasoning_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Auto-detect model variant and load the appropriate HF class
        is_moe = bool(re.search(r"A\d+B", pretrained))
        model_cls, dtype_key = _resolve_model_class(pretrained, is_moe)

        model_kwargs = {
            dtype_key: "bfloat16",
            "device_map": self.device_map,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = model_cls.from_pretrained(pretrained, **model_kwargs).eval()
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_num_frames = max_num_frames
        self.fps = fps
        self.enable_thinking = enable_thinking

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None

        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        self._max_length = 2048
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen3 VL models")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _build_video_kwargs(self):
        """Build video processing kwargs based on model configuration."""
        video_kwargs = {"min_pixels": self.min_pixels}

        if self.fps is not None:
            video_kwargs["fps"] = self.fps
            video_kwargs["max_frames"] = self.max_num_frames
        elif self.total_pixels is not None:
            video_kwargs["max_frames"] = self.max_num_frames
        else:
            video_kwargs["nframes"] = self.max_num_frames

        if self.total_pixels is not None:
            video_kwargs["total_pixels"] = self.total_pixels
        else:
            video_kwargs["max_pixels"] = self.max_pixels

        return video_kwargs

    def _apply_chat_template(self, batched_messages, **kwargs):
        """Apply chat template with optional enable_thinking support."""
        template_kwargs = {}
        if self.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.enable_thinking
        template_kwargs.update(kwargs)
        return self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True, **template_kwargs)

    def _build_generate_kwargs(self, gen_kwargs):
        """Build model.generate() kwargs from user gen_kwargs merged with defaults."""
        current = {**self.DEFAULT_GEN_KWARGS, **gen_kwargs}
        pad_token_id = self.tokenizer.pad_token_id

        if current.get("temperature", 0) > 0:
            current["do_sample"] = True
        else:
            current["do_sample"] = False
            current["temperature"] = None
            current["top_p"] = None
            current.pop("top_k", None)

        generate_kwargs = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": pad_token_id,
            "max_new_tokens": current["max_new_tokens"],
            "use_cache": self.use_cache,
            "do_sample": current["do_sample"],
        }
        for key in ("temperature", "top_p", "top_k", "num_beams"):
            val = current.get(key)
            if val is not None:
                generate_kwargs[key] = val

        return generate_kwargs

    def _strip_thinking(self, answer):
        """Strip <think>...</think> content from model output if enable_thinking is set."""
        if self.enable_thinking:
            _, _, remaining = answer.partition("</think>")
            return remaining.strip()
        return answer

    def _preprocess_chunk(self, chunk):
        """Preprocess a batch chunk on CPU: message building, video decoding, tokenization.

        Returns (inputs, contexts, gen_kwargs, until, frame_by_doc_chunk) with inputs still on CPU.
        frame_by_doc_chunk maps (task_name, doc_id) -> total frame count for that sample.
        """
        contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
        visual_list = [doc_to_visual[0](self.task_dict[t][s][i]) for t, s, i in zip(task, split, doc_id)]
        gen_kwargs = all_gen_kwargs[0]

        until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
        if isinstance(until, str):
            until = [until]
        elif not isinstance(until, list):
            raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")
        until = [item for item in until if item != "\n\n"]

        if isinstance(contexts, tuple):
            contexts = list(contexts)

        for i in range(len(contexts)):
            if "<image>" in contexts[i]:
                contexts[i] = contexts[i].replace("<image>", "")

        video_kwargs = self._build_video_kwargs()

        batched_messages = []
        for i, context in enumerate(contexts):
            if "<image>" in context:
                context = context.replace("<image>", "")

            message = [{"role": "system", "content": self.system_prompt}]

            if self.reasoning_prompt:
                context = context.strip() + self.reasoning_prompt
                contexts[i] = context

            processed_visuals = []
            if visual_list[i] is not None:
                for visual in visual_list[i]:
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
                        processed_visuals.append(
                            {
                                "type": "video",
                                "video": visual,
                                **video_kwargs,
                            }
                        )
                    elif isinstance(visual, Image.Image):
                        processed_visuals.append(
                            {
                                "type": "image",
                                "image": visual,
                                "max_pixels": self.max_pixels,
                                "min_pixels": self.min_pixels,
                            }
                        )

            if self.interleave_visuals is False:
                message.append(
                    {
                        "role": "user",
                        "content": processed_visuals + [{"type": "text", "text": context}],
                    }
                )
            else:
                image_placeholders = re.findall(r"<image \d+>", context)
                content_parts = []
                text_parts = re.split(r"<image \d+>", context)
                if text_parts[0]:
                    content_parts.append({"type": "text", "text": text_parts[0]})

                for placeholder_idx, placeholder in enumerate(image_placeholders):
                    img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                    image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                    if processed_visuals and image_idx < len(processed_visuals):
                        content_parts.append(processed_visuals[image_idx])
                    if placeholder_idx + 1 < len(text_parts) and text_parts[placeholder_idx + 1]:
                        content_parts.append({"type": "text", "text": text_parts[placeholder_idx + 1]})

                message.append(
                    {
                        "role": "user",
                        "content": content_parts,
                    }
                )

            batched_messages.append(message)

        texts = self._apply_chat_template(batched_messages)
        image_inputs, video_inputs_raw, processed_video_kwargs = process_vision_info(
            batched_messages,
            return_video_kwargs=True,
            image_patch_size=16,
            return_video_metadata=True,
        )
        batch_frame_counts = _video_frame_counts_per_conversation(batched_messages, video_inputs_raw)
        frame_by_doc_chunk: Dict[Tuple[str, str], int] = {}
        for i, nf in enumerate(batch_frame_counts):
            if nf <= 0:
                continue
            frame_by_doc_chunk[(str(task[i]), str(doc_id[i]))] = nf

        video_inputs = video_inputs_raw
        video_metadata_list = None
        if video_inputs is not None:
            video_inputs, video_metadata_list = map(list, zip(*video_inputs))

        if self.batch_size > 1:
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadata_list,
                **processed_video_kwargs,
                do_resize=False,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadata_list,
                **processed_video_kwargs,
                do_resize=False,
                return_tensors="pt",
            )

        return inputs, contexts, gen_kwargs, until, frame_by_doc_chunk

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        frame_by_doc: Dict[Tuple[str, str], int] = {}

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = list(re_ords.get_batched(n=self.batch_size, batch_fn=None))

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._preprocess_chunk, chunks[0]) if chunks else None

            for idx in range(len(chunks)):
                inputs, contexts, gen_kwargs, until, chunk_frames = future.result()
                frame_by_doc.update(chunk_frames)

                if idx + 1 < len(chunks):
                    future = executor.submit(self._preprocess_chunk, chunks[idx + 1])

                if self.device_map == "auto":
                    inputs = inputs.to("cuda")
                else:
                    inputs = inputs.to(self.device)

                generate_kwargs = self._build_generate_kwargs(gen_kwargs)
                cont = self.model.generate(**inputs, **generate_kwargs)

                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
                answers = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for i, ans in enumerate(answers):
                    for term in until:
                        if len(term) > 0:
                            ans = ans.split(term)[0]
                    answers[i] = ans

                for ans, context in zip(answers, contexts):
                    ans = self._strip_thinking(ans)
                    res.append(ans)
                    self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                    pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        _merge_and_log_video_frame_stats(self, frame_by_doc)
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")