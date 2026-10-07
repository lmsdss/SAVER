import base64
import json
import os
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.qwen.vision_process import process_vision_info
from lmms_eval.models.model_utils.qwen.modeling_qwen3_vl_patched import Qwen3VLForConditionalGeneration
from lmms_eval.models.simple.early_exit import compute_first_boxed_answer_probs


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


def _merge_and_log_video_frame_stats(lm: "Qwen3_VL_AutoThink", frame_by_doc: Dict[Tuple[str, str], int]) -> None:
    merged: Dict[Tuple[str, str], int] = dict(frame_by_doc)
    if lm._world_size > 1:
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


COT_SYSTEM_PROMPT_ANSWER_TWICE = (
    "You are a helpful assistant.\n"
    "FIRST: Output your initial answer inside the first \\boxed{...} without any analysis or explanations. "
    "If you cannot determine the answer without reasoning, output \\boxed{Let's analyze the problem step by step.} instead.\n"
    "THEN: Think through the reasoning as an internal monologue enclosed within <think>...</think>.\n"
    "AT LAST: Output the final answer again inside \\boxed{...}. If you believe the previous answer was correct, repeat it; otherwise, correct it.\n"
    "Output format: \\boxed{...}<think>...</think>\\boxed{...}\n"
)


@register_model("qwen3_vl_autothink")
class Qwen3_VL_AutoThink(lmms):
    """
    Qwen3_VL Model
    "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        use_flash_attention_2: Optional[bool] = True,
        image_min_pixels: Optional[int] = 4 * 28 * 28,
        image_max_pixels: Optional[int] = 16384 * 28 * 28,
        video_min_pixels: Optional[int] = 128 * 28 * 28,
        video_max_pixels: Optional[int] = 768 * 28 * 28,
        video_total_pixels: Optional[int] = 115200 * 28 * 28,
        min_frames: Optional[int] = 4,
        max_frames: Optional[int] = 768,
        nframes: Optional[int] = None,
        fps: Optional[float] = 2.0,
        interleave_visuals: Optional[bool] = False,
        early_exit_thresh: Optional[float] = 0.98,
        inference_mode: Optional[str] = "auto",  # first, second, auto
        **kwargs,
    ) -> None:
        """Initialize the Qwen3_VL model.

        Args:
            video_min_pixels (int, optional): The minimum pixels number for a frame of a video. Defaults to 128 * 28 * 28.
            video_max_pixels (int, optional): The maximum pixels number for a frame of a video. Defaults to 768 * 28 * 28.
            video_total_pixels (int, optional): The total pixels number for a video. Defaults to 115200 * 28 * 28.
            min_frames (int, optional): The minimum frames number allowed for a video. Defaults to 4.
            max_frames (int, optional): The maximum frames number allowed for a video. Defaults to 768.
            nframes (int, optional): The exact number of frames to extract for a video. Defaults to None, meaning the frame number is decided by fps and max_frames.
            fps (float, optional): The fps to extract frames for a video. Defaults to None.

        """
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Prepare model loading arguments
        model_kwargs = {
            "dtype": "bfloat16",
            "device_map": self.device_map,
        }

        # Add attention implementation if specified
        if use_flash_attention_2 is not None:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self._model = Qwen3VLForConditionalGeneration.from_pretrained(pretrained, **model_kwargs).eval()

        # data configuration for fetch_image
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels
        eval_logger.info(f"image_min_pixels: {self.image_min_pixels}, image_max_pixels: {self.image_max_pixels}")

        # data configuration for fetch_video
        self.video_min_pixels = video_min_pixels
        self.video_max_pixels = video_max_pixels
        self.video_total_pixels = video_total_pixels
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.nframes = nframes
        self.fps = fps
        eval_logger.info(
            f"video_min_pixels: {self.video_min_pixels}, "
            f"video_max_pixels: {self.video_max_pixels}, "
            f"video_total_pixels: {self.video_total_pixels}, "
            f"min_frames: {self.min_frames}, "
            f"max_frames: {self.max_frames}, "
            f"nframes: {self.nframes}, "
            f"fps: {self.fps}"
        )

        self.system_prompt = COT_SYSTEM_PROMPT_ANSWER_TWICE
        eval_logger.info(f"system_prompt: {self.system_prompt}")

        self.inference_mode = inference_mode
        assert self.inference_mode in ["first", "second", "auto"], f"Invalid inference_mode: {self.inference_mode}"
        eval_logger.info(f"inference_mode: {self.inference_mode}")

        if self.inference_mode == "auto":
            self.early_exit_thresh = early_exit_thresh
            eval_logger.info(f"early_exit_thresh: {self.early_exit_thresh}")

        self.processor = AutoProcessor.from_pretrained(pretrained)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        self._max_length = kwargs.get("max_length", 4096)
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
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
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
        raise NotImplementedError("Loglikelihood is not implemented for Qwen3_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # (task, doc_id) -> total frames for that sample (dedupes DDP padding repeats)
        frame_by_doc: Dict[Tuple[str, str], int] = {}
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            tasks_per_item = task
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            # Set default until or update values from gen_kwargs if present
            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])

            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(
                    f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}"
                )

            # Avoid using '\n\n' as a stopper for Qwen3_VL to prevent truncation, which can lead to incorrect results
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            batched_messages = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")

                message = [{"role": "system", "content": self.system_prompt}]

                processed_visuals = []
                if visual_list[i] is not None:
                    for visual in visual_list[i]:
                        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                            visual_dict = {
                                "type": "video",
                                "video": visual,
                                "min_pixels": self.video_min_pixels,
                                "max_pixels": self.video_max_pixels,
                                "total_pixels": self.video_total_pixels,
                                "min_frames": self.min_frames,
                                "max_frames": self.max_frames,
                                "fps": self.fps,
                            }
                            if self.nframes is not None:
                                visual_dict["nframes"] = self.nframes
                                visual_dict.pop("fps")

                            processed_visuals.append(visual_dict)
                        elif isinstance(visual, Image.Image):  # Handle both single and multiple images
                            base64_image = visual.convert("RGB")
                            buffer = BytesIO()
                            base64_image.save(buffer, format="JPEG")
                            base64_bytes = base64.b64encode(buffer.getvalue())
                            base64_string = base64_bytes.decode("utf-8")
                            processed_visuals.append(
                                {
                                    "type": "image",
                                    "image": f"data:image/jpeg;base64,{base64_string}",
                                    "min_pixels": self.image_min_pixels,
                                    "max_pixels": self.image_max_pixels,
                                }
                            )

                if task == "video_mmmu_adaptation":
                    self.interleave_visuals = True
                    eval_logger.info("Interleaving visuals for video_mmmu_adaptation")

                if self.interleave_visuals is False:
                    message.append(
                        {
                            "role": "user",
                            "content": processed_visuals + [{"type": "text", "text": context}],
                        }
                    )
                else:  # currently support find <image x> in the context
                    content_parts = []

                    if processed_visuals[0]["type"] == "video":
                        content_parts.append(processed_visuals[0])
                        processed_visuals = processed_visuals[1:]

                    image_placeholders = re.findall(r"<image \d+>", context)
                    text_parts = re.split(r"<image \d+>", context)
                    if text_parts[0]:
                        content_parts.append({"type": "text", "text": text_parts[0]})

                    for i, placeholder in enumerate(image_placeholders):
                        img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                        image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                        if processed_visuals and image_idx < len(processed_visuals):
                            content_parts.append(processed_visuals[image_idx])
                        if i + 1 < len(text_parts) and text_parts[i + 1]:
                            content_parts.append({"type": "text", "text": text_parts[i + 1]})

                    message.append({"role": "user", "content": content_parts})

                batched_messages.append(message)

            texts = self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True)

            if task.startswith("mvp_"):
                texts = [f"{text}\\boxed{{Answer:" for text in texts]

            image_inputs, video_inputs_raw, video_kwargs = process_vision_info(
                batched_messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

            batch_frame_counts = _video_frame_counts_per_conversation(batched_messages, video_inputs_raw)
            for i, nf in enumerate(batch_frame_counts):
                if nf <= 0:
                    continue
                key = (str(tasks_per_item[i]), str(doc_id[i]))
                frame_by_doc[key] = nf

            video_inputs = video_inputs_raw
            if video_inputs is not None:
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs, video_metadatas = (
                    list(video_inputs),
                    list(video_metadatas),
                )
            else:
                video_metadatas = None

            padding_side = "left" if self.batch_size > 1 else "right"
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadatas,
                do_resize=False,
                padding=True,
                padding_side=padding_side,
                return_tensors="pt",
                **video_kwargs,
            )

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)
            # 无论任务写 16 还是多少，这里都会变成 4096（除非以后有人改这段逻辑）。
            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            # extend the max_new_tokens when thinking
            current_gen_kwargs["max_new_tokens"] = 4096

            gen_out = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
                return_dict_in_generate=True,
                output_scores=True,
            )

            cont = gen_out.sequences
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

            for b, (ans, context, gen_ids) in enumerate(zip(answers, contexts, generated_ids_trimmed)):

                if self.inference_mode == "first":
                    # always use the first boxed content as the answer
                    ans  = ans.split("<think>")[0]
                
                elif self.inference_mode == "second":
                    # always use the second boxed content as the answer
                    ans = ans.split("</think>")[-1]
                
                else:  # auto
                    # adaptive inference based on the confidence of the first boxed answer
                    first_box_probs = compute_first_boxed_answer_probs(
                        b=b,
                        gen_ids=gen_ids,
                        gen_out=gen_out,
                        ans=ans,
                        task=task,
                        tokenizer=self.tokenizer,
                    )
                    if self.early_exit_thresh > 0:
                        if first_box_probs >= self.early_exit_thresh:
                            ans = ans.split("<think>")[0]
                        else:
                            ans = ans.split("</think>")[-1]
                    else:
                        # if negative threshold, always use the second boxed content and record the confidence
                        # which we will return for analysis
                        ans = ans + rf"\n\n\nprobs:{first_box_probs}"

                res.append(ans)

                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        _merge_and_log_video_frame_stats(self, frame_by_doc)
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
