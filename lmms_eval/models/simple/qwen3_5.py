from typing import Optional, Union

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL


@register_model("qwen3_5")
class Qwen3_5(Qwen3_VL):
    """
    Qwen3.5 model -- thin wrapper over Qwen3_VL with Qwen3.5-specific defaults.
    https://huggingface.co/Qwen/Qwen3.5-4B
    """

    DEFAULT_GEN_KWARGS = {
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
    }

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3.5-4B",
        min_pixels: int = 64 * 32 * 32,
        max_pixels: int = 128 * 32 * 32,
        total_pixels: int = 224 * 1024 * 32 * 32,
        max_num_frames: int = 768,
        max_frames: Optional[int] = None,
        enable_thinking: Optional[bool] = False, # disable thinking by default for Qwen3.5！！！！！
        adaptive_video_fps: bool = False,
        adaptive_fps_initial: float = 1.0,
        adaptive_fps_retry: float = 2.0,
        adaptive_confidence_thresh: float = 0.5,
        adaptive_confidence_mode: str = "boxed",
        **kwargs,
    ):
        # Backward-compatible aliases used by existing eval scripts.
        image_min_pixels = kwargs.pop("image_min_pixels", None)
        image_max_pixels = kwargs.pop("image_max_pixels", None)
        video_min_pixels = kwargs.pop("video_min_pixels", None)
        video_max_pixels = kwargs.pop("video_max_pixels", None)
        video_total_pixels = kwargs.pop("video_total_pixels", None)

        # Prefer video-specific settings for mixed aliases; otherwise fallback to image aliases.
        if video_min_pixels is not None:
            min_pixels = video_min_pixels
        elif image_min_pixels is not None:
            min_pixels = image_min_pixels

        if video_max_pixels is not None:
            max_pixels = video_max_pixels
        elif image_max_pixels is not None:
            max_pixels = image_max_pixels

        if video_total_pixels is not None:
            total_pixels = video_total_pixels

        # Accept max_frames as backward-compat alias for max_num_frames
        if max_frames is not None:
            max_num_frames = max_frames
        super().__init__(
            pretrained=pretrained,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            total_pixels=total_pixels,
            max_num_frames=max_num_frames,
            enable_thinking=enable_thinking,
            adaptive_video_fps=adaptive_video_fps,
            adaptive_fps_initial=adaptive_fps_initial,
            adaptive_fps_retry=adaptive_fps_retry,
            adaptive_confidence_thresh=adaptive_confidence_thresh,
            adaptive_confidence_mode=adaptive_confidence_mode,
            **kwargs,
        )