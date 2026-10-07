# Evaluation data preparation


Set `LMMS_DATA_ROOT` to the directory containing the extracted benchmark video folders. If unset, the launcher uses `HF_HOME`, then `~/.cache/huggingface`. Task YAML files declare their annotation dataset identifiers and cache directory names. Downloaded Hugging Face annotation datasets do not necessarily include the video archives.

| Benchmark |  Video directory under LMMS_DATA_ROOT |
| --- | --- |
| Charades-STA | charades_sta |
| ActivityNet | activitynet |
| NeXT-GQA |nextgqa |
| MVBench |  mvbench_video |
| MMVU |  mmvu |
| LongVideoBench | longvideobench |
| VideoMMMU | VideoMMMU |
| Video-MME |  videomme |
| MVP | minimal_video_pairs |

Consult each task's YAML and `utils.py` for the exact internal video/subdirectory layout and upstream dataset identifier. Preserve archive subdirectories expected by those adapters. Authenticate with Hugging Face only if a dataset requires it; credentials do not belong in this repository.

Grounding and QA task metrics are separate. NeXT-GQA reports multiple metrics; use its grounding metric for a temporal-grounding aggregate. MVP includes multiple subtasks. When computing paper averages, follow the paper's exact metric and subtask aggregation definitions, using results from the same model and frame budget.
