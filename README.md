# SAVER

**Less from More: Reinforcing Sparse Video Reasoning from Dense References**

SAVER is a dense-to-sparse post-training framework for video reasoning under limited temporal evidence. This initial release provides evaluation code and links to model checkpoints. Training code will be released in a future update.

## Model checkpoints

| Model | Hugging Face |
| --- | --- |
| SAVER-0.8B | [lmsdss/SAVER-0.8B](https://huggingface.co/lmsdss/SAVER-0.8B) |
| SAVER-2B | [lmsdss/SAVER-2B](https://huggingface.co/lmsdss/SAVER-2B) |
| SAVER-4B | [lmsdss/SAVER-4B](https://huggingface.co/lmsdss/SAVER-4B) |

## Installation

Use Linux with a CUDA-capable GPU. Run commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The launchers configure the vendored evaluation and video-processing module paths. Training libraries such as TRL and DeepSpeed are not required.

## Benchmark data

Prepare the benchmark videos following [data preparation](docs/DATA.md). Set `LMMS_DATA_ROOT` to the directory containing the extracted benchmark video folders:

```bash
export LMMS_DATA_ROOT=/path/to/benchmark/videos
```

The release includes task configurations for Charades-STA, ActivityNet, NeXT-GQA, MVBench, MMVU, LongVideoBench, VideoMMMU, Video-MME, and MVP. Raw videos and training annotations are not included.

## Evaluate SAVER

```bash
MODEL_PATH=lmsdss/SAVER-2B FPS=0.1 NPROC_PER_NODE=1 \
  bash scripts/eval/evaluate.sh
```

`MODEL_PATH` also accepts the 0.8B or 4B checkpoint, or a local model directory. To run a small one-task check:

```bash
EVAL_TASKS=mmvu_val_mc_boxed EVAL_LIMIT=10 \
  bash scripts/eval/evaluate.sh
```

Remove `EVAL_LIMIT` for full benchmark evaluation.

## Evaluate original Qwen3.5
The baseline launcher evaluates the original Qwen3.5 model under the same video sampling setting.
```bash
MODEL_PATH=Qwen/Qwen3.5-2B FPS=0.1 NPROC_PER_NODE=1 \
  bash scripts/eval/evaluate_qwen3_5.sh
```

## Evaluation outputs

Each invocation writes a separate folder under `outputs/eval/` with its configuration, task metrics, generated answers, and frame statistics. `OUTPUT_PATH` overrides the result folder.

## Repository layout

```text
lmms_eval/          Evaluation engine, Qwen3.5 adapter, and benchmark tasks
saver/              Video processing and inference confidence utilities
scripts/eval/       SAVER and original Qwen3.5 evaluation launchers
docs/DATA.md        Benchmark data preparation
requirements.txt    Evaluation dependencies
```

## Acknowledgments and license

This implementation builds on [VideoAuto-R1](https://github.com/IVUL-KAUST/VideoAuto-R1), [Qwen](https://github.com/QwenLM/Qwen3-VL), and [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). Original source notices and the inherited [Apache-2.0 license](LICENSE) are retained. See [third-party notices](THIRD_PARTY_NOTICES.md). Model weights and datasets retain their providers' licenses.
