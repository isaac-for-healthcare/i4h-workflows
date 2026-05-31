# Agentic Annotator

Verify task success in Arena HDF5 recordings or live Zenoh camera streams using
a VLM endpoint. The default model is `Qwen/Qwen3-VL-8B-Instruct` at
`http://localhost:8000/v1`.

## Setup

```bash
workflows/agentic/annotator/setup.sh
```

Start or reuse the local vLLM server:

```bash
workflows/agentic/annotator/vllm.sh start
workflows/agentic/annotator/vllm.sh status
```

## Offline Annotation

```bash
workflows/agentic/annotator/run.sh \
  --env scissor_pick_and_place \
  offline \
  --hdf5-path recording.hdf5 \
  --sample-frames 5 \
  --output annotations.jsonl
```

The task text is read from `config/environments/<env>.yaml` unless overridden with
`--task-description`. Use `--cameras room,wrist` to restrict HDF5 `obs/<camera>`
streams.

## Live Annotation

```bash
workflows/agentic/annotator/run.sh \
  --env scissor_pick_and_place \
  live \
  --count 10 \
  --interval 2.0 \
  --dump-frames-dir annotation_frames \
  --output live_annotations.jsonl
```

Use `--count 0` to run until interrupted. Camera names default to
`zenoh.camera_names` in `config/environments/<env>.yaml`; use `--cameras room,wrist`
to restrict them. Add `--dump-frames-only` with `--dump-frames-dir` to skip the
VLM API call and only save sampled frames for manual/agent inspection.

## Endpoint Options

```bash
workflows/agentic/annotator/run.sh --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --env scissor_pick_and_place \
  offline --hdf5-path path/to/recording.hdf5
```

The endpoint must support OpenAI-compatible chat completions with image inputs.
