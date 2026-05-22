# Llama 4-bit Runner

For server-scale reproduction, the recommended answer-generation path is a local OpenAI-compatible endpoint backed by vLLM.

## Why

- Do not load a 70B answer model inside every prediction process.
- Start one local inference server on the 4 x RTX 3090 machine.
- Let mini-R-EQA call that server through the `openai_compatible` runner.

## Recommended Launch Pattern

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --host 0.0.0.0 \
  --port 8000
```

Depending on your vLLM version, the exact 4-bit `bitsandbytes` flags may differ. Adjust the command to match the server's installed vLLM release.

## mini-R-EQA Usage

Use:

- `--runner openai_compatible`
- `--model meta-llama/Llama-3.1-70B-Instruct`
- `--base_url http://localhost:8000/v1`

## Important Note

This local 4-bit Llama setting is the intended full-run answer-model setting for this refactor, but it is still a specific deployment variant.

If exact numerical alignment with the original R-EQA answer model matters, you must also match the original answer model family and precision.
