# How To Run ByteSeed

Use Windows PowerShell from the project directory.

```powershell
cd D:\ByteSeed
.\.venv\Scripts\activate
python chat.py
```

The root launcher auto-selects the best available checkpoint. The current stable checkpoint is:

```text
checkpoints/anchor_v2_3_finetuned.pt
```

Default preset is `precise`. Default history mode is off.

Preset examples:

```powershell
python chat.py
python chat.py --preset balanced
python chat.py --preset creative
```

To explicitly use the stable checkpoint:

```powershell
python chat.py --checkpoint checkpoints\anchor_v2_3_finetuned.pt
```

## Useful Chat Commands

- `/reset`
- `/history`
- `/history on`
- `/history off`
- `/temp <value>`
- `/topk <value>`
- `/max <value>`
- `/raw`
- `/help`
- `/exit`

## Inference Dtype And Benchmarks

Default dtype mode is `auto`: fp16 on CUDA, fp32 on CPU. You can override it:

```powershell
python chat.py --dtype auto
python chat.py --dtype fp32
python chat.py --dtype fp16
```

Benchmark examples:

```powershell
python scripts/benchmark_generation.py --dtype fp32
python scripts/benchmark_generation.py --dtype fp16
python scripts/benchmark_generation.py --dtype auto
```

`torch.compile` is available through `--compile`, but it is optional and experimental.
