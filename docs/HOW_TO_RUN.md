# How To Run ByteSeed

Use Windows PowerShell from the project directory.

```powershell
cd D:\ByteSeed
.\.venv\Scripts\activate
python chat.py
```

The root launcher auto-selects the best available checkpoint. The current stable checkpoint is:

```text
checkpoints/anchor_v2_2_finetuned.pt
```

To explicitly use it:

```powershell
python chat.py --checkpoint checkpoints\anchor_v2_2_finetuned.pt
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

Default history mode is off.
