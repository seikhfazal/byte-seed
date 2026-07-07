# Repo Hygiene

- Do not commit checkpoints by default.
- Do not commit `.venv`.
- Do not commit API keys or `.env`.
- Keep large model files in external storage or GitHub Releases later if needed.
- Before the first GitHub push, run:

```powershell
git status
```

Check that `checkpoints/` and `.venv/` are not staged.

The recommended first commit should include code, configs, docs, scripts, and small example data only.
