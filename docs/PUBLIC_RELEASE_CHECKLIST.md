# Public Release Checklist

- Run `git status`.
- Run `git status --ignored`.
- Confirm `checkpoints/` is ignored.
- Confirm `.venv/` is ignored.
- Confirm tokenizer binary files are ignored or intentionally released separately.
- Run secret and personal-data scans:

```powershell
git grep -n -i "api_key"
git grep -n -i "password"
git grep -n -i "secret"
git grep -n -i "ghp_"
git grep -n -i "token="
git grep -n -i "phone"
git grep -n -i "email"
git grep -n -i "address"
```

- Confirm README says ByteSeed is a tiny learning project, not a production LLM.
- Confirm demo transcript is honest.
- Decide whether to provide checkpoint/tokenizer files through GitHub Releases later.
- Make public only after v0.3-speed is committed and the final safety scan passes.
