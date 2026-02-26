# Cargo Ship Loader — Project Notes

## Python Environment

Always run Python using the `personal` conda environment:

```
conda run -n personal python <script.py>
```

Or for inline commands:

```
conda run -n personal python -c "..."
```

Do **not** invoke bare `python` — the system Python will be missing project dependencies.
