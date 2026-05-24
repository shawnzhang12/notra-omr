# Reproduce Step 3 Outputs

```bash
# Validate a correct score IR
uv run notra validate artifacts/executive/step-3/examples/01_valid_four_quarters.ir.json

# Convert valid IR to MusicXML
uv run notra convert artifacts/executive/step-3/examples/01_valid_four_quarters.ir.json --to musicxml --out artifacts/executive/step-3/examples/03_valid_four_quarters.musicxml

# Validate an intentionally invalid score (underfilled 4/4)
uv run notra validate artifacts/executive/step-3/examples/04_invalid_underfill.ir.json
```
