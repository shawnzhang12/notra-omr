# Reproduce Step 7 Outputs

```bash
uv run notra validate artifacts/executive/step-7/examples/00_complex_all_features.ir.json
uv run notra convert artifacts/executive/step-7/examples/00_complex_all_features.ir.json --to musicxml --out artifacts/executive/step-7/examples/02_complex_all_features.musicxml
python -m http.server 8787
# open http://localhost:8787/viewer/
```
