# notra-omr

<img src="./docs/assets/branding/notra.png" alt="notra logo" width="96" />

MusicXML-first optical music recognition for printed sheet music.

No black-box magic, just inspectable notation engineering.

## Status

`pre-alpha`

This repository is scaffolding toward a full package. Module structure is in place, but release packaging and stable CLI behavior are still in progress.

## Vision

notra is aiming to be the best open-source OMR stack for printed music by focusing on:

- deterministic, auditable outputs
- typed intermediate representations (IR)
- high-fidelity export targets (MusicXML, MEI, JSON IR)
- measurable evaluation with symbolic and visual diffs

## Design Principles

- `MusicXML-first`: align with real engraving and publishing workflows
- `IR-first`: make each stage explicit and testable
- `Debuggable`: preserve artifacts so failures are easy to inspect
- `Regression-driven`: treat quality as a metric, not an impression

## Repository Layout

```text
notra/
  cli/        # command surface (in progress)
  core/       # errors, ids, geometry, provenance
  ir/         # score schema, tokens, validation
  pipeline/   # stage orchestration and artifacts
  layout/     # staff/system/page geometry
  render/     # svg/raster/musicxml rendering helpers
  importers/  # MusicXML/MEI/manual fixtures
  exporters/  # MusicXML/MEI/JSON/Tokens
  eval/       # metric and report utilities
  datasets/   # dataset adapters and prep
  synthetic/  # generated data cases
scripts/      # local workflow scripts
tests/        # unit, roundtrip, render, pipeline tests
viewer/       # optional inspection UI (WIP)
configs/      # runtime and validation config
```

## Planned CLI Surface

```bash
notra recognize <input-image> --out out.musicxml
notra inspect <artifact-dir>
notra diff <score-a.musicxml> <score-b.musicxml>
notra validate <score.musicxml>
notra render <score.musicxml> --overlay
```

This reflects the intended command surface and may change before the first tagged release.

## Development Quickstart

```bash
uv sync --dev

# lint and format
make lint
make fmt

# type checking and tests
make typecheck
make test

# CLI scaffold
uv run notra --help
```

## Roadmap (Near Term)

- [x] Complete `pyproject.toml` package metadata
- [x] Wire a functional CLI entrypoint
- [ ] Add baseline fixture-to-MusicXML recognition pass
- [ ] Ship reproducible benchmark and diff reports
- [ ] Publish contribution and dataset prep guides

## Contributing

Early contributions are welcome, especially around fixtures, validation, rendering correctness, and reproducible evaluation.

## License

Apache-2.0. See `LICENSE`.
