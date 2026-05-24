# Notra Glyph-Link Viewer

A zero-build, plain JavaScript viewer that uses Verovio WebAssembly for rendering and links rendered glyph groups back to MusicXML note IDs.

## Run

```bash
python -m http.server 8787
# then open http://localhost:8787/viewer/
```

The default loaded file is:

`artifacts/executive/step-7/examples/02_complex_all_features.musicxml`

## Why this viewer is fast

- No bundler, no transpilation, no framework runtime.
- Single-page plain JS app with direct DOM updates.
- Verovio WASM handles rendering in-browser.

## Glyph linking strategy

1. Primary: direct SVG ID match to MusicXML note `id` attributes.
2. Fallback: sequential mapping of note-like SVG groups to parsed MusicXML note order.

This is enough for deterministic debugging and manual validation loops.

## Duration display

The table shows durations as:

`<musicxml-type><dots> (<fraction-of-whole-note>)`

Example: `quarter (1/4)`, `eighth. (3/16)`.

Fractions are computed from MusicXML `<duration>` units and active `<divisions>` per measure.
