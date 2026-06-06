# Duration Evidence Methodology

This stage keeps duration recognition as a constrained inference problem:
visual evidence proposes candidates, and the measure solver chooses a legal
assignment when possible.

## Pipeline

1. Detect layout first: staff bands, systems, barlines, and measure boundaries.
2. Detect local symbol evidence: notehead fill state, rests, stems, beams or flags,
   accidentals, and augmentation dots.
3. Convert local evidence into ranked duration candidates:
   - open head, no stem -> whole
   - open head with stem -> half
   - filled head with stem -> quarter
   - filled head with one beam or flag -> eighth
   - filled head with two beams or flags -> 16th
   - detected augmentation dot -> dotted variant candidates
4. Keep fallbacks in the lattice because local evidence is noisy. A false beam
   must not force an eighth note if measure duration proves a quarter is needed.
5. Decode each system-local measure independently. Measures are keyed by
   `(system_index, measure_number)` so same-number measures on different systems
   never get mixed.
6. Validate after inference using MusicXML only as ground truth for scoring, not
   as an oracle during duration selection.

## Current Cello Validation

Command:

```bash
uv run python scripts/eval_cello_duration_policy.py
```

Default scope excludes `untitled_score`.

Current result:

- valid measures: `242/247`
- valid measures excluding `lista-trio_sonata_in_d_minor`: `204/204`
- all-valid pages: `16/17`
- flat exact duration pages: `0/17`
- predicted events vs ground truth events: `1073/967`

The remaining invalid measures are in `lista-trio_sonata_in_d_minor`. They are
not solved by changing local duration rules alone:

- two measures have two selected quarter-like events for a 3/4 measure and are
  missing an eighth-note event
- one measure has no assigned events in the detected system/measure

Adding fake gap rests would make the measure-validity number look better while
hiding upstream note/rest detection errors. The next correct step is better
event selection and symbol grouping, not an unconditional duration filler.

## Event-Selection Finding

The first event-grouping fix corrected `merge_bisected_components`, which was
comparing an x-coordinate against a y-coordinate when checking vertical gaps.
That bug prevented staff-line-split notehead halves from merging.

A simulated structural-opening cutoff improves page-level event-count deltas,
but does not improve exact measure sequences and removes real early candidates
from under-detected pages. Do not add a blanket opening cutoff yet; make the
next candidate-selection pass measure-aware.
