"""Token linearization helpers for Notra score IR."""

from __future__ import annotations

from notra.ir.note import Note
from notra.ir.rest import Rest
from notra.ir.score import Score


def linearize(score: Score) -> list[str]:
    """Linearize score into coarse tokens for debugging and eval baselines."""
    tokens: list[str] = ["<BOS>", f"<score:{score.id}>"]

    for part in score.parts:
        tokens.append(f"<part:{part.id}>")
        for measure in part.measures:
            tokens.append(f"<measure:{measure.number}>")
            for voice in measure.voices:
                tokens.append(f"<voice:{voice.id}>")
                for event in voice.events:
                    if isinstance(event, Note):
                        pitch = event.pitch
                        duration = event.duration
                        tokens.append(
                            f"note:{pitch.step}{pitch.alter:+d}:{pitch.octave}:{duration.numerator}/{duration.denominator}"
                        )
                    elif isinstance(event, Rest):
                        duration = event.duration
                        tokens.append(f"rest:{duration.numerator}/{duration.denominator}")

    tokens.append("<EOS>")
    return tokens
