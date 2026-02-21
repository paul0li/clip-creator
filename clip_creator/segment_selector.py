"""Asks the LLM to pick the best moments from a transcript."""

from __future__ import annotations

import json

from clip_creator.config import Config
from clip_creator.llm_client import LLMClient
from clip_creator.models import (
    CandidateSegment,
    LLMError,
    TopicBoundary,
    Transcript,
)

SYSTEM_PROMPT = """\
Eres un editor de podcast experto. Tu trabajo es identificar los mejores \
momentos de un episodio para convertirlos en clips cortos para redes sociales.

Reglas:
- Cada segmento debe durar entre {min_seconds} y {max_seconds} segundos.
- Cada segmento debe empezar y terminar en los límites de una oración completa. \
Usa los timestamps de las oraciones del transcript.
- Busca momentos con: opiniones fuertes, humor, datos sorprendentes, o reflexiones \
interesantes.
- Cada segmento debe ser autocontenido — que se entienda sin el contexto del \
episodio completo.
- Los {count} segmentos deben ser sobre temas diferentes.
- Responde SOLAMENTE con JSON válido, sin texto adicional.

Formato de respuesta:
[
  {{"start": 187.3, "end": 218.9, "rationale": "Descripción breve en español..."}},
  ...
]
"""

USER_PROMPT_TEMPLATE = """\
Transcript del episodio:

{formatted_transcript}

{boundaries_section}

Selecciona los {count} mejores segmentos para clips de redes sociales.\
"""


def _format_transcript(transcript: Transcript) -> str:
    lines = []
    for seg in transcript.segments:
        minutes = int(seg.start // 60)
        seconds = int(seg.start % 60)
        lines.append(f"[{minutes:02d}:{seconds:02d}] {seg.text}")
    return "\n".join(lines)


def _format_boundaries(boundaries: list[TopicBoundary]) -> str:
    if not boundaries:
        return (
            "No se detectaron cambios de tema. "
            "Infiere los temas a partir del contenido."
        )
    lines = ["Cambios de tema detectados (timestamps donde suena el jingle):"]
    for b in boundaries:
        minutes = int(b.timestamp // 60)
        seconds = int(b.timestamp % 60)
        lines.append(f"  - [{minutes:02d}:{seconds:02d}] (confianza: {b.confidence:.2f})")
    return "\n".join(lines)


def _parse_segments(raw: str) -> list[CandidateSegment]:
    """Try to parse the LLM output as a list of CandidateSegment."""
    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    data = json.loads(text)
    return [CandidateSegment(**item) for item in data]


def _validate_segments(
    segments: list[CandidateSegment], config: Config
) -> list[str]:
    """Return a list of validation errors (empty = all good)."""
    errors = []
    for i, seg in enumerate(segments):
        duration = seg.end - seg.start
        if duration < config.segments.min_seconds:
            errors.append(
                f"Segmento {i + 1}: dura {duration:.1f}s, mínimo es {config.segments.min_seconds}s"
            )
        if duration > config.segments.max_seconds:
            errors.append(
                f"Segmento {i + 1}: dura {duration:.1f}s, máximo es {config.segments.max_seconds}s"
            )
        if seg.start >= seg.end:
            errors.append(f"Segmento {i + 1}: start ({seg.start}) >= end ({seg.end})")
    if len(segments) != config.segments.count:
        errors.append(
            f"Se esperaban {config.segments.count} segmentos, se recibieron {len(segments)}"
        )
    return errors


def select_segments(
    transcript: Transcript,
    boundaries: list[TopicBoundary],
    config: Config,
    llm_client: LLMClient | None = None,
) -> list[CandidateSegment]:
    """Ask the LLM to pick the best segments. Retries once on failure."""
    client = llm_client or LLMClient(config)

    system = SYSTEM_PROMPT.format(
        min_seconds=config.segments.min_seconds,
        max_seconds=config.segments.max_seconds,
        count=config.segments.count,
    )
    user = USER_PROMPT_TEMPLATE.format(
        formatted_transcript=_format_transcript(transcript),
        boundaries_section=_format_boundaries(boundaries),
        count=config.segments.count,
    )

    for attempt in range(2):
        raw = client.complete(system, user)

        # Try to parse JSON
        try:
            segments = _parse_segments(raw)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            if attempt == 0:
                user = (
                    f"Tu respuesta anterior no fue JSON válido: {e}\n\n"
                    "Por favor responde SOLAMENTE con JSON válido."
                )
                continue
            raise LLMError(f"LLM returned invalid JSON after retry: {e}") from e

        # Validate
        errors = _validate_segments(segments, config)
        if not errors:
            return segments

        if attempt == 0:
            user = (
                "Tu respuesta tuvo estos problemas:\n"
                + "\n".join(f"- {e}" for e in errors)
                + "\n\nPor favor corrige y responde SOLAMENTE con JSON válido."
            )
            continue

        raise LLMError(
            f"LLM segments failed validation after retry: {'; '.join(errors)}"
        )

    # Should not reach here, but just in case
    raise LLMError("Segment selection failed after all attempts")
