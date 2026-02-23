"""Asks the LLM to pick the best moments from a transcript.

Uses a windowed approach: split the transcript into ~10-minute windows,
ask the LLM for the best candidate per window, then pick the top N.
This avoids hallucination from sending the full transcript in one prompt.
"""

from __future__ import annotations

import json
import sys

from clip_creator.config import Config
from clip_creator.llm_client import LLMClient
from clip_creator.models import (
    CandidateSegment,
    LLMError,
    TopicBoundary,
    Transcript,
    TranscriptSegment,
    format_timestamp,
)

# --- Per-window prompt: pick the single best moment from this section ---

WINDOW_SYSTEM_PROMPT = """\
Eres un editor de podcast experto. Analiza esta sección del episodio y \
selecciona el MEJOR momento para un clip de redes sociales.

Reglas:
- El segmento debe durar entre {min_seconds} y {max_seconds} segundos.
- Debe empezar y terminar en los límites de una oración completa. \
Usa los timestamps de las oraciones del transcript.
- Busca: opiniones fuertes, humor, datos sorprendentes, reflexiones interesantes.
- Debe ser autocontenido — que se entienda sin el contexto del episodio completo.
- En "rationale", cita textualmente una frase del transcript. No inventes ni parafrasees.
- Siempre selecciona un momento — todos los episodios tienen contenido interesante.
- Responde SOLAMENTE con JSON válido, sin texto adicional.

Formato de respuesta:
[{{"start": "HH:MM:SS", "end": "HH:MM:SS", "rationale": "..."}}]
"""

WINDOW_USER_PROMPT = """\
Sección del episodio ({time_range}):

{formatted_transcript}

Selecciona el mejor momento para un clip de redes sociales.\
"""

# --- Final selection prompt: pick the best N from all candidates ---

FINAL_SYSTEM_PROMPT = """\
Eres un editor de podcast experto. De estos candidatos pre-seleccionados, \
elige los {count} mejores para clips de redes sociales.

Reglas:
- Los {count} segmentos deben ser sobre temas diferentes.
- Prioriza: impacto, variedad temática, y que sean autocontenidos.
- No modifiques los timestamps ni el rationale — cópialos exactos.
- Responde SOLAMENTE con JSON válido, sin texto adicional.

Formato de respuesta:
[{{"start": "HH:MM:SS", "end": "HH:MM:SS", "rationale": "..."}}]
"""

FINAL_USER_PROMPT = """\
Candidatos pre-seleccionados de distintas secciones del episodio:

{candidates_json}

Elige los {count} mejores clips. Responde solo con JSON.\
"""


def _format_segments(segments: list[TranscriptSegment]) -> str:
    return "\n".join(
        f"[{format_timestamp(seg.start)}] {seg.text}" for seg in segments
    )


def _format_boundaries(boundaries: list[TopicBoundary]) -> str:
    if not boundaries:
        return (
            "No se detectaron cambios de tema. "
            "Infiere los temas a partir del contenido."
        )
    lines = ["Cambios de tema detectados (timestamps donde suena el jingle):"]
    for b in boundaries:
        lines.append(f"  - [{format_timestamp(b.timestamp)}] (confianza: {b.confidence:.2f})")
    return "\n".join(lines)


def _parse_segments(raw: str) -> list[CandidateSegment]:
    """Try to parse the LLM output as a list of CandidateSegment."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    data = json.loads(text)
    return [CandidateSegment(**item) for item in data]


def _validate_segment(seg: CandidateSegment, config: Config) -> list[str]:
    """Return validation errors for a single segment."""
    errors = []
    duration = seg.end - seg.start
    if duration < config.segments.min_seconds:
        errors.append(f"dura {duration:.1f}s, mínimo es {config.segments.min_seconds}s")
    if duration > config.segments.max_seconds:
        errors.append(f"dura {duration:.1f}s, máximo es {config.segments.max_seconds}s")
    if seg.start >= seg.end:
        errors.append(
            f"start ({format_timestamp(seg.start)}) >= end ({format_timestamp(seg.end)})"
        )
    return errors


def _split_into_windows(
    transcript: Transcript, window_minutes: int = 10,
) -> list[list[TranscriptSegment]]:
    """Split transcript segments into time-based windows."""
    if not transcript.segments:
        return []

    windows: list[list[TranscriptSegment]] = []
    current: list[TranscriptSegment] = []
    window_start = transcript.segments[0].start

    for seg in transcript.segments:
        if seg.start - window_start >= window_minutes * 60 and current:
            windows.append(current)
            current = []
            window_start = seg.start
        current.append(seg)

    if current:
        windows.append(current)
    return windows


def _window_time_range(window: list[TranscriptSegment]) -> str:
    """Human-readable time range like '00:04:00 – 00:14:00'."""
    return f"{format_timestamp(window[0].start)} – {format_timestamp(window[-1].end)}"


def _nominate_candidate(
    window: list[TranscriptSegment],
    window_index: int,
    total_windows: int,
    config: Config,
    client: LLMClient,
) -> CandidateSegment | None:
    """Ask the LLM for the best candidate from one window. Returns None on failure."""
    time_range = _window_time_range(window)
    label = f"[window {window_index + 1}/{total_windows}] {time_range}"

    system = WINDOW_SYSTEM_PROMPT.format(
        min_seconds=config.segments.min_seconds,
        max_seconds=config.segments.max_seconds,
    )
    user = WINDOW_USER_PROMPT.format(
        time_range=time_range,
        formatted_transcript=_format_segments(window),
    )

    for attempt in range(2):
        try:
            raw = client.complete(system, user)
        except Exception as e:
            print(f"  {label}: LLM call failed: {e}", file=sys.stderr)
            return None

        try:
            segments = _parse_segments(raw)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            if attempt == 0:
                user = (
                    f"Tu respuesta anterior no fue JSON válido: {e}\n\n"
                    "Por favor responde SOLAMENTE con JSON válido."
                )
                continue
            print(f"  {label}: invalid JSON after retry", file=sys.stderr)
            return None

        # Empty array = LLM found nothing (shouldn't happen with new prompt)
        if not segments:
            print(f"  {label}: LLM returned empty array", file=sys.stderr)
            print(f"    raw response: {raw[:200]}", file=sys.stderr)
            return None

        candidate = segments[0]  # We asked for the single best
        errors = _validate_segment(candidate, config)
        if not errors:
            print(
                f"  {label}: candidate {format_timestamp(candidate.start)}"
                f"–{format_timestamp(candidate.end)}",
                file=sys.stderr,
            )
            return candidate

        if attempt == 0:
            user = (
                "Tu respuesta tuvo estos problemas:\n"
                + "\n".join(f"- {e}" for e in errors)
                + "\n\nPor favor corrige y responde SOLAMENTE con JSON válido."
            )
            continue

        print(f"  {label}: validation failed after retry", file=sys.stderr)
        return None

    return None


def _pick_best(
    candidates: list[CandidateSegment],
    config: Config,
    client: LLMClient,
) -> list[CandidateSegment]:
    """Final LLM pass: pick the top N from all per-window candidates."""
    count = config.segments.count

    # If we have exactly the right number (or fewer), skip the final call
    if len(candidates) <= count:
        return candidates

    candidates_json = json.dumps(
        [seg.model_dump(mode="json") for seg in candidates],
        indent=2,
        ensure_ascii=False,
    )

    system = FINAL_SYSTEM_PROMPT.format(count=count)
    user = FINAL_USER_PROMPT.format(
        candidates_json=candidates_json,
        count=count,
    )

    for attempt in range(2):
        try:
            raw = client.complete(system, user)
        except Exception as e:
            raise LLMError(f"Final selection LLM call failed: {e}") from e

        try:
            selected = _parse_segments(raw)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            if attempt == 0:
                user = (
                    f"Tu respuesta anterior no fue JSON válido: {e}\n\n"
                    "Por favor responde SOLAMENTE con JSON válido."
                )
                continue
            raise LLMError(f"Final selection: invalid JSON after retry: {e}") from e

        # Validate each selected segment
        all_valid = True
        for seg in selected:
            errors = _validate_segment(seg, config)
            if errors:
                all_valid = False
                break

        if all_valid and selected:
            return selected[:count]

        if attempt == 0:
            user = (
                "Tu respuesta tuvo problemas de validación. "
                "Recuerda: no modifiques los timestamps de los candidatos.\n\n"
                "Por favor corrige y responde SOLAMENTE con JSON válido."
            )
            continue

        raise LLMError("Final selection failed validation after retry")

    raise LLMError("Final selection failed after all attempts")


def select_segments(
    transcript: Transcript,
    boundaries: list[TopicBoundary],
    config: Config,
    llm_client: LLMClient | None = None,
) -> list[CandidateSegment]:
    """Pick the best segments using a windowed approach.

    1. Split transcript into ~10-minute windows
    2. Ask the LLM for the best candidate per window
    3. Final LLM pass to pick the top N from all candidates
    """
    client = llm_client or LLMClient(config)

    # Step 1: Split into windows
    windows = _split_into_windows(transcript)
    print(
        f"Segment selection: {len(windows)} windows from "
        f"{len(transcript.segments)} sentences",
        file=sys.stderr,
    )

    # Step 2: Nominate one candidate per window
    candidates: list[CandidateSegment] = []
    for i, window in enumerate(windows):
        candidate = _nominate_candidate(
            window, i, len(windows), config, client,
        )
        if candidate is not None:
            candidates.append(candidate)

    print(
        f"Segment selection: {len(candidates)} candidates from "
        f"{len(windows)} windows",
        file=sys.stderr,
    )

    if not candidates:
        raise LLMError("No valid candidates from any window")

    # Step 3: Final selection
    selected = _pick_best(candidates, config, client)
    print(
        f"Segment selection: picked {len(selected)} segments",
        file=sys.stderr,
    )
    return selected
