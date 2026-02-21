# Select 3 Candidate Segments from Transcript

## Context / Problem

"El Club de las Tres de la Tarde" publishes 1-hour podcast episodes. The biggest bottleneck is manually reviewing the full episode to decide which moments to cut into short reels. This step blocks all downstream clip production.

We considered video-only heuristics (energy, silence gaps) but concluded that transcript-based analysis is required -- the podcast's value is conversational, and good clips depend on narrative coherence, not just audio peaks.

## Objective

Given a full episode transcript with timestamps, automatically select 3 candidate segments (25-40 seconds each) and output their start/end timestamps with a short rationale for each. No video processing happens in this step.

## Scope

**In scope:**
- Ingest a full transcript with word-level or sentence-level timestamps.
- Define candidate windows (25-40 seconds).
- Score/rank windows using criteria: clear opening, complete closure, presence of a strong opinion/reflection/humor.
- Select top 3 segments ensuring thematic diversity.
- Output: JSON or structured list with `[start, end, rationale]` for each segment.

**Out of scope:**
- Transcript generation (assumes transcript already exists, e.g., from Whisper or a prior step).
- Video cutting, vertical formatting, subtitles, or delivery.
- Any autoposting to social platforms.

## Acceptance Criteria

- [ ] Receives a transcript (with timestamps) as input and outputs exactly 3 segments.
- [ ] Each segment has a start timestamp, end timestamp, and a 1-2 sentence rationale.
- [ ] Segments do not start or end mid-sentence.
- [ ] Segments are thematically diverse (not 3 clips from the same topic).
- [ ] The process can be triggered from an n8n workflow (CLI script or HTTP-callable).

## Implementation Notes

- Orchestration: n8n triggers a script/service. We chose n8n over OpenClaw because it is simpler for this MVP's volume (1 episode/week, 1 user).
- LLM prompt design is the core work here. Start with a single prompt that receives the transcript and returns structured JSON.
- Consider chunking long transcripts if they exceed context limits.
- Language: episodes are in Spanish; the LLM must handle Spanish transcript content.
