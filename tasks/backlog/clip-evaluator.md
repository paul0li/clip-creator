# Clip Evaluation and Progressive Ranking Improvement

## Context / Problem

The MVP selects clips using a static LLM prompt. Without feedback from Su, the system cannot learn which clips actually perform well or match the podcast's editorial voice. Quality plateaus at whatever the initial prompt produces.

## Objective

After receiving the 3 clips, Su rates each one via WhatsApp and assigns a category. The system stores this feedback and uses it to progressively improve segment ranking in future episodes.

## Scope

**In scope:**
- After delivering clips, send a follow-up WhatsApp message asking Su to rate each clip.
- Collect per-clip feedback:
  - Category: funny / controversial / interesting / emotional.
  - Rating: simple scale (e.g., 1-5 or thumbs up/down).
  - Approved or rejected for posting.
- Store feedback alongside the clip's transcript segment and embeddings.
- Adjust the ranking prompt/weights for future episodes based on accumulated feedback:
  - Prioritize similarity to well-rated clips.
  - Penalize patterns associated with rejected clips.
  - Maintain category diversity.

**Out of scope:**
- Fully automated retraining or fine-tuning of models.
- Complex recommendation systems -- keep the feedback loop simple (e.g., few-shot examples from top-rated clips injected into the prompt).

## Acceptance Criteria

- [ ] Su can rate clips and assign categories via WhatsApp without leaving the app.
- [ ] All evaluations are persisted in a queryable store (database or structured file).
- [ ] After 5+ episodes with feedback, clip selection quality shows measurable alignment with Su's preferences.
- [ ] The feedback step does not block or delay the main pipeline.

## Implementation Notes

- WhatsApp interactive messages (buttons/lists) can simplify input, but require approved templates.
- Storage: start with a simple JSON or SQLite database. Migrate to a proper DB only if needed.
- Ranking improvement: inject top-rated clip transcripts as few-shot examples into the segment selection prompt. This is the simplest path before exploring embeddings or vector search.
- Depends on: WhatsApp delivery being operational.
