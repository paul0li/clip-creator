# Backlog B: YouTube Live Chat Signals for Segment Selection

## Context / Problem

The podcast streams live on YouTube before publishing the edited episode. The live chat contains real-time audience reactions -- message spikes, laughter emojis, and exclamation patterns -- that correlate with high-engagement moments. This signal is currently unused in clip selection.

## Objective

Incorporate YouTube live chat activity as a complementary signal in the segment ranking process (Card 1), improving detection of "funny" or high-reaction moments that transcript analysis alone may miss.

## Scope

**In scope:**
- Extract YouTube live chat replay with timestamps for a given stream/video.
- Compute a "chat score" per time window:
  - Message volume spikes (messages per second relative to baseline).
  - Density of laughter tokens and emojis (jaja, jajaja, LOL, crying-laughing emoji, etc.).
- Align chat time windows with transcript candidate windows from Card 1.
- Use the chat score as a weighted factor in the ranking (not a replacement for transcript analysis).

**Out of scope:**
- Sentiment analysis or NLP on individual chat messages beyond keyword/emoji matching.
- Real-time processing during the live stream.
- Chat moderation or filtering.

## Acceptance Criteria

- [ ] Chat data is extracted with accurate timestamps for a given YouTube stream.
- [ ] Chat score correlates reasonably with moments that feel high-energy when reviewed manually.
- [ ] At least 1 of the 3 selected clips per episode can be influenced by the chat signal.
- [ ] The chat signal does not dominate selection -- transcript quality remains the primary factor.

## Implementation Notes

- YouTube chat replay can be extracted via `yt-dlp --write-chat` or the YouTube Data API (liveChatMessages endpoint).
- Scoring: simple sliding window (e.g., 30 seconds) counting messages and laughter tokens. Normalize against the episode's average chat rate.
- Integration: pass the chat scores as an additional input to the Card 1 selection prompt or as a post-ranking multiplier.
- Spanish laughter tokens: "jaja", "jajaja", "jsjs", "xd", "xDD". Include relevant emojis.
- Depends on: Card 1 (segment selection) being operational.
