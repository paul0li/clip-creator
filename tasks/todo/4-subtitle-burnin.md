# Card 4: Subtitle Generation and Burn-in Render

## Context / Problem

Short-form video performs significantly better with burned-in subtitles. Most viewers watch on mobile with sound off. Currently Su adds subtitles manually in CapCut, which is the most time-consuming editing step per clip.

## Objective

For each of the 3 vertical clips (from Card 3), generate accurate Spanish subtitles from the transcript and burn them into the video. Output final rendered videos optimized for mobile readability.

## Scope

**In scope:**
- Extract the transcript segment corresponding to each clip's timestamps (from Card 1 output).
- Generate word-level or phrase-level SRT/ASS subtitle files for each clip.
- Style subtitles for mobile readability: large font, high contrast, centered lower-third positioning, max 2 lines.
- Burn subtitles into each vertical video using FFmpeg.
- Output 3 final rendered videos ready for delivery.

**Out of scope:**
- Transcript generation from scratch (reuse the existing episode transcript).
- Animated or karaoke-style word highlighting -- keep it simple for MVP.
- Thumbnails or cover images.
- Delivery -- handled in Card 5.

## Acceptance Criteria

- [ ] Each clip has accurately timed subtitles matching the spoken audio.
- [ ] Subtitles are readable on a mobile screen (minimum ~48px equivalent at 1080x1920).
- [ ] Subtitles do not overflow the screen or overlap with each other.
- [ ] Output videos are self-contained MP4 files (no external subtitle tracks).
- [ ] The script can be called from n8n.

## Implementation Notes

- Subtitle timing: slice the original transcript timestamps to the clip's time range and re-zero them.
- Use ASS format for styling control (font, size, outline, position) and FFmpeg's `ass` filter for burn-in.
- Default style suggestion: white text, black outline, bold sans-serif font, positioned at ~85% from top.
- Re-encoding is required here since we are overlaying subtitles. Use H.264 CRF ~20 for quality (these are the final outputs).
- Language: all subtitles are in Spanish.
