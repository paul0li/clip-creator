# Cut Selected Segments into Separate Video Files

## Context / Problem

Once segment selection produces 3 timestamp ranges, someone still has to manually open the source video, find each timestamp, and export clips. This is mechanical work that adds delay and introduces human error (wrong timestamps, missing audio, etc.).

## Objective

Given the source video file and 3 timestamp ranges, automatically cut and export 3 independent video files in the original aspect ratio (16:9). No format conversion happens in this step.

## Scope

**In scope:**
- Receive source video path + 3 `[start, end]` timestamp pairs (output of segment selection).
- Cut video using FFmpeg (stream copy where possible for speed; re-encode only if needed for clean cuts).
- Export 3 files to a defined output directory with predictable naming (e.g., `clip_1.mp4`, `clip_2.mp4`, `clip_3.mp4`).
- Preserve original resolution, codec, and audio.

**Out of scope:**
- Vertical (9:16) conversion -- handled in the vertical conversion step.
- Subtitles -- handled in the subtitle burn-in step.
- Delivery -- handled in the WhatsApp delivery step.
- Any re-encoding beyond what is needed for frame-accurate cuts.

## Acceptance Criteria

- [ ] Produces exactly 3 video files from the source video.
- [ ] Each clip matches its specified timestamps (tolerance: < 0.5 seconds).
- [ ] No audio loss, corruption, or sync drift.
- [ ] Output files are saved to a configurable directory.
- [ ] The script can be called from n8n (CLI with arguments or HTTP endpoint).

## Implementation Notes

- FFmpeg is the expected tool. Use `-ss` before `-i` for fast seeking; use `-c copy` when keyframe alignment allows, otherwise re-encode with a fast preset.
- Wrap in a simple script (Python or shell) that accepts a JSON input with the video path and timestamp list.
- Keep the script stateless: input in, files out.
