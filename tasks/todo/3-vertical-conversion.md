# Convert Clips to Vertical 9:16 Format

## Context / Problem

The 3 clips from clip cutting are in the original 16:9 horizontal format. Reels and short-form platforms require 9:16 vertical video. Today this is done manually in CapCut, adding time and friction.

## Objective

Automatically convert each 16:9 clip into a 9:16 vertical video using a simple, deterministic approach (center crop or blurred-background fill). No subtitles are added in this step.

## Scope

**In scope:**
- Receive 3 horizontal clip files (output of clip cutting).
- Apply one conversion strategy per clip:
  - Option A: Center-crop to 9:16 (loses edges, keeps sharpness).
  - Option B: Blurred/scaled background with original video centered (preserves full frame, adds background).
- Start with one default strategy (recommend Option B for talking-head podcast content).
- Export 3 vertical files (1080x1920) to the output directory.

**Out of scope:**
- Dynamic speaker tracking or face detection -- too complex for MVP.
- Subtitles -- handled in the subtitle burn-in step.
- Delivery -- handled in the WhatsApp delivery step.
- Any manual CapCut or Canva steps.

## Acceptance Criteria

- [ ] Produces 3 vertical (1080x1920) video files from the 3 horizontal inputs.
- [ ] Audio is preserved without sync issues.
- [ ] The conversion strategy is configurable (default: blurred background).
- [ ] Output files are playable on mobile (H.264 + AAC, MP4 container).
- [ ] The script can be called from n8n.

## Implementation Notes

- FFmpeg filter chains can handle both strategies:
  - Blurred background: scale + blur the source to 1080x1920, overlay the original centered.
  - Center crop: `crop=ih*9/16:ih` then scale to 1080x1920.
- Encoding: H.264 with a reasonable CRF (e.g., 23) and AAC audio. Prioritize compatibility over file size.
- Wrap in a script that takes input directory (or file list) and outputs vertical files.
