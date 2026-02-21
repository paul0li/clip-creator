# Card 5: WhatsApp Delivery of Final Clips

## Context / Problem

The end-to-end pipeline produces 3 final vertical videos with subtitles. The last mile is getting them to Su (the host) so she can review, choose, and manually upload to social platforms. Email or cloud links add friction; Su's primary communication tool is WhatsApp.

We explicitly do NOT want autoposting to any social platform. The system sends clips to one person for manual review.

## Objective

Send the 3 final videos to Su via WhatsApp, accompanied by a structured message with the rationale for each clip. One recipient, no automation beyond delivery.

## Scope

**In scope:**
- Upload each video to a media host accessible by WhatsApp Business Cloud API (or use the API's built-in media upload).
- Send 3 video messages + 1 summary text message to Su's WhatsApp number.
- Summary message includes: episode name/date, and for each clip: clip number, timestamp range, and the rationale from Card 1.
- Handle API errors gracefully (retry once, then log failure).

**Out of scope:**
- Autoposting to Instagram, TikTok, YouTube, or any other platform.
- Receiving replies or interactive flows (see Backlog Card A: Clip Evaluator).
- Multi-recipient delivery.
- n8n does not eliminate WhatsApp API costs; it only orchestrates the API calls. We accept the WhatsApp Business Cloud API cost for this low-volume use case (3 videos/week to 1 number).

## Acceptance Criteria

- [ ] Su receives 3 video messages on WhatsApp after a pipeline run.
- [ ] Su receives 1 text message summarizing all 3 clips (episode, timestamps, rationales).
- [ ] Videos are playable directly in WhatsApp (no download links).
- [ ] Failed sends are logged with enough detail to debug.
- [ ] No messages are sent to anyone other than the configured recipient.

## Implementation Notes

- WhatsApp Business Cloud API (Meta) is required. Set up a Business Account, register a phone number, and obtain a permanent access token.
- n8n has a WhatsApp Business Cloud node; use it for orchestration.
- Video size limit: WhatsApp supports up to 16 MB per video. If clips exceed this, compress with FFmpeg before sending (lower CRF or resolution).
- Message template: WhatsApp requires pre-approved templates for business-initiated messages. Register a simple template or use the 24-hour session window after Su sends an initial message.
- Store the recipient number and API credentials in n8n environment variables, not in code.
