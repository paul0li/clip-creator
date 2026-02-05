# Clip Detector

## Purpose
Build software that can detect and split clips from a 1-hour stream, understand the content, remove silences, and select the best moments. The user should be able to open the app on a cellphone, review candidate moments for each clip, and choose one to be subtitled in a social media format (TikTok, Shorts, Reels).

## Core Workflow
- Ingest a 1-hour stream.
- Detect clip boundaries using two known audio cues and their patterns.
- Analyze content and remove silences inside each clip.
- Score and surface the best moments per clip.
- User selects a moment on mobile.
- Generate subtitles and export in vertical short-form format.

## Platform Direction (TBD)
- Web app is preferred because it is easiest to open on a cellphone.
- iOS app is still possible and not ruled out.

## Clip-Splitting Approach
- The app listens for two specific sounds.
- It matches their patterns to mark clip boundaries.

## Status
Early concept and requirements definition. Implementation details are still being decided.
