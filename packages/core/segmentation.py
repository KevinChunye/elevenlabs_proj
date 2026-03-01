from __future__ import annotations

from typing import List

from .schemas import TranscriptSegment, TranscriptWord


PUNCT_BREAKS = {".", "?", "!"}


def _append_token_text(buf: str, token: TranscriptWord) -> str:
    txt = token.text or ""
    if not txt:
        return buf
    if txt in {",", ".", "?", "!", ":", ";"}:
        return buf.rstrip() + txt + " "
    if token.token_type == "punctuation":
        return buf.rstrip() + txt + " "
    return buf + txt + " "


def build_segments(
    words: List[TranscriptWord],
    max_segment_seconds: float = 8.0,
    max_gap_seconds: float = 1.2,
) -> List[TranscriptSegment]:
    if not words:
        return []

    out: List[TranscriptSegment] = []
    current_words: List[TranscriptWord] = []
    current_text = ""

    for idx, word in enumerate(words):
        if not current_words:
            current_words = [word]
            current_text = _append_token_text("", word)
            continue

        prev = current_words[-1]
        gap = max(0.0, word.start - prev.end)
        speaker_changed = (prev.speaker_id or "") != (word.speaker_id or "")
        duration = word.end - current_words[0].start

        force_break = False
        if speaker_changed and len(current_words) >= 1:
            force_break = True
        if gap >= max_gap_seconds:
            force_break = True
        if duration >= max_segment_seconds and prev.text in PUNCT_BREAKS:
            force_break = True

        if force_break:
            out.append(
                TranscriptSegment(
                    start=current_words[0].start,
                    end=current_words[-1].end,
                    text=current_text.strip(),
                    speaker_id=current_words[0].speaker_id,
                    words=current_words,
                )
            )
            current_words = [word]
            current_text = _append_token_text("", word)
        else:
            current_words.append(word)
            current_text = _append_token_text(current_text, word)

    if current_words:
        out.append(
            TranscriptSegment(
                start=current_words[0].start,
                end=current_words[-1].end,
                text=current_text.strip(),
                speaker_id=current_words[0].speaker_id,
                words=current_words,
            )
        )

    return out
