import unittest

from packages.core.fusion import fuse_segments
from packages.core.schemas import TranscriptSegment, TranscriptWord, VisionEmotion, VoiceEmotion


def make_scores(top_label: str):
    keys = [
        "joy",
        "sadness",
        "anger",
        "fear",
        "disgust",
        "surprise",
        "confidence",
        "anxiety",
        "boredom",
        "engagement",
        "confusion",
        "frustration",
        "neutral",
    ]
    out = {k: 0.0 for k in keys}
    out[top_label] = 1.0
    return out


class FusionTests(unittest.TestCase):
    def test_fusion_prefers_multimodal_agreement(self):
        seg = TranscriptSegment(
            start=0.0,
            end=2.0,
            text="This is unacceptable and I am angry.",
            speaker_id="spk_a",
            words=[TranscriptWord(text="angry", start=1.0, end=1.4)],
        )

        voice = VoiceEmotion(
            label="anger",
            confidence=0.9,
            scores=make_scores("anger"),
            cues=["high pitch"],
            pitch_hz=260.0,
            energy=0.8,
            speaking_rate_wps=3.0,
            pause_ratio=0.1,
        )
        vision = VisionEmotion(
            label="anger",
            confidence=0.8,
            scores=make_scores("anger"),
            face_cues=["brow furrow"],
            gesture_cues=["jerky motion"],
            smile=0.1,
            brow_furrow=0.8,
            eye_openness=0.5,
            gesture_intensity=0.7,
        )

        fused = fuse_segments([seg], [voice], [vision], taxonomy="ekman")
        self.assertEqual(fused[0].emotion_label, "anger")
        self.assertGreaterEqual(fused[0].confidence, 0.6)


if __name__ == "__main__":
    unittest.main()
