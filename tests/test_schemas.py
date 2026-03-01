import unittest

from packages.core.schemas import FusedEmotionSegment, to_dict


class SchemaTests(unittest.TestCase):
    def test_fused_segment_serialization(self):
        seg = FusedEmotionSegment(
            start=0.0,
            end=1.2,
            speaker_id="spk_a",
            transcript="hello",
            emotion_label="joy",
            pucek_label="joy",
            confidence=0.8,
            valence=0.6,
            arousal=0.7,
            cues={"voice": ["high energy"], "face": [], "gesture": [], "language": []},
            modality_scores={"fused": {"joy": 0.8}},
        )
        payload = to_dict(seg)
        self.assertEqual(payload["emotion_label"], "joy")
        self.assertIn("cues", payload)
        self.assertIn("modality_scores", payload)


if __name__ == "__main__":
    unittest.main()
