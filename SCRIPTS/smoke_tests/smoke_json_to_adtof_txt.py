import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DATASET_GENERATOR.json_to_adtof_txt import convert_split
from adtof import config
from adtof.ressources import instrumentsMapping


class JsonToAdtofTxtTest(unittest.TestCase):
    def test_vibration_vocabulary_is_separate_from_original_vocabulary(self):
        self.assertEqual(config.VIBRO_LABELS_5, [35, 38, 47, 45, 43])
        self.assertEqual(
            config.VIBRO_LABELS_5TXT, ["KD", "SD", "T12", "T14", "T16"]
        )
        self.assertEqual(config.VIBRO_WEIGHTS_5, [1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertEqual(
            instrumentsMapping.MIDI_VIBRO_5,
            {35: 35, 36: 35, 38: 38, 40: 38, 47: 47, 45: 45, 43: 43},
        )
        self.assertIs(
            instrumentsMapping.VIBRO_MIDI_5, instrumentsMapping.MIDI_VIBRO_5
        )
        self.assertEqual(instrumentsMapping.MIDI_REDUCED_5[45], 47)
        self.assertEqual(instrumentsMapping.MIDI_REDUCED_5[43], 47)

    def test_vibration_classes_are_written_as_separate_midi_pitches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            labels_dir = root / "BASIC_GROOVE" / "LABELS"
            labels_dir.mkdir(parents=True)

            source_path = labels_dir / "synthetic.json"
            source_path.write_text(
                json.dumps(
                    {
                        "events": [
                            {"timestamp": 0.0, "class": "KD"},
                            {"timestamp": 0.1, "class": "SD"},
                            {"timestamp": 0.2, "class": "T12"},
                            {"timestamp": 0.2, "class": "T14"},
                            {"timestamp": 0.2, "class": "T16"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            converted = convert_split("BASIC_GROOVE", root)

            self.assertEqual(len(converted), 1)
            output_lines = converted[0].output_path.read_text(
                encoding="utf-8"
            ).splitlines()
            pitches = [int(line.split("\t")[1]) for line in output_lines]
            self.assertEqual(pitches, [35, 38, 47, 45, 43])


if __name__ == "__main__":
    unittest.main()
