import unittest
from blink.detector import Detector


class BlinkTester(unittest.TestCase):
    def test_blinks(self):
        det = Detector(predictor="../blink/data/predictor68.dat")
        blinks = det.process_video('../example/example.mp4')
        self.assertEqual(len(blinks), 2)


if __name__ == '__main__':
    unittest.main()
