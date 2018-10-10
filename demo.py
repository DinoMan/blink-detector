import argparse
from blink.detector import Detector
from blink import utils

parser = argparse.ArgumentParser()

parser.add_argument("--video", "-v", help="folder containing input videos")
parser.add_argument("--output", "-o", default="stats.csv", help="location of the output file")
parser.add_argument("--distribution", "-d", help="Path to save distribution histogram image")
args = parser.parse_args()

# Initialize the blink detector
blinks_det = Detector()

# Process all the videos in the directory
blinks_det.process_dir(args.video, csv_name=args.output)

# Calculate the distribution of blinks and save to file
utils.save_blink_distribution(args.output, "Blink Distribution", save_path=args.distribution)
