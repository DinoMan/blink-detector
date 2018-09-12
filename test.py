import detector
import numpy as np

obj = detector.BlinkDetector("predictor/shape_predictor_68_face_landmarks.dat")

# read landmarks
landmarks = obj.read_from_csv("example/test.csv")

# process video
frames, e_ratio, points = obj.process_video('example/test.mp4')

# get blink start-end
blink_start_end = obj.get_blink_start_end(points, e_ratio)

# plot
obj.plot_signals( e_ratio, np.negative(obj.get_derivative(e_ratio)), points, frames, [])


