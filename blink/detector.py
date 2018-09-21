from imutils import face_utils
from scipy.spatial import distance as dist
from skimage.color import rgb2grey
from skimage import img_as_ubyte
import dlib
import os
import scipy.signal as sci
import skvideo.io
from .utils import *
import numpy as np
import warnings


class Detector:
    def __init__(self, predictor=None):
        if predictor is None:
            predictor = os.path.split(__file__)[0] + "/data/predictor68.dat"

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor)
        self.left_eye_landmarks = slice(*face_utils.FACIAL_LANDMARKS_IDXS["left_eye"])
        self.right_eye_landmarks = slice(*face_utils.FACIAL_LANDMARKS_IDXS["right_eye"])
        self.EAR_THRESH = 0.2
        self.CONS_FRAME_THRESH = 2
        self.EAR_TOL = 0.01
        self.EYE_PEAK_CONST = 4
        self.EYE_RUNNING_MEAN_WIDTH = 2
        self.EYE_THRESH = 0.0425
        self.supported_types = {'.mpg', '.mp4', '.avi'}

        self.blink_found = False
        self.last_ear = 0
        self.counter = 0
        self.consecutive_closed_eyes = 0

    def clear(self):
        self.blink_found = False
        self.last_ear = 0
        self.counter = 0
        self.consecutive_closed_eyes = 0

    def _get_blink_start_end_(self, points, signal):

        start_end = {"start": [], "end": []}

        for index in range(len(points)):

            if signal[points[index] - 1] <= signal[points[index]]:  # if start point is a peak
                start_index = points[index]
            else:
                left_side = sci.find_peaks(signal[:points[index] + 1], (min(signal) + max(signal)) / 2, None, 1)

                if len(left_side[0]) == 0:
                    start_index = 0
                else:
                    start_index = left_side[0][-1]

            if index != len(points) - 1:
                right_side = sci.find_peaks(signal[points[index]:points[index + 1]],
                                            (min(signal) + max(signal)) / 2, None, 1)
            else:
                right_side = sci.find_peaks(signal[points[index]:], (min(signal) + max(signal)) / 2, None, 1)

            if len(right_side[0]) == 0:  # if no peak on right side
                if index != len(points) - 1:
                    continue
                else:
                    end_index = len(signal) - 1
            else:
                end_index = points[index] + right_side[0][0]

            start_end["start"].append(start_index)
            start_end["end"].append(end_index)

        return start_end

    @staticmethod
    def get_frame_rate(video_file):
        return skvideo.io.ffprobe(video_file)["video"]["@r_frame_rate"]

    @staticmethod
    def get_ear(eye_landmarks):
        vert_dist1 = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        vert_dist2 = dist.euclidean(eye_landmarks[2], eye_landmarks[4])

        hor_dist = dist.euclidean(eye_landmarks[0], eye_landmarks[3])

        ear = (vert_dist1 + vert_dist2) / (2.0 * hor_dist)

        return ear

    @staticmethod
    def parse_landmarks_file(landmarks_file, landmark_type='2d'):
        video_landmarks = []

        no_coordinates = 2
        if landmark_type == '3d':
            no_coordinates += 1

        with open(landmarks_file, 'rt', encoding="ascii") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')

            for frame_no, landmarks in enumerate(csvreader):
                frame_landmarks = np.zeros([68, 2])
                for point in range(1, len(landmarks), no_coordinates):
                    frame_landmarks[point // no_coordinates, 0] = int(landmarks[point + no_coordinates - 1])
                    frame_landmarks[point // no_coordinates, 1] = int(landmarks[point + no_coordinates - 2])

                    if int(landmarks[point]) == -1:
                        return []
                video_landmarks.append(frame_landmarks)

        return video_landmarks

    def get_avg_ear(self, landmarks):

        left_eye_lmks = landmarks[self.left_eye_landmarks]
        right_eye_lmks = landmarks[self.right_eye_landmarks]

        left_ear = self.get_ear(left_eye_lmks)
        right_ear = self.get_ear(right_eye_lmks)

        ear = (left_ear + right_ear) / 2.0

        return ear

    def process_frame(self, frame):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grey_frame = img_as_ubyte(rgb2grey(frame))

        # detect the face
        rects = self.detector(grey_frame, 0)
        landmarks = self.predictor(grey_frame, rects[0])
        landmarks = face_utils.shape_to_np(landmarks)
        avg_ear = self.get_avg_ear(landmarks)

        self.counter += 1
        if self.blink_found:
            if avg_ear - self.last_ear > self.EAR_TOL and avg_ear > self.EAR_THRESH:
                duration = self.counter
                self.clear()
                return duration
        else:
            if avg_ear < self.EAR_THRESH:
                self.consecutive_closed_eyes += 1
            else:
                self.consecutive_closed_eyes = 0
                if abs(self.last_ear - avg_ear) < self.EAR_TOL or avg_ear > self.last_ear:
                    self.counter = 0

            if self.consecutive_closed_eyes >= self.CONS_FRAME_THRESH:
                self.blink_found = True

        self.last_ear = avg_ear

        return None

    def process_dir(self, input_dir, csv_name='default.csv'):

        with open(csv_name, 'w', newline='') as csvfile:

            newcsv = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for root, dirs, filenames in os.walk(input_dir):
                for filename in filenames:
                    extension = os.path.splitext(filename)[1]
                    if extension in self.supported_types:
                        blink_start_end = self.process_video(video_file=os.path.join(root, filename))

                        if len(blink_start_end["start"]) == 0:
                            temp_row = [0, 0, filename]
                            newcsv.writerow(temp_row)

                        else:
                            for index in range(len(blink_start_end["start"])):
                                temp_row = [blink_start_end["start"][index], blink_start_end["end"][index], filename]
                                newcsv.writerow(temp_row)

    def process_video(self, video_file=None, csv_file=None, landmark_type='2d'):

        video = skvideo.io.vread(video_file)

        e_ratio = []
        frames = []
        prev_rects = None

        if csv_file is None:  # process through video
            for frame_no in range(video.shape[0]):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    grey_frame = img_as_ubyte(rgb2grey(video[frame_no]))

                # detect the face
                rects = self.detector(grey_frame, 0)

                try:
                    landmarks = self.predictor(grey_frame, rects[0])
                    prev_rects = rects
                except IndexError:
                    if prev_rects is not None:
                        landmarks = self.predictor(grey_frame, prev_rects[0])
                    else:
                        frames.append(frame_no)
                        e_ratio.append(0)
                        continue

                landmarks = face_utils.shape_to_np(landmarks)

                ear = self.get_avg_ear(landmarks)

                frames.append(frame_no)
                e_ratio.append(ear)
        else:  # get landmarks from the csv file
            all_landmarks = self.parse_landmarks_file(csv_file, landmark_type)
            for landmarks in all_landmarks:
                ear = self.get_avg_ear(landmarks)
                e_ratio.append(ear)

        der = np.array(get_derivative(e_ratio))  # use derivative
        der = np.negative(der)  # use negative of the derivative

        points = sci.find_peaks(der, self.EYE_THRESH, None, self.EYE_PEAK_CONST)
        return self._get_blink_start_end_(points[0], e_ratio)
