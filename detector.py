from imutils import face_utils
from scipy.spatial import distance as dist
from skimage.color import rgb2grey
from skimage import img_as_ubyte
import csv
import cv2
import dlib
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import progressbar
import scipy.signal as sci
import skvideo.io


class BlinkDetector:
    def __init__(self, predictor):
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

    @staticmethod
    def draw_points(image, points, tag=True, in_place=False, color=(255, 255, 255)):
        font = cv2.FONT_HERSHEY_SIMPLEX

        if in_place:
            img = image
        else:
            img = np.copy(image)

        for i in range(points.shape[0]):
            if tag:
                cv2.putText(img, str(i), (int(points[i, 0]), int(points[i, 1])), font, 0.23, color)
            else:
                cv2.circle(img, (int(points[i, 0]), int(points[i, 1])), 1, color)
        return img

    @staticmethod
    def get_absolute_error(list1, list2):

        err = 0
        if len(list1) != len(list2):
            return -1

        else:
            if len(list1) == 0:
                return 0

            for i in range(len(list1)):
                err += math.fabs(list1[i] - list2[i])
            return err / len(list1)

    @staticmethod
    def get_avg_blink_duration_and_blink_amount(blink_start_end):

        tot_blink_duration = 0
        blink_amount = len(blink_start_end["start"])

        for index in range(len(blink_start_end["start"])):
            tot_blink_duration += blink_start_end["end"][index] - blink_start_end["start"][index]

        if blink_amount != 0:
            avg_blink_duration = tot_blink_duration / blink_amount
        else:
            avg_blink_duration = 0

        return avg_blink_duration, blink_amount

    @staticmethod
    def get_blink_distribution(csv_file, title, display_results=True):

        blink_amounts = {}
        avg_blink_durations = {}
        temp_blink_start_end = {"start": [], "end": []}
        first_flag = True
        video_amount = 1

        with open(csv_file, newline='') as blink_data:
            blink_data_reader = csv.reader(blink_data, delimiter=',', quotechar='|')

            for row in blink_data_reader:

                if first_flag:
                    previous_file = row[-1]
                    first_flag = False

                    temp_blink_start_end["start"].append(int(row[0]))
                    temp_blink_start_end["end"].append(int(row[1]))
                    continue

                current_file = row[-1]

                if current_file == previous_file:
                    temp_blink_start_end["start"].append(int(row[0]))
                    temp_blink_start_end["end"].append(int(row[1]))

                else:
                    video_amount += 1
                    avg_blink_duration, blink_amount = BlinkDetector.get_avg_blink_duration_and_blink_amount(
                        temp_blink_start_end)

                    # debug
                    # print(previous_file + ": " + str(avg_blink_dur) + " " + str(blink_amount))

                    if blink_amount == 1 and avg_blink_duration == 0:

                        if 0 not in blink_amounts:
                            blink_amounts[0] = 1
                        else:
                            blink_amounts[0] += 1

                    else:

                        if blink_amount not in blink_amounts:
                            blink_amounts[blink_amount] = 1
                        else:
                            blink_amounts[blink_amount] += 1

                    avg_blink_duration = round(avg_blink_duration)  # rounding

                    if avg_blink_duration not in avg_blink_durations:
                        avg_blink_durations[avg_blink_duration] = 1
                    else:
                        avg_blink_durations[avg_blink_duration] += 1

                    temp_blink_start_end = {"start": [], "end": []}
                    temp_blink_start_end["start"].append(int(row[0]))
                    temp_blink_start_end["end"].append(int(row[1]))

                previous_file = current_file

        # finish the last file
        avg_blink_duration, blink_amount = BlinkDetector.get_avg_blink_duration_and_blink_amount(temp_blink_start_end)

        # debug
        # print(previous_file + ": " + str(avg_blink_dur) + " " + str(blink_amount))

        if blink_amount == 1 and avg_blink_duration == 0:

            if 0 not in blink_amounts:
                blink_amounts[0] = 1
            else:
                blink_amounts[0] += 1

        else:

            if blink_amount not in blink_amounts:
                blink_amounts[blink_amount] = 1
            else:
                blink_amounts[blink_amount] += 1

        if avg_blink_duration not in avg_blink_durations:
            avg_blink_durations[avg_blink_duration] = 1
        else:
            avg_blink_durations[avg_blink_duration] += 1

        # debug
        # print(video_amount)

        if display_results:
            plt.subplot(2, 1, 1)
            plt.scatter(blink_amounts.keys(), [blink_amounts[key] for key in blink_amounts.keys()], color='b')
            plt.xlabel("Blink Amount per Video")
            plt.ylabel("Count")

            plt.subplot(2, 1, 2)
            plt.scatter(avg_blink_durations.keys(), [avg_blink_durations[key] for key in avg_blink_durations.keys()], color='m')
            plt.xlabel("Avg Blink Duration per Video")
            plt.ylabel("Count")
            plt.savefig(title)
            plt.show()

    @staticmethod
    def get_blink_start_end(points, signal):

        start_end = {"start": [], "end": []}

        for index in range(len(points)):

            if signal[points[index] - 1] <= signal[points[index]]:  # if start point is a peak
                start_index = points[index]
                start_end["start"].append(start_index)
            else:
                left_side = sci.find_peaks(signal[:points[index] + 1], (min(signal) + max(signal)) / 2, None, 1)

                if len(left_side[0]) == 0:
                    start_index = 0
                else:
                    start_index = left_side[0][-1]

                start_end["start"].append(start_index)

            if index != len(points) - 1:
                right_side = sci.find_peaks(signal[points[index]:points[index + 1]],
                                            (min(signal) + max(signal)) / 2, None, 1)

            else:
                right_side = sci.find_peaks(signal[points[index]:], (min(signal) + max(signal)) / 2, None, 1)

            if len(right_side[0]) == 0:  # if no peak on right side
                end_index = len(signal) - 1

            else:
                end_index = points[index] + right_side[0][0]

            start_end["end"].append(end_index)

        return start_end

    @staticmethod
    def get_frame_rate(video_file):
        return skvideo.io.ffprobe(video_file)["video"]["@r_frame_rate"]

    @staticmethod
    def get_derivative(signal):
        return [(signal[i + 1] - signal[i]) for i in range(len(signal) - 1)]

    @staticmethod
    def get_ear(eye_landmarks):
        vert_dist1 = dist.euclidean(eye_landmarks[1, :], eye_landmarks[5, :])
        vert_dist2 = dist.euclidean(eye_landmarks[2, :], eye_landmarks[4, :])

        hor_dist = dist.euclidean(eye_landmarks[0, :], eye_landmarks[3, :])

        ear = (vert_dist1 + vert_dist2) / (2.0 * hor_dist)

        return ear

    @staticmethod
    def get_mean_square_error(list1, list2):

        err = 0
        if len(list1) != len(list2):
            return -1
        for i in range(len(list1)):
            err += (list1[i] - list2[i]) ** 2
        return (err / len(list1)) ** 0.5

    @staticmethod
    def plot_signals(signal, signal_der, detected, frames, ground_truth):
        # Highly specific plotting function to display signal, its derivative and the blink ground truths
        plt.figure(1)
        plt.subplot(2, 1, 1)

        plt.title("Abs Error: " + str(BlinkDetector.get_absolute_error(detected[0], ground_truth)))

        plt.scatter(detected[0], [signal[i] for i in detected[0]], color='r')

        plt.scatter(ground_truth, [signal[i] for i in ground_truth], color='g')
        plt.plot(frames, signal, '.-')
        plt.ylabel('Signal')

        plt.subplot(2, 1, 2)
        plt.scatter(detected[0], detected[1].get('peak_heights'), color='r')
        plt.plot(frames[:-1], signal_der, '.-')
        plt.ylabel('Derivative')
        plt.xlabel('Frame')

        plt.show()

    @staticmethod
    def read_landmarks_from_csv(filename, landmark_type='2d'):

        with open(filename, newline='') as landmarks_file:
            landmark_reader = csv.reader(landmarks_file, delimiter=',', quotechar='|')

            landmarks = []

            for row in landmark_reader:

                frame_landmarks = []

                row = row[1:]

                if landmark_type == '3d':

                    for index in range(0, len(row), 3):

                        coordinates = [int(row[index + 2]), int(row[index + 1])]

                        frame_landmarks.append(coordinates)

                else:

                    for index in range(0, len(row), 2):
                        coordinates = [int(row[index + 1]), int(row[index])]

                        frame_landmarks.append(coordinates)


                landmarks.append(frame_landmarks)

        return np.array(landmarks)

    @staticmethod
    def smooth_signal(x, n):
        return np.convolve(x, np.ones((n,)) / n)[(n - 1):]

    @staticmethod
    def visualize_points(video_file, csv_file, visualize_with_numbers=False, save_video=False,
                         output_filename='default.mp4', landmark_type='2d'):

        cap = cv2.VideoCapture(video_file)
        all_landmarks = BlinkDetector.read_landmarks_from_csv(csv_file, landmark_type)
        frame_no = 0

        if save_video:
            out = skvideo.io.FFmpegWriter(output_filename, inputdict={},
                                          outputdict={'-vcodec': 'libx264', '-pix_fmt': 'rgb24', '-r': '30'})

        while cap.isOpened():

            ret, frame = cap.read()

            if ret:

                preds = all_landmarks[frame_no]
                frame_no += 1

                temp_img = MouthDetector.draw_points(frame, preds, tag=visualize_with_numbers)

                cv2.imshow('Frame', temp_img)
                cv2.waitKey(29)

                if save_video:
                    out.writeFrame(cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB))

            else:
                break

        cap.release()

        if save_video:
            out.close()

    def get_avg_ear(self, landmarks):

        left_eye_lmks = landmarks[self.left_eye_landmarks]
        right_eye_lmks = landmarks[self.right_eye_landmarks]

        left_ear = self.get_ear(left_eye_lmks)
        right_ear = self.get_ear(right_eye_lmks)

        ear = (left_ear + right_ear) / 2.0

        return ear

    def process_frame(self, video_file):
        video = skvideo.io.vread(video_file)
        # fps = BlinkDetector.get_frame_rate(video_file)

        blink_frames = []
        consecutive_closed_eyes = 0
        blink_found = False
        prev_ear = 0
        blinks = []
        for frame_no in range(video.shape[0]):

            grey_frame = img_as_ubyte(rgb2grey(video[frame_no]))

            # detect the face
            rects = self.detector(grey_frame, 0)
            landmarks = self.predictor(grey_frame, rects[0])
            landmarks = face_utils.shape_to_np(landmarks)
            avg_ear = self.get_avg_ear(landmarks)

            blink_frames.append(frame_no)
            cv2.imshow("Image", grey_frame)
            cv2.waitKey()
            if blink_found:
                if avg_ear - prev_ear > self.EAR_TOL and avg_ear > self.EAR_THRESH:
                    blinks.append(blink_frames)
                    blink_frames = []
                    blink_found = False
            else:
                if avg_ear < self.EAR_THRESH:
                    consecutive_closed_eyes += 1
                else:
                    consecutive_closed_eyes = 0
                    if abs(prev_ear - avg_ear) < self.EAR_TOL or avg_ear > prev_ear:
                        blink_frames = []

                print("Consecutive closed eyes: " + str(consecutive_closed_eyes))
                if consecutive_closed_eyes >= self.CONS_FRAME_THRESH:
                    blink_found = True

            prev_ear = avg_ear

        return blinks

    def process_dir(self, input_direc, csv_name='default.csv'):

        current_file_num = 0
        tot_file_num = 0

        with open(csv_name, 'w', newline='') as csvfile:

            newcsv = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            for root, dirs, filenames in os.walk(input_direc):
                for filename in filenames:
                    if filename[-4:] == '.mpg' or filename[-4:] == '.mp4':
                        tot_file_num += 1

            bar = progressbar.ProgressBar(maxval=tot_file_num).start()  # initialize progress bar

            for root, dirs, filenames in os.walk(input_direc):

                dirs[:] = [d for d in dirs if d != 's31']

                for filename in filenames:
                    if filename[-4:] == '.mpg' or filename[-4:] == '.mp4':

                        print(filename)
                        frames, e_ratio, points, blink_start_end = self.process_video(video_file=os.path.join(root, filename))

                        if len(blink_start_end["start"]) == 0:
                            temp_row = [0, 0, filename]

                            newcsv.writerow(temp_row)

                        else:
                            for index in range(len(blink_start_end["start"])):
                                temp_row = [blink_start_end["start"][index], blink_start_end["end"][index], filename]

                                newcsv.writerow(temp_row)

                        current_file_num += 1
                        bar.update(current_file_num)

    def process_video(self, video_file=None, csv_file=None, landmark_type='2d'):

        video = skvideo.io.vread(video_file)

        e_ratio = []
        frames = []
        prev_rects = None
        to_be_filled = False

        if csv_file is None:  # process through video
            for frame_no in range(video.shape[0]):
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
                        to_be_filled = True
                        continue

                landmarks = face_utils.shape_to_np(landmarks)

                ear = self.get_avg_ear(landmarks)

                frames.append(frame_no)

                if to_be_filled:
                    e_ratio.append(ear)
                    to_be_filled = False

                e_ratio.append(ear)

        else:  # get landmarks from the csv file
            all_landmarks = self.read_landmarks_from_csv(csv_file, landmark_type)
            for frame_no in range(all_landmarks.shape[0]):
                landmarks = all_landmarks[frame_no]

                ear = self.get_avg_ear(landmarks)

                frames.append(frame_no)
                e_ratio.append(ear)

        der = np.array(self.get_derivative(e_ratio))  # use derivative
        der = np.negative(der)  # use negative of the derivative

        points = sci.find_peaks(der, self.EYE_THRESH, None, self.EYE_PEAK_CONST)

        return frames, e_ratio, points
