from scipy.spatial import distance as dist
from imutils import face_utils
import skvideo.io
import dlib
from skimage.color import rgb2grey
from skimage import img_as_ubyte
import cv2


class blink_detector:
    def __init__(self, predictor=None):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor)
        self.left_eye_landmarks = slice(*face_utils.FACIAL_LANDMARKS_IDXS["left_eye"])
        self.right_eye_landmarks = slice(*face_utils.FACIAL_LANDMARKS_IDXS["right_eye"])
        self.EAR_THRESH = 0.2
        self.CONS_FRAME_THRESH = 2
        self.EAR_TOL = 0.01

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

    def get_avg_ear(self, img, face_bbox):
        landmarks = self.predictor(img, face_bbox)
        landmarks = face_utils.shape_to_np(landmarks)

        left_eye_lmks = landmarks[self.left_eye_landmarks]
        right_eye_lmks = landmarks[self.right_eye_landmarks]

        left_ear = self.get_ear(left_eye_lmks)
        right_ear = self.get_ear(right_eye_lmks)

        ear = (left_ear + right_ear) / 2.0

        return ear

    def process_video(self, video_file):
        video = skvideo.io.vread(video_file)
        fps = blink_detector.get_frame_rate(video_file)

        blink_frames = []
        consecutive_closed_eyes = 0
        blink_found = False
        prev_ear = 0
        blinks = []
        for frame_no in range(video.shape[0]):

            grey_frame = img_as_ubyte(rgb2grey(video[frame_no]))

            # detect the face
            rects = self.detector(grey_frame, 0)
            avg_ear = self.get_avg_ear(grey_frame, rects[0])

            blink_frames.append(frame_no)
            cv2.imshow("image", grey_frame)
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

                print("consecutive_closed_eyes " + str(consecutive_closed_eyes))
                if consecutive_closed_eyes >= self.CONS_FRAME_THRESH:
                    blink_found = True

            prev_ear = avg_ear

        return blinks
