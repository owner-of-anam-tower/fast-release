import cv2
import mediapipe as mp
import cv2
from imutils.face_utils.helpers import FACIAL_LANDMARKS_IDXS
import dlib
import numpy as np

# https://stackoverflow.com/questions/42009202/how-to-call-a-async-function-contained-in-a-class

class FaceDetection:
    __predictor_model = "shape_predictor_68_face_landmarks.dat"
    __detector = dlib.get_frontal_face_detector()
    __predictor = dlib.shape_predictor(__predictor_model)
    __desiredLeftEye = (0.40, 0.40)
    __desiredFaceWidth = 256
    __desiredFaceHeight = 256

    def __init__(self, image):
        self.image = image
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.rect = self.__detector(self.gray, 2)[0]

    def detect_faces_dlib(self):
        return self.__predictor(self.gray, self.rect)

    def detect_faces_mediapipe(self):
        mpFaceMesh = mp.solutions.face_mesh
        faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)

        imgRGB = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        results = faceMesh.process(imgRGB)

        faceLms = results.multi_face_landmarks[0]
        return faceLms

    def align(self):
        # 랜드 마크 (x, y) 좌표를 NumPy 배열로 변환
        shape = self.__predictor(self.gray, self.rect)
        shapes = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            shapes[i] = (shape.part(i).x, shape.part(i).y)
        # 왼쪽 및 오른쪽 눈 (x, y) 좌표 추출
        (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEyePts = shapes[lStart:lEnd]
        rightEyePts = shapes[rStart:rEnd]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.__desiredLeftEye[0]
        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.__desiredLeftEye[0])
        desiredDist *= self.__desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image

        eyesCenter = (int((leftEyeCenter[0] + rightEyeCenter[0]) // 2),
                      int((leftEyeCenter[1] + rightEyeCenter[1]) // 2))
        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        # update the translation component of the matrix
        tX = self.__desiredFaceWidth * 0.5
        tY = self.__desiredFaceHeight * self.__desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.__desiredFaceWidth, self.__desiredFaceHeight)
        output = cv2.warpAffine(self.image, M, (w, h),
                                flags=cv2.INTER_CUBIC)
        # return the aligned face
        return output

    @staticmethod
    def face_end_point_dlib(dlib_landmarks):
        dlib_face_left = min(dlib_landmarks.part(0).x, dlib_landmarks.part(1).x)
        dlib_face_right = max(dlib_landmarks.part(15).x, dlib_landmarks.part(16).x)
        return [dlib_face_left, dlib_face_right]