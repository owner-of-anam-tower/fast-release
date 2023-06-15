import math
import time


# faceLms : landmark {
#   x: 0.5236276984214783
#   y: 0.5092944502830505
#   z: -0.09080404788255692
# } ... landmark{}...
#
# hairLineLms : [[h1,w1],[h2,w2] .... [h7,w7]] 좌표값으로 구성되어있음

# face_right = [389,264,447,366,401,435]
# face_left = [162,127,234,93,227,177,58]


def calculate_rectangle_area(coordinates):
    x_values = [coordinate[0] for coordinate in coordinates]
    y_values = [coordinate[1] for coordinate in coordinates]

    width = max(x_values) - min(x_values)
    height = max(y_values) - min(y_values)

    area = width * height
    return area


def calculate_inner_area(points):
    n = len(points)
    area = 0

    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1

    return abs(area) / 2


class FaceRatio:
    def __init__(self, media_face_lms, dlib_face_lms, hair_line_lms, face_right, face_left, image_width, image_height):
        self.media_face_lms = media_face_lms
        self.dlib_face_lms = dlib_face_lms
        self.hairLineLms = hair_line_lms
        self.face_right = face_right
        self.face_left = face_left
        self.face_width = face_right - face_left
        self.iw = image_width
        self.ih = image_height

    def temple(self):  # 눈옆 여백
        eyetail = (self.media_face_lms.landmark[130].x * self.iw - self.face_left) + (
                self.face_right - self.media_face_lms.landmark[359].x * self.iw)
        temple_ratio = eyetail / self.face_width
        temple_wide = temple_ratio > 0.33 if True else False
        return {'templeWide': temple_wide, 'templeWidth': eyetail, 'templeRatio': temple_ratio}

    def height_three_part(self):  # 상안부 중안부 하안부
        upper_x, upper_y = self.hairLineLms[3][1], sorted(self.hairLineLms)[0][0]

        middle_x, middle_y = (self.media_face_lms.landmark[9].x * self.iw + self.media_face_lms.landmark[
            8].x * self.iw) // 2, (
                                     self.media_face_lms.landmark[9].y * self.iw + self.media_face_lms.landmark[
                                 8].y * self.ih) // 2
        lower_x, lower_y = self.media_face_lms.landmark[2].x * self.iw, self.media_face_lms.landmark[2].y * self.ih
        bottom_x, bottom_y = self.media_face_lms.landmark[152].x * self.iw, self.media_face_lms.landmark[
            152].y * self.ih

        # 좌표간의 거리
        top_face = math.hypot(upper_x, upper_y, middle_x, middle_y)
        middle_face = math.hypot(middle_x, middle_y, lower_x, lower_y)
        bottom_face = math.hypot(lower_x, lower_y, bottom_x, bottom_y)

        max_proportion = max(top_face, middle_face, bottom_face)

        face_proportion_base = top_face if top_face == max_proportion else (
            middle_face if middle_face == max_proportion else bottom_face)

        return {'upperFace': round(top_face / face_proportion_base, 2),
                'midFace': round(middle_face / face_proportion_base, 2),
                'lowerFace': round(bottom_face / face_proportion_base, 2)}

    def wh_ratio(self):
        face_height = self.media_face_lms.landmark[152].y * self.ih - sorted(self.hairLineLms)[0][0]
        return {'faceWidth': 1, 'faceHeight': face_height / self.face_width}

    def eye_face_ratio(self):
        left_eye_area = 0
        for points in [[36, 37, 41], [37, 40, 41], [37, 38, 40], [38, 39, 40]]:
            li_l = []
            for p in points:
                li_l.append(self.__dlib_point(p))
            left_eye_area += calculate_inner_area(li_l)

        right_eye_area = 0

        for points in [[42, 43, 47], [43, 44, 47], [47, 44, 46], [44, 45, 46]]:
            li_r = []
            for p in points:
                li_r.append(self.__dlib_point(p))
            right_eye_area += calculate_inner_area(li_r)

        eye_area = right_eye_area + left_eye_area

        face_area = calculate_rectangle_area(
            [[self.face_left, self.dlib_face_lms.part(36).y], [self.face_right, self.dlib_face_lms.part(45).y],
             self.__dlib_point(1), self.__dlib_point(15)])

        for i in range(1, 7):
            face_area += calculate_rectangle_area(
                [self.__dlib_point(i), self.__dlib_point(16 - i), self.__dlib_point(i + 1), self.__dlib_point(15 - i)])

        face_area += calculate_inner_area([self.__dlib_point(7), self.__dlib_point(8), self.__dlib_point(9)])
        return {'eyesRatio': eye_area / face_area}


    def __dlib_point(self, p):
        return [self.dlib_face_lms.part(p).x, self.dlib_face_lms.part(p).y]