import imutils
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from starlette import status
from starlette.responses import JSONResponse
import cv2
import numpy as np

from schemas.face import ResponseFaceRatio, RequestFaceInfo
from service.face.face_ratio import FaceRatio
from service.face.face_detection import FaceDetection
from service.face.hair_line_detection import HairLineDetection

router = APIRouter(default_response_class=JSONResponse)



@router.post("/analyze")
async def analyze(request_face_info: RequestFaceInfo = Depends(),
                  file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg"]:
        return JSONResponse(status_code=400, content={'message': "지원하는 형식의 이미지가 아닙니다."})

    img_file = await file.read()
    img_np_arr = np.fromstring(img_file, np.uint8)
    img = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)

    # 이미지의 픽셀 조절
    image = imutils.resize(img, width=256)

    # detection 할 이미지 객체 생성, 기울기 조정
    fa = FaceDetection(image)
    face_aligned_img = fa.align()

    # 이미지 detection
    fa_align = FaceDetection(face_aligned_img)
    face_landmark_dlib = fa_align.detect_faces_dlib()
    face_landmark_mediapipe = fa_align.detect_faces_mediapipe()

    # 얼굴 양쪽 끝 좌표, 헤어라인 좌표
    face_end_point = fa_align.face_end_point_dlib(face_landmark_dlib)
    hair_line_landmark = HairLineDetection.detect_hair_line_mediapipe(face_aligned_img, face_landmark_mediapipe)

    face_ratio = FaceRatio(face_landmark_mediapipe, face_landmark_dlib, hair_line_landmark, face_end_point[1], face_end_point[0], 256, 256)

    return ResponseFaceRatio(**face_ratio.temple(),
                             **face_ratio.wh_ratio(),
                             **face_ratio.height_three_part(),
                             **face_ratio.eye_face_ratio(),
                             **{'jawShape': request_face_info.jawShape, 'cheekbone': request_face_info.cheekbone})