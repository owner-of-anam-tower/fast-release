from pydantic import BaseModel
from enum import Enum
from pydantic.class_validators import Optional


class FaceShape(BaseModel):
    id: int
    userId: str
    upperFace: int
    midFace: int
    lowerFace: int
    templeRatio: int
    templeType: str
    jawShape: str
    jawShapeName: str
    faceWidth: int
    faceLength: int
    faceRatio: int
    cheekbone: bool

    class Config:
        allow_mutation = False
        orm_mode = True


class JawShape(str, Enum):
    triangle = '역삼각형'
    oval = '계란형'
    round = '둥근형'
    pentagon = '각진형'
    square = '사각형'

# class TempleType(str, Enum):
#     wide = "넓음"
#     normal = "보통"
#     narrow = "좁음"


class RequestFaceInfo(BaseModel):
    cheekbone: bool
    jawShape: JawShape

    class Config:
        allow_mutation = False
        orm_mode = True
        use_enum_values = True


class ResponseFaceRatio(BaseModel):
    upperFace: float
    midFace: float
    lowerFace: float
    templeRatio: float
    templeWide: bool
    jawShape: str
    cheekbone: bool
    faceWidth: float
    faceHeight: float
    eyesRatio: float

    class Config:
        allow_mutation = False
        orm_mode = True

#
# class heightRatio(BaseModel):
#     upperFace: float
#     midFace: float
#     lowerFace: float
#
#     class Config:
#         allow_mutation = False
#         orm_mode = True
#
# class temple(BaseModel):
#     templeRatio: float
#     templeType: Optional[str]
#
#     class Config:
#         allow_mutation = False
#         orm_mode = True