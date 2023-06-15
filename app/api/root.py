from fastapi.routing import APIRouter
from api.router import route_face

router = APIRouter()

router.include_router(route_face.router, prefix="/face", tags=["face"])