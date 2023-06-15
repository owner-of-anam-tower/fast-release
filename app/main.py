import uvicorn
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware

from api.root import router
from middlewares.custom_logger import access_control
from middlewares.validate_upload_file import ValidateUploadFile


def include_router(app: FastAPI):
    app.include_router(router)


def add_middleware(app: FastAPI):
    app.add_middleware(middleware_class=BaseHTTPMiddleware, dispatch=access_control)
    app.add_middleware(ValidateUploadFile, max_upload_size=1048576)  # ~50MB

def start_application():
    app = FastAPI()
    include_router(app)
    add_middleware(app)
    return app


app = start_application()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

