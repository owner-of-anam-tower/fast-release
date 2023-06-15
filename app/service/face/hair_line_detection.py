import numpy as np
import mediapipe as mp
import cv2

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import tasks
from mediapipe.python._framework_bindings import image as image_module
from mediapipe.python._framework_bindings import image_frame

_Image = image_module.Image
_ImageFormat = image_frame.ImageFormat

BG_COLOR = (192, 192, 192)  # gray
MASK_COLOR = (255, 255, 255)  # white
OutputType = vision.ImageSegmenterOptions.OutputType
Activation = vision.ImageSegmenterOptions.Activation
VisionRunningMode = vision.RunningMode
BaseOptions = tasks.BaseOptions

# Create the options that will be used for ImageSegmenter
base_options = python.BaseOptions(model_asset_path='hair_segmentation.tflite')
options = vision.ImageSegmenterOptions(base_options=base_options, running_mode=VisionRunningMode.IMAGE,
                                       output_type=OutputType.CATEGORY_MASK)

# Create the image segmenter
segmenter = vision.ImageSegmenter.create_from_options(options)


class HairLineDetection:
    def detect_hair_line_mediapipe(image, faceLms):
        ih, iw, ic = image.shape
        srgb_image = mp.Image(image_format=_ImageFormat.SRGB, data=image)
        rgba_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        # set alpha channel to empty.
        rgba_image[:, :, 3] = 0
        # create MP Image object from numpy array
        image_seg = _Image(image_format=_ImageFormat.SRGBA, data=rgba_image)

        # Retrieve the masks for the segmented image
        category_masks = segmenter.segment(image_seg)

        # Generate solid color images for showing the output segmentation mask.
        image_data = srgb_image.numpy_view()
        fg_image = np.zeros(image_data.shape, dtype=np.uint8)
        fg_image[:] = MASK_COLOR
        bg_image = np.zeros(image_data.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR

        for name in category_masks:
            condition = np.stack((name.numpy_view(),) * 3, axis=-1) > 0.2
            output_image = np.where(condition, fg_image, bg_image)

        foreheadLms_x = list(map(int, [faceLms.landmark[103].x * iw, faceLms.landmark[67].x * iw,
                                       faceLms.landmark[109].x * iw, faceLms.landmark[10].x * iw,
                                       faceLms.landmark[338].x * iw, faceLms.landmark[297].x * iw,
                                       faceLms.landmark[332].x * iw]))
        foreheadLms_y = list(map(int, [faceLms.landmark[103].y * ih, faceLms.landmark[67].y * ih,
                                       faceLms.landmark[109].y * ih, faceLms.landmark[10].y * ih,
                                       faceLms.landmark[338].y * ih, faceLms.landmark[297].y * ih,
                                       faceLms.landmark[332].y * ih]))
        # -2:헤어 외부 / -1:헤어 / 0~ : 높이 좌표
        hairLineLms = []

        for w_idx in range(7):
            w = foreheadLms_x[w_idx]
            for h_idx in range(foreheadLms_y[w_idx], 1, -1):
                b, g, r = output_image[h_idx, w]
                next_b, next_g, next_r = output_image[h_idx - 1, w]
                if b != 255 and next_b == 255:
                    hairLineLms.append([h_idx - 1, w])
                    break
                elif b == 255 and next_b == 255:
                    hairLineLms.append([h_idx - 1, w])
                    break
        return hairLineLms