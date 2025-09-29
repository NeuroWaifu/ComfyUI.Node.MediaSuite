from .image_resize import ImageResize
from .video_resize import VideoResize
from .resize_calculator import ResizeCalculator
from .mask_operations import MaskOperations, MaskBlur, MaskThreshold
from .dominant_color import DominantColor, ColorPalette
from .tone_masks import ShadowsHighlightsMidtones, ToneCurve, ColorGrading
from .video_upscale import VideoUpscaleWithModel
from .video_info import GetVideoSize, GetVideoInfo, CompareVideoSize
from .math_nodes import (
    MathOperation, MathFunction, BooleanLogic, CompareNumbers,
    ConvertNumber, RandomNumber, Clamp, Remap, Smooth, Constants
)


NODE_CLASS_MAPPINGS = {
    "ImageResize": ImageResize,
    "VideoResize": VideoResize,
    "ResizeCalculator": ResizeCalculator,
    "MaskOperations": MaskOperations,
    "MaskBlur": MaskBlur,
    "MaskThreshold": MaskThreshold,
    "DominantColor": DominantColor,
    "ColorPalette": ColorPalette,
    "ShadowsHighlightsMidtones": ShadowsHighlightsMidtones,
    "ToneCurve": ToneCurve,
    "ColorGrading": ColorGrading,
    "VideoUpscaleWithModel": VideoUpscaleWithModel,
    "GetVideoSize": GetVideoSize,
    "GetVideoInfo": GetVideoInfo,
    "CompareVideoSize": CompareVideoSize,
    "MathOperation": MathOperation,
    "MathFunction": MathFunction,
    "BooleanLogic": BooleanLogic,
    "CompareNumbers": CompareNumbers,
    "ConvertNumber": ConvertNumber,
    "RandomNumber": RandomNumber,
    "Clamp": Clamp,
    "Remap": Remap,
    "Smooth": Smooth,
    "Constants": Constants,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageResize": "Image Resize",
    "VideoResize": "Video Resize",
    "ResizeCalculator": "Resize Calculator",
    "MaskOperations": "Mask Operations",
    "MaskBlur": "Mask Blur",
    "MaskThreshold": "Mask Threshold",
    "DominantColor": "Dominant Color",
    "ColorPalette": "Color Palette",
    "ShadowsHighlightsMidtones": "Shadows/Highlights/Midtones",
    "ToneCurve": "Tone Curve",
    "ColorGrading": "Color Grading",
    "VideoUpscaleWithModel": "Upscale Video (using Model)",
    "GetVideoSize": "Get Video Size",
    "GetVideoInfo": "Get Video Info",
    "CompareVideoSize": "Compare Video Size",
    "MathOperation": "Math Operation",
    "MathFunction": "Math Function",
    "BooleanLogic": "Boolean Logic",
    "CompareNumbers": "Compare Numbers",
    "ConvertNumber": "Convert Number",
    "RandomNumber": "Random Number",
    "Clamp": "Clamp",
    "Remap": "Remap",
    "Smooth": "Smooth",
    "Constants": "Constants",
}


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']