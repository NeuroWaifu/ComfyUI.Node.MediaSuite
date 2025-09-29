import torch
from .resize_core import ResizeCore


class ImageResize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_mode": (["both", "width", "height", "longest_side", "shortest_side"],),
                "size_mode": (["both", "upscale_only", "downscale_only"],),
                "fill_mode": (["keep_aspect", "fill", "fit", "crop"],),
                "target_width": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 16384,
                    "step": 1
                }),
                "target_height": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 16384,
                    "step": 1
                }),
                "step": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 512,
                    "step": 1
                }),
                "interpolation": (["bilinear", "bicubic", "lanczos", "nearest", "area"],),
            },
            "optional": {
                "border_mode": (["color", "transparent", "blur"],),
                "border_color": ("STRING", {
                    "default": "0,0,0",
                    "multiline": False,
                    "tooltip": "Border color: RGB (255,255,255), RGBA (255,255,255,255), or hex (#RRGGBB, #RRGGBBAA)"
                }),
                "blur_strength": ("FLOAT", {
                    "default": 21.0,
                    "min": 1.0,
                    "max": 101.0,
                    "step": 2.0
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "width", "height", "aspect_ratio", "info")
    FUNCTION = "resize_image"
    CATEGORY = "image/transform"

    def resize_image(self, image, scale_mode, size_mode, fill_mode, target_width, 
                    target_height, step, interpolation, border_mode="color", 
                    border_color="0,0,0", blur_strength=21.0):
        """Resize image"""
        
        batch_size, orig_height, orig_width, channels = image.shape
        
        # Get upscale method
        upscale_method = ResizeCore.get_upscale_method(interpolation)
        
        # Calculate target dimensions
        final_width, final_height = ResizeCore.calculate_dimensions(
            orig_width, orig_height, target_width, target_height,
            scale_mode, size_mode, fill_mode, step
        )
        
        # Process images
        output, mask, actual_width, actual_height = ResizeCore.resize_with_padding(
            image, final_width, final_height, fill_mode, border_mode,
            border_color, blur_strength, upscale_method
        )
        
        # Calculate aspect ratio
        aspect_ratio = actual_width / actual_height
        
        # Create info string
        info_parts = [
            f"Size: {actual_width}x{actual_height}",
            f"Aspect: {aspect_ratio:.3f}",
            f"Mode: {scale_mode}/{fill_mode}",
            f"Original: {orig_width}x{orig_height}"
        ]
        
        if border_mode != "color" and fill_mode == "fit":
            info_parts.append(f"Border: {border_mode}")
        
        info = " | ".join(info_parts)
        
        return (output, mask, actual_width, actual_height, aspect_ratio, info)