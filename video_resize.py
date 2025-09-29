import torch
import comfy.utils
from fractions import Fraction
from comfy_api.input_impl import VideoFromComponents
from comfy_api.util import VideoComponents
from .resize_core import ResizeCore


class VideoResize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
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
                    "step": 1,
                    "tooltip": "Round dimensions to nearest multiple (use 2 for video compatibility)"
                }),
                "interpolation": (["bilinear", "bicubic", "lanczos", "nearest", "area"],),
                "force_even_dimensions": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Force width and height to be divisible by 2 (required for most video codecs)"
                }),
            },
            "optional": {
                "border_mode": (["color", "blur"],),
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

    RETURN_TYPES = ("VIDEO", "MASK", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("VIDEO", "MASK", "width", "height", "aspect_ratio", "info")
    FUNCTION = "resize_video"
    CATEGORY = "video/transform"

    def resize_video(self, video, scale_mode, size_mode, fill_mode, target_width, 
                    target_height, step, interpolation, force_even_dimensions=True,
                    border_mode="color", border_color="0,0,0", blur_strength=21.0):
        """Resize video"""
        
        # Get video components
        components = video.get_components()
        frames = components.images
        audio = components.audio
        frame_rate = components.frame_rate
        
        # Get dimensions
        batch_size, orig_height, orig_width, channels = frames.shape
        
        # Get upscale method
        upscale_method = ResizeCore.get_upscale_method(interpolation)
        
        # Calculate target dimensions
        final_width, final_height = ResizeCore.calculate_dimensions(
            orig_width, orig_height, target_width, target_height,
            scale_mode, size_mode, fill_mode, step
        )
        
        # For video, ensure dimensions are divisible by 2 (required by most codecs)
        if force_even_dimensions:
            # Make dimensions even by adding 1 if odd
            final_width = final_width + (final_width % 2)
            final_height = final_height + (final_height % 2)
        
        # Process frames
        output_frames, mask, actual_width, actual_height = ResizeCore.resize_with_padding(
            frames, final_width, final_height, fill_mode, border_mode,
            border_color, blur_strength, upscale_method
        )
        
        # For video with keep_aspect mode, ensure output dimensions are even
        if force_even_dimensions and fill_mode == "keep_aspect":
            actual_width = actual_width + (actual_width % 2)
            actual_height = actual_height + (actual_height % 2)
            # Resize output to ensure even dimensions
            if output_frames.shape[2] != actual_width or output_frames.shape[1] != actual_height:
                samples = output_frames.movedim(-1, 1)
                output_frames = comfy.utils.common_upscale(samples, actual_width, actual_height, upscale_method, "disabled")
                output_frames = output_frames.movedim(1, -1)
                # Update mask too
                mask = torch.ones((frames.shape[0], actual_height, actual_width), dtype=torch.float32)
        
        # Calculate aspect ratio
        aspect_ratio = actual_width / actual_height
        
        # Create info string
        info_parts = [
            f"Size: {actual_width}x{actual_height}",
            f"Frames: {frames.shape[0]}",
            f"FPS: {float(frame_rate):.2f}",
            f"Aspect: {aspect_ratio:.3f}",
            f"Mode: {scale_mode}/{fill_mode}",
            f"Original: {orig_width}x{orig_height}"
        ]
        
        if border_mode != "color" and fill_mode == "fit":
            info_parts.append(f"Border: {border_mode}")
        
        info = " | ".join(info_parts)
        
        # Create output video
        output_video = VideoFromComponents(
            VideoComponents(
                images=output_frames,
                audio=audio,
                frame_rate=frame_rate
            )
        )
        
        return (output_video, mask, actual_width, actual_height, aspect_ratio, info)