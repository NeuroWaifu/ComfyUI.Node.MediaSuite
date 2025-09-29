import torch
import comfy.utils
import comfy.model_management
from comfy_api.input_impl import VideoFromComponents
from comfy_api.util import VideoComponents


class VideoUpscaleWithModel:
    """Upscale video frames using an upscale model"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "video": ("VIDEO",),
            }
        }

    RETURN_TYPES = ("VIDEO", "INT", "INT", "STRING")
    RETURN_NAMES = ("VIDEO", "width", "height", "info")
    FUNCTION = "upscale_video"
    CATEGORY = "video/upscaling"


    def upscale_video(self, upscale_model, video):
        """Upscale video frames using the same method as the original ImageUpscaleWithModel"""
        
        # Get video components
        components = video.get_components()
        frames = components.images
        audio = components.audio
        frame_rate = components.frame_rate
        
        # Get device
        device = comfy.model_management.get_torch_device()
        
        # Calculate memory requirements
        memory_required = comfy.model_management.module_size(upscale_model.model)
        memory_required += (512 * 512 * 3) * frames.element_size() * max(upscale_model.scale, 1.0) * 384.0
        memory_required += frames.nelement() * frames.element_size()
        comfy.model_management.free_memory(memory_required, device)
        
        # Move model to device
        upscale_model.to(device)
        
        # Process all frames at once (like the original node)
        in_img = frames.movedim(-1, -3).to(device)
        
        tile = 512
        overlap = 32
        
        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                pbar = comfy.utils.ProgressBar(steps)
                s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
                oom = False
            except comfy.model_management.OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise e
        
        # Move model back to CPU
        upscale_model.to("cpu")
        
        # Convert back to ComfyUI format and clamp
        output_frames = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
        
        # Get output dimensions
        output_height, output_width = output_frames.shape[1], output_frames.shape[2]
        scale_factor = output_width / frames.shape[2]
        
        # Create info string
        info_parts = [
            f"Frames: {frames.shape[0]}",
            f"Original: {frames.shape[2]}x{frames.shape[1]}",
            f"Upscaled: {output_width}x{output_height}",
            f"Scale: {scale_factor:.0f}x",
            f"FPS: {float(frame_rate):.2f}"
        ]
        
        info = " | ".join(info_parts)
        
        # Create output video
        output_video = VideoFromComponents(
            VideoComponents(
                images=output_frames,
                audio=audio,
                frame_rate=frame_rate
            )
        )
        
        return (output_video, output_width, output_height, info)