import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
import numpy as np
import comfy.utils


class ResizeCore:
    """Core resize logic shared between image and video resize nodes"""
    
    @staticmethod
    def parse_color(color_string):
        """Parse color string in format 'R,G,B' or 'R,G,B,A' or hex '#RRGGBB'"""
        color_string = color_string.strip()
        
        # Try hex format first
        if color_string.startswith('#'):
            hex_color = color_string.lstrip('#')
            try:
                if len(hex_color) == 6:
                    r = int(hex_color[0:2], 16)
                    g = int(hex_color[2:4], 16)
                    b = int(hex_color[4:6], 16)
                    return [r, g, b, 255]
                elif len(hex_color) == 8:
                    r = int(hex_color[0:2], 16)
                    g = int(hex_color[2:4], 16)
                    b = int(hex_color[4:6], 16)
                    a = int(hex_color[6:8], 16)
                    return [r, g, b, a]
            except:
                pass
        
        # Try comma-separated format
        try:
            parts = [int(x.strip()) for x in color_string.split(',')]
            if len(parts) == 3:
                return parts + [255]  # Add alpha channel
            elif len(parts) == 4:
                return parts
            else:
                raise ValueError
        except:
            return [0, 0, 0, 255]  # Default to black

    @staticmethod
    def round_to_step(value, step):
        """Round value to nearest multiple of step"""
        return int(round(value / step) * step)

    @staticmethod
    def get_upscale_method(interpolation):
        """Map interpolation method to ComfyUI upscale method"""
        mapping = {
            "nearest": "nearest-exact",
            "bilinear": "bilinear",
            "bicubic": "bicubic",
            "lanczos": "lanczos",
            "area": "area"
        }
        return mapping.get(interpolation, "bilinear")

    @staticmethod
    def calculate_dimensions(orig_width, orig_height, target_width, target_height, 
                           scale_mode, size_mode, fill_mode, step):
        """Calculate target dimensions based on scaling parameters"""
        
        # Calculate initial target dimensions based on scale_mode
        if scale_mode == "width":
            scale = target_width / orig_width
            new_width = target_width
            new_height = int(orig_height * scale)
        elif scale_mode == "height":
            scale = target_height / orig_height
            new_width = int(orig_width * scale)
            new_height = target_height
        elif scale_mode == "both":
            new_width = target_width
            new_height = target_height
        elif scale_mode == "longest_side":
            if orig_width > orig_height:
                scale = target_width / orig_width
                new_width = target_width
                new_height = int(orig_height * scale)
            else:
                scale = target_height / orig_height
                new_width = int(orig_width * scale)
                new_height = target_height
        elif scale_mode == "shortest_side":
            if orig_width < orig_height:
                scale = target_width / orig_width
                new_width = target_width
                new_height = int(orig_height * scale)
            else:
                scale = target_height / orig_height
                new_width = int(orig_width * scale)
                new_height = target_height

        # Apply size_mode restrictions
        if size_mode == "upscale_only":
            new_width = max(new_width, orig_width)
            new_height = max(new_height, orig_height)
        elif size_mode == "downscale_only":
            new_width = min(new_width, orig_width)
            new_height = min(new_height, orig_height)

        # Round to step
        new_width = ResizeCore.round_to_step(new_width, step)
        new_height = ResizeCore.round_to_step(new_height, step)

        # Ensure minimum size
        new_width = max(step, new_width)
        new_height = max(step, new_height)

        return new_width, new_height

    @staticmethod
    def create_blur_background(image, target_width, target_height, blur_strength):
        """Create blurred background from image using PIL"""
        # Convert first frame to PIL
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        
        # Calculate scale to fill target dimensions
        orig_width, orig_height = pil_img.size
        scale_x = target_width / orig_width
        scale_y = target_height / orig_height
        scale = max(scale_x, scale_y) * 1.1  # Scale up 10% more
        
        # Resize
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        resized = pil_img.resize((new_width, new_height), Image.Resampling.BILINEAR)
        
        # Center crop
        x_start = (new_width - target_width) // 2
        y_start = (new_height - target_height) // 2
        cropped = resized.crop((x_start, y_start, x_start + target_width, y_start + target_height))
        
        # Apply blur
        blur_radius = blur_strength / 2
        blurred = cropped.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Convert back to tensor
        blurred_np = np.array(blurred).astype(np.float32) / 255.0
        blurred_tensor = torch.from_numpy(blurred_np)
        
        # Add batch dimension
        return blurred_tensor.unsqueeze(0)

    @staticmethod
    def resize_with_padding(images, target_width, target_height, fill_mode, 
                          border_mode, border_color, blur_strength, upscale_method):
        """Resize images with specified padding/cropping mode"""
        
        batch_size, orig_height, orig_width, channels = images.shape
        
        # Calculate scale factors
        scale_x = target_width / orig_width
        scale_y = target_height / orig_height
        
        if fill_mode == "keep_aspect":
            # Simple resize maintaining aspect ratio
            if scale_x < scale_y:
                new_width = target_width
                new_height = int(orig_height * scale_x)
            else:
                new_height = target_height
                new_width = int(orig_width * scale_y)
            
            # Use ComfyUI's common_upscale
            samples = images.movedim(-1, 1)  # BHWC -> BCHW
            resized = comfy.utils.common_upscale(samples, new_width, new_height, upscale_method, "disabled")
            output = resized.movedim(1, -1)  # BCHW -> BHWC
            
            # Add alpha channel if transparent mode is selected and input is RGB
            if border_mode == "transparent" and channels == 3:
                alpha_channel = torch.ones((batch_size, new_height, new_width, 1), dtype=torch.float32)
                output = torch.cat([output, alpha_channel], dim=3)
            
            # Create simple mask
            mask = torch.ones((batch_size, new_height, new_width), dtype=torch.float32)
            
            return output, mask, new_width, new_height
            
        elif fill_mode == "fill":
            # Scale to fill entire target (may crop)
            scale = max(scale_x, scale_y)
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            
            # Resize
            samples = images.movedim(-1, 1)
            resized = comfy.utils.common_upscale(samples, new_width, new_height, upscale_method, "disabled")
            resized = resized.movedim(1, -1)
            
            # Center crop to target size
            y_start = (new_height - target_height) // 2
            x_start = (new_width - target_width) // 2
            
            output = resized[:, y_start:y_start+target_height, x_start:x_start+target_width, :]
            mask = torch.ones((batch_size, target_height, target_width), dtype=torch.float32)
            
        elif fill_mode == "fit":
            # Scale to fit within target (may add borders)
            scale = min(scale_x, scale_y)
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            
            # Resize
            samples = images.movedim(-1, 1)
            resized = comfy.utils.common_upscale(samples, new_width, new_height, upscale_method, "disabled")
            resized = resized.movedim(1, -1)
            
            # Create output with padding
            if border_mode == "transparent":
                # Always create RGBA output for transparent mode
                output = torch.zeros((batch_size, target_height, target_width, 4), dtype=torch.float32)
                # If input is RGB, add alpha channel to resized image
                if channels == 3:
                    alpha_channel = torch.ones((batch_size, new_height, new_width, 1), dtype=torch.float32)
                    resized = torch.cat([resized, alpha_channel], dim=3)
            elif border_mode == "blur":
                # Create blurred background for each frame
                blur_frames = []
                for i in range(batch_size):
                    blur_bg = ResizeCore.create_blur_background(images[i:i+1], target_width, target_height, blur_strength)
                    blur_frames.append(blur_bg)
                output = torch.cat(blur_frames, dim=0)
                if channels == 4 and output.shape[3] == 3:
                    # Add alpha channel
                    alpha = torch.ones((batch_size, target_height, target_width, 1), dtype=torch.float32)
                    output = torch.cat([output, alpha], dim=3)
            else:  # color
                color = ResizeCore.parse_color(border_color)
                output = torch.ones((batch_size, target_height, target_width, channels), dtype=torch.float32)
                for i in range(min(channels, len(color))):
                    output[:, :, :, i] *= color[i] / 255.0
            
            # Calculate paste position
            y_offset = (target_height - new_height) // 2
            x_offset = (target_width - new_width) // 2
            
            # Paste resized image
            if border_mode == "transparent":
                # Paste with alpha channel
                output[:, y_offset:y_offset+new_height, x_offset:x_offset+new_width, :] = resized
            else:
                output[:, y_offset:y_offset+new_height, x_offset:x_offset+new_width, :] = resized
            
            # Create mask
            mask = torch.zeros((batch_size, target_height, target_width), dtype=torch.float32)
            mask[:, y_offset:y_offset+new_height, x_offset:x_offset+new_width] = 1.0
            
        else:  # crop
            # Center crop to aspect ratio, then resize
            target_aspect = target_width / target_height
            orig_aspect = orig_width / orig_height
            
            if orig_aspect > target_aspect:
                # Original is wider - crop width
                new_width = int(orig_height * target_aspect)
                x_start = (orig_width - new_width) // 2
                cropped = images[:, :, x_start:x_start+new_width, :]
            else:
                # Original is taller - crop height
                new_height = int(orig_width / target_aspect)
                y_start = (orig_height - new_height) // 2
                cropped = images[:, y_start:y_start+new_height, :, :]
            
            # Resize cropped image
            samples = cropped.movedim(-1, 1)
            output = comfy.utils.common_upscale(samples, target_width, target_height, upscale_method, "disabled")
            output = output.movedim(1, -1)
            
            mask = torch.ones((batch_size, target_height, target_width), dtype=torch.float32)
        
        return output, mask, target_width, target_height