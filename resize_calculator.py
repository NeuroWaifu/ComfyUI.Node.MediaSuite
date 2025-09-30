from .resize_core import ResizeCore


class ResizeCalculator:
    """Calculate resize dimensions from input dimensions"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_width": ("INT", {
                    "default": 1920,
                    "min": 1,
                    "max": 16384,
                    "step": 1,
                    "tooltip": "Original width"
                }),
                "input_height": ("INT", {
                    "default": 1080,
                    "min": 1,
                    "max": 16384,
                    "step": 1,
                    "tooltip": "Original height"
                }),
                "scale_mode": (["width", "height", "both", "longest_side", "shortest_side"],),
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
                    "tooltip": "Round dimensions to nearest multiple"
                }),
                "force_even": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force dimensions to be even (for video compatibility)"
                }),
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT", "FLOAT", "STRING", "INT", "INT")
    RETURN_NAMES = ("width", "height", "aspect_ratio", "scale_factor", "info", "maximum_side", "minimum_side")
    FUNCTION = "calculate"
    CATEGORY = "utils/math"

    def calculate(self, input_width, input_height, scale_mode, size_mode, fill_mode,
                  target_width, target_height, step, force_even):
        """Calculate final dimensions from input dimensions"""
        
        # Use provided dimensions
        original_width = input_width
        original_height = input_height
        
        # Calculate target dimensions
        final_width, final_height = ResizeCore.calculate_dimensions(
            original_width, original_height, target_width, target_height,
            scale_mode, size_mode, fill_mode, step
        )
        
        # For keep_aspect mode, recalculate actual dimensions
        if fill_mode == "keep_aspect":
            scale_x = target_width / original_width
            scale_y = target_height / original_height
            
            if scale_x < scale_y:
                actual_width = target_width
                actual_height = int(original_height * scale_x)
            else:
                actual_height = target_height
                actual_width = int(original_width * scale_y)
                
            # Round to step
            actual_width = ResizeCore.round_to_step(actual_width, step)
            actual_height = ResizeCore.round_to_step(actual_height, step)
            
            # Ensure minimum size
            actual_width = max(step, actual_width)
            actual_height = max(step, actual_height)
        else:
            actual_width = final_width
            actual_height = final_height
        
        # Force even dimensions if requested
        if force_even:
            actual_width = actual_width + (actual_width % 2)
            actual_height = actual_height + (actual_height % 2)
        
        # Calculate aspect ratio and scale factor
        aspect_ratio = actual_width / actual_height
        scale_factor = min(actual_width / original_width, actual_height / original_height)
        
        # Create info string
        info_parts = [
            f"Original: {original_width}x{original_height}",
            f"Target: {target_width}x{target_height}",
            f"Final: {actual_width}x{actual_height}",
            f"Aspect: {aspect_ratio:.3f}",
            f"Scale: {scale_factor:.3f}x",
            f"Mode: {scale_mode}/{fill_mode}"
        ]
        
        if step > 1:
            info_parts.append(f"Step: {step}")
        if force_even:
            info_parts.append("Even: Yes")
            
        info = " | ".join(info_parts)
        
        # Calculate maximum and minimum sides
        maximum_side = max(actual_width, actual_height)
        minimum_side = min(actual_width, actual_height)
        
        return (actual_width, actual_height, aspect_ratio, scale_factor, info, maximum_side, minimum_side)