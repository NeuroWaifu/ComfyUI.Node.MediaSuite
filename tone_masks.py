import torch
import torch.nn.functional as F


class ShadowsHighlightsMidtones:
    """Extract shadows, highlights, and midtones masks from image"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "shadow_threshold": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Upper threshold for shadows"
                }),
                "highlight_threshold": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Lower threshold for highlights"
                }),
                "smoothness": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Smoothness of transitions between regions"
                }),
                "method": (["luminance", "value", "lightness", "average"],),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("shadows", "midtones", "highlights", "info")
    FUNCTION = "extract_tone_masks"
    CATEGORY = "mask/tone"

    def extract_tone_masks(self, image, shadow_threshold, highlight_threshold, smoothness, method):
        """Extract tone masks from image"""
        
        batch_size = image.shape[0]
        height, width = image.shape[1], image.shape[2]
        
        # Convert to grayscale based on method
        if method == "luminance":
            # Standard luminance weights
            gray = 0.299 * image[:, :, :, 0] + 0.587 * image[:, :, :, 1] + 0.114 * image[:, :, :, 2]
        elif method == "value":
            # Max of RGB (V in HSV)
            gray = torch.max(image[:, :, :, :3], dim=3)[0]
        elif method == "lightness":
            # Average of max and min (L in HSL)
            max_vals = torch.max(image[:, :, :, :3], dim=3)[0]
            min_vals = torch.min(image[:, :, :, :3], dim=3)[0]
            gray = (max_vals + min_vals) / 2
        else:  # average
            # Simple average
            gray = torch.mean(image[:, :, :, :3], dim=3)
        
        # Create masks with smooth transitions
        if smoothness > 0:
            # Use sigmoid for smooth transitions
            k = 10.0 / smoothness
            
            # Shadows: values below shadow_threshold
            shadows = 1.0 - torch.sigmoid(k * (gray - shadow_threshold))
            
            # Highlights: values above highlight_threshold
            highlights = torch.sigmoid(k * (gray - highlight_threshold))
            
            # Midtones: values between thresholds
            # Peak at center with falloff to both sides
            center = (shadow_threshold + highlight_threshold) / 2
            width = (highlight_threshold - shadow_threshold) / 2
            
            # Create bell curve for midtones
            distance_from_center = torch.abs(gray - center)
            midtones = torch.exp(-(distance_from_center / (width * 0.5)) ** 2)
            
        else:
            # Hard thresholds
            shadows = (gray < shadow_threshold).float()
            highlights = (gray > highlight_threshold).float()
            midtones = ((gray >= shadow_threshold) & (gray <= highlight_threshold)).float()
        
        # Normalize masks to ensure they don't exceed 1.0
        shadows = torch.clamp(shadows, 0.0, 1.0)
        highlights = torch.clamp(highlights, 0.0, 1.0)
        midtones = torch.clamp(midtones, 0.0, 1.0)
        
        # Create info string
        info_parts = [
            f"Method: {method}",
            f"Shadows: <{shadow_threshold:.2f}",
            f"Highlights: >{highlight_threshold:.2f}",
            f"Midtones: {shadow_threshold:.2f}-{highlight_threshold:.2f}",
            f"Smoothness: {smoothness:.2f}"
        ]
        info = " | ".join(info_parts)
        
        return (shadows, midtones, highlights, info)


class ToneCurve:
    """Apply tone curve adjustments using masks"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "shadows_adjustment": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Brightness adjustment for shadows (-1 to 1)"
                }),
                "midtones_adjustment": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Brightness adjustment for midtones (-1 to 1)"
                }),
                "highlights_adjustment": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Brightness adjustment for highlights (-1 to 1)"
                }),
                "preserve_colors": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Preserve color relationships when adjusting"
                }),
            },
            "optional": {
                "shadows_mask": ("MASK",),
                "midtones_mask": ("MASK",),
                "highlights_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_tone_curve"
    CATEGORY = "image/adjust"

    def apply_tone_curve(self, image, shadows_adjustment, midtones_adjustment, 
                        highlights_adjustment, preserve_colors, 
                        shadows_mask=None, midtones_mask=None, highlights_mask=None):
        """Apply tone curve adjustments to image"""
        
        # If masks not provided, generate them
        if shadows_mask is None or midtones_mask is None or highlights_mask is None:
            # Use default luminance method
            gray = 0.299 * image[:, :, :, 0] + 0.587 * image[:, :, :, 1] + 0.114 * image[:, :, :, 2]
            
            # Default thresholds
            shadow_threshold = 0.25
            highlight_threshold = 0.75
            k = 10.0  # Smoothness factor
            
            if shadows_mask is None:
                shadows_mask = 1.0 - torch.sigmoid(k * (gray - shadow_threshold))
            
            if highlights_mask is None:
                highlights_mask = torch.sigmoid(k * (gray - highlight_threshold))
            
            if midtones_mask is None:
                center = 0.5
                width = 0.25
                distance_from_center = torch.abs(gray - center)
                midtones_mask = torch.exp(-(distance_from_center / width) ** 2)
        
        # Ensure masks have correct shape
        if len(shadows_mask.shape) == 3:
            shadows_mask = shadows_mask.unsqueeze(-1)
        if len(midtones_mask.shape) == 3:
            midtones_mask = midtones_mask.unsqueeze(-1)
        if len(highlights_mask.shape) == 3:
            highlights_mask = highlights_mask.unsqueeze(-1)
        
        # Clone image for processing
        result = image.clone()
        
        if preserve_colors:
            # Calculate luminance
            luminance = 0.299 * image[:, :, :, 0:1] + 0.587 * image[:, :, :, 1:2] + 0.114 * image[:, :, :, 2:3]
            
            # Apply adjustments to luminance
            adjusted_luminance = luminance.clone()
            
            # Apply shadow adjustment
            adjusted_luminance = adjusted_luminance + shadows_adjustment * shadows_mask
            
            # Apply midtone adjustment  
            adjusted_luminance = adjusted_luminance + midtones_adjustment * midtones_mask
            
            # Apply highlight adjustment
            adjusted_luminance = adjusted_luminance + highlights_adjustment * highlights_mask
            
            # Clamp luminance
            adjusted_luminance = torch.clamp(adjusted_luminance, 0.0, 1.0)
            
            # Calculate scaling factor
            scale = torch.where(luminance > 0.001, adjusted_luminance / luminance, torch.ones_like(luminance))
            
            # Apply scaling to preserve colors
            result[:, :, :, :3] = torch.clamp(image[:, :, :, :3] * scale, 0.0, 1.0)
            
        else:
            # Apply adjustments directly to each channel
            # Shadows
            result[:, :, :, :3] = result[:, :, :, :3] + shadows_adjustment * shadows_mask[:, :, :, :3]
            
            # Midtones
            result[:, :, :, :3] = result[:, :, :, :3] + midtones_adjustment * midtones_mask[:, :, :, :3]
            
            # Highlights
            result[:, :, :, :3] = result[:, :, :, :3] + highlights_adjustment * highlights_mask[:, :, :, :3]
            
            # Clamp values
            result[:, :, :, :3] = torch.clamp(result[:, :, :, :3], 0.0, 1.0)
        
        # Preserve alpha channel if exists
        if image.shape[3] == 4:
            result[:, :, :, 3] = image[:, :, :, 3]
        
        return (result,)


class ColorGrading:
    """Advanced color grading with tone-based adjustments"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "shadows_hue": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Hue shift for shadows (-1 to 1)"
                }),
                "shadows_saturation": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Saturation multiplier for shadows"
                }),
                "midtones_hue": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "midtones_saturation": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                }),
                "highlights_hue": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "highlights_saturation": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                }),
                "balance_mode": (["multiply", "overlay", "soft_light"],),
            },
            "optional": {
                "shadows_mask": ("MASK",),
                "midtones_mask": ("MASK",),
                "highlights_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_color_grading"
    CATEGORY = "image/color"

    def rgb_to_hsv(self, rgb):
        """Convert RGB to HSV"""
        r, g, b = rgb[:, :, :, 0], rgb[:, :, :, 1], rgb[:, :, :, 2]
        
        max_val = torch.max(rgb[:, :, :, :3], dim=3)[0]
        min_val = torch.min(rgb[:, :, :, :3], dim=3)[0]
        diff = max_val - min_val
        
        # Value
        v = max_val
        
        # Saturation
        s = torch.where(max_val > 0, diff / max_val, torch.zeros_like(max_val))
        
        # Hue
        h = torch.zeros_like(max_val)
        
        # Red is max
        mask = (r == max_val) & (diff > 0)
        h = torch.where(mask, (g - b) / diff, h)
        
        # Green is max
        mask = (g == max_val) & (diff > 0)
        h = torch.where(mask, 2.0 + (b - r) / diff, h)
        
        # Blue is max
        mask = (b == max_val) & (diff > 0)
        h = torch.where(mask, 4.0 + (r - g) / diff, h)
        
        h = h / 6.0
        h = torch.where(h < 0, h + 1.0, h)
        
        return h, s, v

    def hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB"""
        i = (h * 6.0).floor()
        f = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        
        i = i % 6
        
        # Initialize RGB
        rgb = torch.zeros(h.shape[0], h.shape[1], h.shape[2], 3, device=h.device, dtype=h.dtype)
        
        # Assign RGB based on hue sector
        mask = (i == 0)
        rgb[mask] = torch.stack([v[mask], t[mask], p[mask]], dim=-1)
        
        mask = (i == 1)
        rgb[mask] = torch.stack([q[mask], v[mask], p[mask]], dim=-1)
        
        mask = (i == 2)
        rgb[mask] = torch.stack([p[mask], v[mask], t[mask]], dim=-1)
        
        mask = (i == 3)
        rgb[mask] = torch.stack([p[mask], q[mask], v[mask]], dim=-1)
        
        mask = (i == 4)
        rgb[mask] = torch.stack([t[mask], p[mask], v[mask]], dim=-1)
        
        mask = (i == 5)
        rgb[mask] = torch.stack([v[mask], p[mask], q[mask]], dim=-1)
        
        return rgb

    def apply_color_grading(self, image, shadows_hue, shadows_saturation,
                           midtones_hue, midtones_saturation,
                           highlights_hue, highlights_saturation,
                           balance_mode, shadows_mask=None, 
                           midtones_mask=None, highlights_mask=None):
        """Apply color grading based on tone regions"""
        
        # Generate masks if not provided
        if shadows_mask is None or midtones_mask is None or highlights_mask is None:
            gray = 0.299 * image[:, :, :, 0] + 0.587 * image[:, :, :, 1] + 0.114 * image[:, :, :, 2]
            
            shadow_threshold = 0.25
            highlight_threshold = 0.75
            k = 10.0
            
            if shadows_mask is None:
                shadows_mask = 1.0 - torch.sigmoid(k * (gray - shadow_threshold))
            
            if highlights_mask is None:
                highlights_mask = torch.sigmoid(k * (gray - highlight_threshold))
            
            if midtones_mask is None:
                center = 0.5
                width = 0.25
                distance_from_center = torch.abs(gray - center)
                midtones_mask = torch.exp(-(distance_from_center / width) ** 2)
        
        # Convert to HSV
        h, s, v = self.rgb_to_hsv(image)
        
        # Prepare masks
        if len(shadows_mask.shape) == 3:
            shadows_mask = shadows_mask.unsqueeze(-1)
        if len(midtones_mask.shape) == 3:
            midtones_mask = midtones_mask.unsqueeze(-1)
        if len(highlights_mask.shape) == 3:
            highlights_mask = highlights_mask.unsqueeze(-1)
        
        # Apply adjustments based on balance mode
        if balance_mode == "multiply":
            # Apply hue shifts
            h = h + shadows_hue * shadows_mask[:, :, :, 0]
            h = h + midtones_hue * midtones_mask[:, :, :, 0]
            h = h + highlights_hue * highlights_mask[:, :, :, 0]
            
            # Apply saturation adjustments
            s = s * (1 + (shadows_saturation - 1) * shadows_mask[:, :, :, 0])
            s = s * (1 + (midtones_saturation - 1) * midtones_mask[:, :, :, 0])
            s = s * (1 + (highlights_saturation - 1) * highlights_mask[:, :, :, 0])
            
        elif balance_mode == "overlay":
            # Weighted average based on masks
            total_weight = shadows_mask + midtones_mask + highlights_mask
            total_weight = torch.clamp(total_weight, 0.001, 3.0)
            
            h_adjustment = (shadows_hue * shadows_mask + 
                          midtones_hue * midtones_mask + 
                          highlights_hue * highlights_mask) / total_weight
            s_adjustment = (shadows_saturation * shadows_mask + 
                          midtones_saturation * midtones_mask + 
                          highlights_saturation * highlights_mask) / total_weight
            
            h = h + h_adjustment[:, :, :, 0]
            s = s * s_adjustment[:, :, :, 0]
            
        else:  # soft_light
            # Soft blending
            h_base = h.clone()
            s_base = s.clone()
            
            # Apply each adjustment with soft blending
            for mask, h_adj, s_adj in [
                (shadows_mask, shadows_hue, shadows_saturation),
                (midtones_mask, midtones_hue, midtones_saturation),
                (highlights_mask, highlights_hue, highlights_saturation)
            ]:
                mask_strength = mask[:, :, :, 0]
                h = h + h_adj * mask_strength * (1 - torch.abs(h - h_base))
                s = s * (1 + (s_adj - 1) * mask_strength * (2 * s_base - s_base * s_base))
        
        # Wrap hue values
        h = torch.fmod(h, 1.0)
        h = torch.where(h < 0, h + 1.0, h)
        
        # Clamp saturation
        s = torch.clamp(s, 0.0, 1.0)
        
        # Convert back to RGB
        rgb = self.hsv_to_rgb(h, s, v)
        
        # Create output image
        result = torch.zeros_like(image)
        result[:, :, :, :3] = rgb
        
        # Preserve alpha if exists
        if image.shape[3] == 4:
            result[:, :, :, 3] = image[:, :, :, 3]
        
        return (result,)