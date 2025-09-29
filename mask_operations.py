import torch
import torch.nn.functional as F


class MaskOperations:
    """Perform logical operations on masks"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_a": ("MASK",),
                "mask_b": ("MASK",),
                "operation": (["OR", "AND", "SUB", "XOR", "ADD", "MIN", "MAX"],),
                "invert_a": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert mask A before operation"
                }),
                "invert_b": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert mask B before operation"
                }),
                "invert_output": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert the result after operation"
                }),
                "clamp_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clamp output values to 0-1 range"
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "process_masks"
    CATEGORY = "mask"

    def process_masks(self, mask_a, mask_b, operation, invert_a, invert_b, invert_output, clamp_output):
        """Process mask operations"""
        
        # Ensure masks have the same shape
        if mask_a.shape != mask_b.shape:
            # Handle different batch sizes
            if mask_a.shape[0] != mask_b.shape[0]:
                # Repeat the smaller batch to match
                if mask_a.shape[0] < mask_b.shape[0]:
                    mask_a = mask_a.repeat(mask_b.shape[0] // mask_a.shape[0], 1, 1)
                else:
                    mask_b = mask_b.repeat(mask_a.shape[0] // mask_b.shape[0], 1, 1)
            
            # Handle different spatial dimensions
            if mask_a.shape[1:] != mask_b.shape[1:]:
                # Resize mask_b to match mask_a dimensions
                mask_b = F.interpolate(
                    mask_b.unsqueeze(1),  # Add channel dimension
                    size=(mask_a.shape[1], mask_a.shape[2]),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)  # Remove channel dimension
        
        # Apply inversions if requested
        if invert_a:
            mask_a = 1.0 - mask_a
        
        if invert_b:
            mask_b = 1.0 - mask_b
        
        # Perform operations
        if operation == "OR":
            # Logical OR - maximum of both masks
            result = torch.maximum(mask_a, mask_b)
        elif operation == "AND":
            # Logical AND - minimum of both masks
            result = torch.minimum(mask_a, mask_b)
        elif operation == "SUB":
            # Subtraction - subtract B from A
            result = mask_a - mask_b
        elif operation == "XOR":
            # Exclusive OR - areas in one mask but not both
            result = torch.abs(mask_a - mask_b)
        elif operation == "ADD":
            # Addition - add masks together
            result = mask_a + mask_b
        elif operation == "MIN":
            # Minimum - take the smaller value at each pixel
            result = torch.minimum(mask_a, mask_b)
        elif operation == "MAX":
            # Maximum - take the larger value at each pixel
            result = torch.maximum(mask_a, mask_b)
        
        # Invert output if requested
        if invert_output:
            result = 1.0 - result
        
        # Clamp values to 0-1 range if requested
        if clamp_output:
            result = torch.clamp(result, 0.0, 1.0)
        
        return (result,)


class MaskBlur:
    """Blur mask edges"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "blur_radius": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.5,
                    "tooltip": "Gaussian blur radius"
                }),
                "iterations": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of blur iterations"
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "blur_mask"
    CATEGORY = "mask"

    def blur_mask(self, mask, blur_radius, iterations):
        """Apply Gaussian blur to mask"""
        
        if blur_radius == 0:
            return (mask,)
        
        # Convert mask to 4D tensor for conv2d
        result = mask.unsqueeze(1)  # Add channel dimension
        
        # Calculate kernel size (must be odd)
        kernel_size = int(blur_radius * 4) | 1  # Ensure odd
        
        # Create Gaussian kernel
        sigma = blur_radius
        kernel_1d = torch.exp(-torch.arange(kernel_size, dtype=torch.float32) ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
        
        # Move to same device as mask
        kernel_2d = kernel_2d.to(mask.device)
        
        # Apply blur iterations
        for _ in range(iterations):
            # Pad the mask
            padding = kernel_size // 2
            result = F.pad(result, (padding, padding, padding, padding), mode='reflect')
            
            # Apply convolution
            result = F.conv2d(result, kernel_2d)
        
        # Remove channel dimension
        result = result.squeeze(1)
        
        return (result,)


class MaskThreshold:
    """Apply threshold to mask values"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Threshold value"
                }),
                "mode": (["binary", "smooth"],),
                "smoothness": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Smoothness for smooth mode"
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "threshold_mask"
    CATEGORY = "mask"

    def threshold_mask(self, mask, threshold, mode, smoothness):
        """Apply threshold to mask"""
        
        if mode == "binary":
            # Hard threshold
            result = (mask > threshold).float()
        else:  # smooth
            # Smooth threshold using sigmoid
            if smoothness == 0:
                result = (mask > threshold).float()
            else:
                # Scale factor for sigmoid steepness
                k = 10.0 / smoothness
                result = torch.sigmoid(k * (mask - threshold))
        
        return (result,)