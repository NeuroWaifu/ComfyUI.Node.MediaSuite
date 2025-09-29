import torch
import numpy as np


def simple_kmeans(pixels, n_clusters, n_iterations=10):
    """Simple k-means clustering implementation"""
    # Initialize cluster centers randomly
    indices = torch.randperm(pixels.shape[0])[:n_clusters]
    centers = pixels[indices].clone()
    
    for _ in range(n_iterations):
        # Assign each pixel to nearest center
        distances = torch.cdist(pixels.unsqueeze(0), centers.unsqueeze(0))[0]
        labels = torch.argmin(distances, dim=1)
        
        # Update centers
        for k in range(n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                centers[k] = pixels[mask].mean(dim=0)
    
    return centers.numpy(), labels.numpy()


class DominantColor:
    """Extract dominant color from image or video"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_colors": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of dominant colors to extract"
                }),
                "sample_size": ("INT", {
                    "default": 10000,
                    "min": 100,
                    "max": 100000,
                    "step": 100,
                    "tooltip": "Number of pixels to sample for faster processing"
                }),
                "method": (["kmeans", "histogram"],),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("hex_color", "red", "green", "blue", "info")
    FUNCTION = "extract_color"
    CATEGORY = "color"

    def extract_color(self, image, num_colors, sample_size, method, mask=None):
        """Extract dominant color from image/video"""
        
        batch_size = image.shape[0]
        height, width = image.shape[1], image.shape[2]
        
        # Process each frame and collect colors
        all_colors = []
        
        for i in range(batch_size):
            frame = image[i]
            
            # Apply mask if provided
            if mask is not None:
                # Handle different mask shapes
                if len(mask.shape) == 3 and mask.shape[0] >= i + 1:
                    # Video mask - use corresponding frame
                    frame_mask = mask[i]
                elif len(mask.shape) == 2:
                    # Single mask for all frames
                    frame_mask = mask
                else:
                    # Use first mask frame for all
                    frame_mask = mask[0]
                
                # Ensure mask has same spatial dimensions
                if frame_mask.shape != (height, width):
                    # Resize mask to match frame
                    frame_mask = torch.nn.functional.interpolate(
                        frame_mask.unsqueeze(0).unsqueeze(0),
                        size=(height, width),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                
                # Apply mask
                masked_pixels = frame[frame_mask > 0.5]
                
                if masked_pixels.shape[0] == 0:
                    # No pixels in mask, use center pixel
                    masked_pixels = frame[height//2:height//2+1, width//2:width//2+1]
            else:
                # No mask - reshape all pixels
                masked_pixels = frame.reshape(-1, 3)
            
            # Sample pixels for efficiency
            if masked_pixels.shape[0] > sample_size:
                indices = torch.randperm(masked_pixels.shape[0])[:sample_size]
                sampled_pixels = masked_pixels[indices]
            else:
                sampled_pixels = masked_pixels
            
            if method == "kmeans":
                # Use k-means clustering
                if sampled_pixels.shape[0] >= num_colors:
                    # Convert to float tensor for clustering
                    pixels_tensor = sampled_pixels * 255.0
                    
                    centers, labels = simple_kmeans(pixels_tensor, num_colors)
                    
                    # If we want the most dominant color, find the largest cluster
                    if num_colors == 1:
                        unique, counts = np.unique(labels, return_counts=True)
                        dominant_cluster = unique[np.argmax(counts)]
                        dominant_color = centers[dominant_cluster]
                        all_colors.append(dominant_color)
                    else:
                        # Return all cluster centers
                        all_colors.extend(centers)
                else:
                    # Not enough pixels, use mean
                    pixels_np = (sampled_pixels * 255).cpu().numpy()
                    all_colors.append(np.mean(pixels_np, axis=0))
                    
            else:  # histogram method
                # Use histogram-based method
                # Convert to numpy
                pixels_np = (sampled_pixels * 255).cpu().numpy().astype(np.uint8)
                
                # Create 3D histogram
                hist_bins = 16  # Reduce color space to 16x16x16
                
                # Quantize colors
                quantized = (pixels_np // (256 // hist_bins)) * (256 // hist_bins) + (256 // hist_bins // 2)
                
                # Find unique colors and their counts
                unique_colors, counts = np.unique(quantized, axis=0, return_counts=True)
                
                # Sort by frequency
                sorted_indices = np.argsort(counts)[::-1]
                
                # Take top colors
                top_colors = unique_colors[sorted_indices[:num_colors]]
                all_colors.extend(top_colors)
        
        # Average colors across all frames
        if len(all_colors) > 0:
            all_colors = np.array(all_colors)
            
            if batch_size > 1 and num_colors == 1:
                # For video, average the dominant colors from each frame
                final_color = np.mean(all_colors, axis=0).astype(np.uint8)
            else:
                # For single image or when extracting multiple colors
                final_color = all_colors[0].astype(np.uint8)
        else:
            # Fallback to black if no colors found
            final_color = np.array([0, 0, 0], dtype=np.uint8)
        
        # Convert to normalized values
        r, g, b = final_color[0] / 255.0, final_color[1] / 255.0, final_color[2] / 255.0
        
        # Convert to hex
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(r * 255),
            int(g * 255),
            int(b * 255)
        )
        
        # Create info string
        info_parts = [
            f"RGB({int(r*255)}, {int(g*255)}, {int(b*255)})",
            f"Hex: {hex_color}",
            f"Method: {method}",
            f"Frames: {batch_size}"
        ]
        
        if mask is not None:
            info_parts.append("Masked: Yes")
            
        info = " | ".join(info_parts)
        
        return (hex_color, float(r), float(g), float(b), info)


class ColorPalette:
    """Extract color palette from image or video"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_colors": ("INT", {
                    "default": 5,
                    "min": 2,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Number of colors in palette"
                }),
                "sample_size": ("INT", {
                    "default": 10000,
                    "min": 100,
                    "max": 100000,
                    "step": 100,
                    "tooltip": "Number of pixels to sample"
                }),
                "sort_by": (["frequency", "brightness", "hue", "saturation"],),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("palette_image", "hex_colors", "info")
    FUNCTION = "extract_palette"
    CATEGORY = "color"

    def rgb_to_hsv(self, r, g, b):
        """Convert RGB to HSV"""
        maxc = max(r, g, b)
        minc = min(r, g, b)
        v = maxc
        if minc == maxc:
            return 0.0, 0.0, v
        s = (maxc-minc) / maxc
        rc = (maxc-r) / (maxc-minc)
        gc = (maxc-g) / (maxc-minc)
        bc = (maxc-b) / (maxc-minc)
        if r == maxc:
            h = bc-gc
        elif g == maxc:
            h = 2.0+rc-bc
        else:
            h = 4.0+gc-rc
        h = (h/6.0) % 1.0
        return h, s, v

    def extract_palette(self, image, num_colors, sample_size, sort_by, mask=None):
        """Extract color palette from image/video"""
        
        batch_size = image.shape[0]
        height, width = image.shape[1], image.shape[2]
        
        # Collect pixels from all frames
        all_pixels = []
        
        for i in range(batch_size):
            frame = image[i]
            
            # Apply mask if provided
            if mask is not None:
                # Handle different mask shapes
                if len(mask.shape) == 3 and mask.shape[0] >= i + 1:
                    frame_mask = mask[i]
                elif len(mask.shape) == 2:
                    frame_mask = mask
                else:
                    frame_mask = mask[0]
                
                # Ensure mask has same spatial dimensions
                if frame_mask.shape != (height, width):
                    frame_mask = torch.nn.functional.interpolate(
                        frame_mask.unsqueeze(0).unsqueeze(0),
                        size=(height, width),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                
                masked_pixels = frame[frame_mask > 0.5]
                if masked_pixels.shape[0] > 0:
                    all_pixels.append(masked_pixels)
            else:
                all_pixels.append(frame.reshape(-1, 3))
        
        # Combine all pixels
        if len(all_pixels) > 0:
            combined_pixels = torch.cat(all_pixels, dim=0)
        else:
            # Fallback to center pixel
            combined_pixels = image[0, height//2:height//2+1, width//2:width//2+1].reshape(-1, 3)
        
        # Sample pixels
        if combined_pixels.shape[0] > sample_size:
            indices = torch.randperm(combined_pixels.shape[0])[:sample_size]
            sampled_pixels = combined_pixels[indices]
        else:
            sampled_pixels = combined_pixels
        
        # Extract colors using k-means
        if sampled_pixels.shape[0] >= num_colors:
            # Convert to float tensor for clustering
            pixels_tensor = sampled_pixels * 255.0
            centers, labels = simple_kmeans(pixels_tensor, num_colors)
            palette_colors = centers.astype(np.uint8)
            
            # Get cluster sizes for frequency sorting
            cluster_sizes = [np.sum(labels == i) for i in range(num_colors)]
        else:
            # Not enough pixels, use what we have
            pixels_np = (sampled_pixels * 255).cpu().numpy().astype(np.uint8)
            palette_colors = pixels_np[:num_colors]
            cluster_sizes = list(range(len(palette_colors), 0, -1))
        
        # Sort colors based on criteria
        if sort_by == "frequency":
            # Sort by cluster size (already have this info)
            sorted_indices = np.argsort(cluster_sizes)[::-1]
        elif sort_by == "brightness":
            # Sort by brightness (V in HSV)
            brightness = [self.rgb_to_hsv(c[0]/255, c[1]/255, c[2]/255)[2] for c in palette_colors]
            sorted_indices = np.argsort(brightness)[::-1]
        elif sort_by == "hue":
            # Sort by hue
            hues = [self.rgb_to_hsv(c[0]/255, c[1]/255, c[2]/255)[0] for c in palette_colors]
            sorted_indices = np.argsort(hues)
        elif sort_by == "saturation":
            # Sort by saturation
            saturations = [self.rgb_to_hsv(c[0]/255, c[1]/255, c[2]/255)[1] for c in palette_colors]
            sorted_indices = np.argsort(saturations)[::-1]
        
        palette_colors = palette_colors[sorted_indices]
        
        # Create palette image
        swatch_size = 64
        palette_width = swatch_size * num_colors
        palette_height = swatch_size
        palette_image = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
        
        for i, color in enumerate(palette_colors):
            x_start = i * swatch_size
            x_end = (i + 1) * swatch_size
            palette_image[:, x_start:x_end] = color
        
        # Convert to torch tensor
        palette_tensor = torch.from_numpy(palette_image).float() / 255.0
        palette_tensor = palette_tensor.unsqueeze(0)  # Add batch dimension
        
        # Generate hex colors
        hex_colors = []
        for color in palette_colors:
            hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
            hex_colors.append(hex_color)
        
        hex_string = ", ".join(hex_colors)
        
        # Create info string
        info_parts = [
            f"Colors: {num_colors}",
            f"Sorted by: {sort_by}",
            f"Frames: {batch_size}"
        ]
        
        if mask is not None:
            info_parts.append("Masked: Yes")
            
        info = " | ".join(info_parts)
        
        return (palette_tensor, hex_string, info)