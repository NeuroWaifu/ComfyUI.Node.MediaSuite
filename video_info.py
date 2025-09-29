class GetVideoSize:
    """Get dimensions and info from video"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("width", "height", "frame_count", "fps", "duration", "info")
    FUNCTION = "get_size"
    CATEGORY = "video"

    def get_size(self, video):
        """Extract video dimensions and info"""
        
        # Get video components
        components = video.get_components()
        frames = components.images
        frame_rate = components.frame_rate
        
        # Get dimensions
        frame_count = frames.shape[0]
        height = frames.shape[1]
        width = frames.shape[2]
        channels = frames.shape[3]
        
        # Calculate duration
        fps = float(frame_rate)
        duration = frame_count / fps if fps > 0 else 0
        
        # Create info string
        info_parts = [
            f"Size: {width}x{height}",
            f"Frames: {frame_count}",
            f"FPS: {fps:.2f}",
            f"Duration: {duration:.2f}s",
            f"Channels: {channels}"
        ]
        
        # Add audio info if available
        if components.audio is not None:
            info_parts.append("Audio: Yes")
        else:
            info_parts.append("Audio: No")
            
        info = " | ".join(info_parts)
        
        return (width, height, frame_count, fps, duration, info)


class GetVideoInfo:
    """Extended video information"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("resolution", "frame_info", "time_info", "format_info", "audio_info", "full_info")
    FUNCTION = "get_info"
    CATEGORY = "video"

    def get_info(self, video):
        """Extract detailed video information"""
        
        # Get video components
        components = video.get_components()
        frames = components.images
        frame_rate = components.frame_rate
        audio = components.audio
        
        # Get dimensions
        frame_count = frames.shape[0]
        height = frames.shape[1]
        width = frames.shape[2]
        channels = frames.shape[3]
        
        # Calculate derived values
        fps = float(frame_rate)
        duration = frame_count / fps if fps > 0 else 0
        aspect_ratio = width / height
        total_pixels = width * height
        
        # Resolution info
        resolution = f"{width}x{height}"
        
        # Frame info
        frame_info = f"{frame_count} frames"
        
        # Time info
        minutes = int(duration // 60)
        seconds = duration % 60
        if minutes > 0:
            time_info = f"{minutes}m {seconds:.1f}s @ {fps:.2f}fps"
        else:
            time_info = f"{seconds:.1f}s @ {fps:.2f}fps"
        
        # Format info
        if channels == 3:
            format_str = "RGB"
        elif channels == 4:
            format_str = "RGBA"
        else:
            format_str = f"{channels}ch"
        
        format_info = f"{format_str} | AR: {aspect_ratio:.3f}"
        
        # Audio info
        if audio is not None:
            audio_info = "Has audio track"
        else:
            audio_info = "No audio"
        
        # Full info
        full_info_parts = [
            f"Video: {width}x{height} {format_str}",
            f"Duration: {time_info}",
            f"Frames: {frame_count}",
            f"Aspect Ratio: {aspect_ratio:.3f}",
            f"Total Pixels: {total_pixels:,}",
            f"Audio: {'Yes' if audio is not None else 'No'}"
        ]
        full_info = "\n".join(full_info_parts)
        
        return (resolution, frame_info, time_info, format_info, audio_info, full_info)


class CompareVideoSize:
    """Compare two videos dimensions"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_a": ("VIDEO",),
                "video_b": ("VIDEO",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "BOOLEAN", "BOOLEAN", "BOOLEAN", "STRING")
    RETURN_NAMES = ("same_size", "same_fps", "same_duration", "same_format", "comparison")
    FUNCTION = "compare"
    CATEGORY = "video"

    def compare(self, video_a, video_b):
        """Compare two videos"""
        
        # Get components
        comp_a = video_a.get_components()
        comp_b = video_b.get_components()
        
        frames_a = comp_a.images
        frames_b = comp_b.images
        
        # Get dimensions
        width_a, height_a = frames_a.shape[2], frames_a.shape[1]
        width_b, height_b = frames_b.shape[2], frames_b.shape[1]
        
        frames_a_count = frames_a.shape[0]
        frames_b_count = frames_b.shape[0]
        
        channels_a = frames_a.shape[3]
        channels_b = frames_b.shape[3]
        
        fps_a = float(comp_a.frame_rate)
        fps_b = float(comp_b.frame_rate)
        
        # Compare
        same_size = (width_a == width_b) and (height_a == height_b)
        same_fps = abs(fps_a - fps_b) < 0.01
        same_duration = frames_a_count == frames_b_count
        same_format = channels_a == channels_b
        
        # Create comparison string
        comparison_parts = []
        
        comparison_parts.append(f"Video A: {width_a}x{height_a}, {frames_a_count} frames @ {fps_a:.2f}fps")
        comparison_parts.append(f"Video B: {width_b}x{height_b}, {frames_b_count} frames @ {fps_b:.2f}fps")
        comparison_parts.append("")
        
        if same_size:
            comparison_parts.append("✓ Same resolution")
        else:
            comparison_parts.append(f"✗ Different resolution (A: {width_a}x{height_a}, B: {width_b}x{height_b})")
            
        if same_fps:
            comparison_parts.append("✓ Same FPS")
        else:
            comparison_parts.append(f"✗ Different FPS (A: {fps_a:.2f}, B: {fps_b:.2f})")
            
        if same_duration:
            comparison_parts.append("✓ Same frame count")
        else:
            comparison_parts.append(f"✗ Different frame count (A: {frames_a_count}, B: {frames_b_count})")
            
        if same_format:
            comparison_parts.append(f"✓ Same format ({channels_a} channels)")
        else:
            comparison_parts.append(f"✗ Different format (A: {channels_a}ch, B: {channels_b}ch)")
        
        comparison = "\n".join(comparison_parts)
        
        return (same_size, same_fps, same_duration, same_format, comparison)