import os
import torch
import numpy as np
import requests
import tempfile
import cv2
import torchaudio
import hashlib
import folder_paths
from urllib.parse import urlparse
from io import BytesIO
from comfy.utils import common_upscale

# Supported media types
AUDIO_EXTENSIONS = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
VIDEO_EXTENSIONS = ['.mp4', '.webm', '.mkv', '.mov', '.avi']

class AudioURLLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {"default": "https://example.com/audio.mp3"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "load_audio"
    CATEGORY = "MediaURLLoader"
    
    def load_audio(self, url):
        try:
            # Check if URL is valid
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid URL: {url}")
            
            # Download audio file
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Create temporary file to save the audio
            extension = os.path.splitext(parsed_url.path)[1].lower()
            if extension not in AUDIO_EXTENSIONS:
                extension = '.mp3'  # Default extension if not recognized
                
            with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp_file:
                temp_path = temp_file.name
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
            
            # Load audio using torchaudio
            waveform, sample_rate = torchaudio.load(temp_path)
            
            # Cleanup temporary file
            os.unlink(temp_path)
            
            # Return audio in ComfyUI format
            return ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate},)
            
        except Exception as e:
            print(f"Error loading audio from URL: {str(e)}")
            # Return empty audio in case of error
            waveform = torch.zeros((1, 2, 1))
            sample_rate = 44100
            return ({"waveform": waveform, "sample_rate": sample_rate},)

class AudioPreview:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    
    RETURN_TYPES = ()
    FUNCTION = "preview_audio"
    OUTPUT_NODE = True
    CATEGORY = "MediaURLLoader"
    
    def preview_audio(self, audio, prompt=None, extra_pnginfo=None):
        try:
            # Get temporary directory for audio preview
            temp_dir = folder_paths.get_temp_directory()
            
            # Generate a unique filename
            # Use md5 of current timestamp to avoid collisions
            import time
            unique_id = hashlib.md5(str(time.time()).encode()).hexdigest()
            filename = f"audio_preview_{unique_id}.flac"
            filepath = os.path.join(temp_dir, filename)
            
            # Save the audio to the temporary directory
            torchaudio.save(filepath, audio["waveform"][0], audio["sample_rate"], format="FLAC")
            
            # Return metadata for UI to display audio player
            return {"ui": {"audio": [{"filename": filename, "subfolder": "", "type": "temp"}]}}
            
        except Exception as e:
            print(f"Error previewing audio: {str(e)}")
            return {"ui": {"audio": []}}

class VideoURLLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {"default": "https://example.com/video.mp4"}),
                "force_rate": ("INT", {"default": 0, "min": 0, "max": 60, "step": 1}),
                "force_size": (["Disabled", "Custom Height", "Custom Width", "Custom", "512x512", "512x768", "768x512"], ),
                "custom_width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "custom_height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "AUDIO", "VHS_VIDEOINFO")
    RETURN_NAMES = ("FRAMES", "frame_count", "audio", "video_info")
    FUNCTION = "load_video"
    CATEGORY = "MediaURLLoader"
    
    def load_video(self, url, force_rate, force_size, custom_width, custom_height, 
                   frame_load_cap, skip_first_frames, select_every_nth):
        try:
            # Check if URL is valid
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid URL: {url}")
            
            # Download video file
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Create temporary file to save the video
            extension = os.path.splitext(parsed_url.path)[1].lower()
            if extension not in VIDEO_EXTENSIONS:
                extension = '.mp4'  # Default extension if not recognized
                
            with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp_file:
                temp_path = temp_file.name
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
            
            # Load video using OpenCV
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file from {url}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            # Apply force_rate
            base_frame_time = 1 / fps
            if force_rate == 0:
                target_frame_time = base_frame_time
            else:
                target_frame_time = 1 / force_rate
            
            # Determine target size
            if force_size != "Disabled":
                if force_size == "Custom Height":
                    new_height = custom_height
                    new_width = int(width * (new_height / height))
                elif force_size == "Custom Width":
                    new_width = custom_width
                    new_height = int(height * (new_width / width))
                elif force_size == "Custom":
                    new_width = custom_width
                    new_height = custom_height
                else:
                    # Parse dimensions from preset option (e.g., "512x768")
                    dimensions = force_size.split("x")
                    new_width = int(dimensions[0])
                    new_height = int(dimensions[1])
            else:
                new_width = width
                new_height = height
            
            # Calculate frames to extract based on parameters
            frames_to_extract = total_frames
            if force_rate != 0:
                frames_to_extract = int(duration * force_rate)
            
            if frame_load_cap != 0:
                frames_to_extract = min(frame_load_cap, frames_to_extract)
            
            # Extract frames
            frames = []
            frame_count = 0
            total_frame_count = 0
            time_offset = 0
            
            # Get audio track if available
            # This is a simplified approach, actual audio extraction might require ffmpeg
            audio_waveform = torch.zeros((1, 2, 1))  # Empty audio by default
            audio_sample_rate = 44100
            
            # Try to extract audio using torchaudio (simplified approach)
            try:
                audio_waveform, audio_sample_rate = torchaudio.load(temp_path)
                # Adjust audio duration based on frame selection
                start_time = skip_first_frames * target_frame_time
                end_time = (skip_first_frames + frames_to_extract) * target_frame_time * select_every_nth
                
                # Convert to samples
                start_sample = int(start_time * audio_sample_rate)
                end_sample = min(int(end_time * audio_sample_rate), audio_waveform.shape[1])
                
                # Slice the audio
                if end_sample > start_sample:
                    audio_waveform = audio_waveform[:, start_sample:end_sample]
                    audio_waveform = audio_waveform.unsqueeze(0)  # Add batch dimension
                else:
                    audio_waveform = torch.zeros((1, 2, 1))
            except Exception as e:
                print(f"Audio extraction failed: {str(e)}")
            
            while cap.isOpened() and (frame_load_cap == 0 or frame_count < frame_load_cap):
                if time_offset < target_frame_time:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    time_offset += base_frame_time
                    total_frame_count += 1
                    
                    if time_offset < target_frame_time:
                        continue
                    
                    time_offset -= target_frame_time
                    
                    # Skip frames based on parameters
                    if total_frame_count <= skip_first_frames:
                        continue
                    
                    # Select every nth frame
                    if (total_frame_count - skip_first_frames - 1) % select_every_nth != 0:
                        continue
                    
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Resize if needed
                    if new_width != width or new_height != height:
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    # Convert to float32 and normalize to 0-1 range
                    frame = frame.astype(np.float32) / 255.0
                    
                    frames.append(frame)
                    frame_count += 1
            
            # Close video file
            cap.release()
            
            # Cleanup temporary file
            os.unlink(temp_path)
            
            if len(frames) == 0:
                raise ValueError("No frames extracted from video")
            
            # Convert frames to tensor
            frames_tensor = torch.from_numpy(np.array(frames))
            
            # Create video info dictionary
            video_info = {
                "source_fps": fps,
                "source_frame_count": total_frames,
                "source_duration": duration,
                "source_width": width,
                "source_height": height,
                "loaded_fps": 1 / target_frame_time,
                "loaded_frame_count": len(frames),
                "loaded_duration": len(frames) * target_frame_time,
                "loaded_width": new_width,
                "loaded_height": new_height,
            }
            
            # Format audio for return
            audio_data = {"waveform": audio_waveform, "sample_rate": audio_sample_rate}
            
            return (frames_tensor, len(frames), audio_data, video_info)
            
        except Exception as e:
            print(f"Error loading video from URL: {str(e)}")
            # Return empty frame in case of error
            empty_frame = torch.zeros((1, 512, 512, 3))
            empty_audio = {"waveform": torch.zeros((1, 2, 1)), "sample_rate": 44100}
            empty_info = {
                "source_fps": 0,
                "source_frame_count": 0,
                "source_duration": 0,
                "source_width": 0,
                "source_height": 0,
                "loaded_fps": 0,
                "loaded_frame_count": 0,
                "loaded_duration": 0,
                "loaded_width": 0,
                "loaded_height": 0,
            }
            return (empty_frame, 1, empty_audio, empty_info)

class VideoPreview:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE",),
                "fps": ("INT", {"default": 8, "min": 1, "max": 60, "step": 1}),
                "video_info": ("VHS_VIDEOINFO",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    
    RETURN_TYPES = ()
    FUNCTION = "preview_video"
    OUTPUT_NODE = True
    CATEGORY = "MediaURLLoader"
    
    def preview_video(self, frames, fps, video_info, prompt=None, extra_pnginfo=None):
        try:
            # Get temporary directory for video preview
            temp_dir = folder_paths.get_temp_directory()
            
            # Generate a unique filename
            import time
            unique_id = hashlib.md5(str(time.time()).encode()).hexdigest()
            filename = f"video_preview_{unique_id}.mp4"
            filepath = os.path.join(temp_dir, filename)
            
            # Convert frames to 8-bit format
            frames_np = (frames.cpu().numpy() * 255).astype(np.uint8)
            
            # Get video dimensions
            height, width = frames_np[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
            
            # Write frames to video
            for frame in frames_np:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
            
            # Release the writer
            writer.release()
            
            # Return metadata for UI to display video player
            return {"ui": {"video": [{"filename": filename, "subfolder": "", "type": "temp"}]}}
            
        except Exception as e:
            print(f"Error previewing video: {str(e)}")
            return {"ui": {"video": []}}

# Register nodes
NODE_CLASS_MAPPINGS = {
    "AudioURLLoader": AudioURLLoader,
    "AudioPreview": AudioPreview,
    "VideoURLLoader": VideoURLLoader,
    "VideoPreview": VideoPreview,
}

# Node display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioURLLoader": "Load Audio From URL",
    "AudioPreview": "Preview Audio",
    "VideoURLLoader": "Load Video From URL",
    "VideoPreview": "Preview Video",
} 