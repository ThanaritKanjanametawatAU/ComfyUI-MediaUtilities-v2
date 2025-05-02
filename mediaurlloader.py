import os
import torch
import numpy as np
import requests
import tempfile
import cv2
import torchaudio
import hashlib
import folder_paths
import json
import io
import time
from urllib.parse import urlparse
from io import BytesIO
from comfy.utils import common_upscale
from comfy.cli_args import args

# Enable debug logging
DEBUG = True  # Set to False in production

# Supported media types
AUDIO_EXTENSIONS = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
VIDEO_EXTENSIONS = ['.mp4', '.webm', '.mkv', '.mov', '.avi']

# Codec options to try, in order of preference
VIDEO_CODEC_OPTIONS = [
    ('avc1', '.mp4'),  # H.264, widely supported
    ('mp4v', '.mp4'),  # MPEG-4, good compatibility 
    ('XVID', '.avi'),  # XVID, widely compatible
    ('MJPG', '.avi'),  # Motion JPEG, last resort
]

# Helper for video previews
def create_preview_video(frames_np, fps, temp_dir, prefix="video_preview"):
    """
    Create a preview video with fallback options.
    
    Args:
        frames_np: Numpy array of video frames
        fps: Frames per second
        temp_dir: Directory to save the video
        prefix: Filename prefix
    
    Returns:
        (success, filename, filepath): Success status, filename and full path
    """
    # Generate a unique filename
    unique_id = hashlib.md5(str(time.time()).encode()).hexdigest()
    filename = f"{prefix}_{unique_id}.mp4"
    filepath = os.path.join(temp_dir, filename)
    
    # Register with ComfyUI
    try:
        folder_paths.add_temp_file(filename)
        if DEBUG:
            print(f"Registered temp file: {filename}")
    except Exception as e:
        if DEBUG:
            print(f"Error registering temp file: {str(e)}")
    
    height, width = frames_np[0].shape[:2]
    success = False
    
    # Codec options to try, in order of preference
    codec_options = [
        ('avc1', '.mp4'),  # H.264, widely supported
        ('mp4v', '.mp4'),  # MPEG-4, good compatibility
        ('MJPG', '.avi'),  # Motion JPEG, last resort
    ]
    
    # Try each codec option
    for codec, ext in codec_options:
        if ext != '.mp4':
            # Update filename for different extensions
            filename = f"{prefix}_{unique_id}{ext}"
            filepath = os.path.join(temp_dir, filename)
            folder_paths.add_temp_file(filename)
        
        try:
            if DEBUG:
                print(f"Trying codec {codec} for {filepath}")
                
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
            
            if writer.isOpened():
                # Write frames to video
                for frame in frames_np:
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writer.write(frame_bgr)
                
                # Release the writer
                writer.release()
                success = True
                
                if DEBUG:
                    print(f"Successfully created video with {codec} codec at {filepath}")
                    print(f"Video size: {os.path.getsize(filepath)} bytes")
                    
                break  # Exit loop if successful
            else:
                if DEBUG:
                    print(f"Failed to open VideoWriter with {codec} codec")
        except Exception as e:
            if DEBUG:
                print(f"Error with {codec} codec: {str(e)}")
    
    # Last resort: try creating a GIF if all video codecs failed
    if not success:
        try:
            # Check if PIL/Pillow is available
            try:
                from PIL import Image
                pil_available = True
            except ImportError:
                pil_available = False
                if DEBUG:
                    print("PIL not available, skipping GIF fallback")
            
            if pil_available:
                if DEBUG:
                    print("Attempting to create GIF as last resort")
                
                # Update filename for GIF
                filename = f"{prefix}_{unique_id}.gif"
                filepath = os.path.join(temp_dir, filename)
                folder_paths.add_temp_file(filename)
                
                # Convert frames to PIL Images
                pil_frames = []
                for frame in frames_np:
                    pil_frame = Image.fromarray(frame)
                    pil_frames.append(pil_frame)
                
                # Calculate duration (in milliseconds)
                duration = int(1000 / fps)
                
                # Save as GIF
                if len(pil_frames) > 0:
                    pil_frames[0].save(
                        filepath,
                        format='GIF',
                        append_images=pil_frames[1:],
                        save_all=True,
                        duration=duration,
                        loop=0,  # 0 means loop forever
                        optimize=False
                    )
                    success = True
                    if DEBUG:
                        print(f"Successfully created GIF at {filepath}")
                        print(f"GIF size: {os.path.getsize(filepath)} bytes")
        except Exception as e:
            if DEBUG:
                print(f"Error creating GIF: {str(e)}")
    
    return success, filename, filepath

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
            
            # Convert frames to numpy array
            frames_np = (frames.cpu().numpy() * 255).astype(np.uint8)
            
            # Use our helper function to create the video file
            success, filename, filepath = create_preview_video(frames_np, fps, temp_dir)
            
            if not success:
                if DEBUG:
                    print("All video codec options failed. Attempting to extract a frame as image.")
                
                # Last resort - create a single image from the first frame
                try:
                    if len(frames_np) > 0:
                        frame = frames_np[0]
                        unique_id = hashlib.md5(str(time.time()).encode()).hexdigest()
                        img_filename = f"preview_frame_{unique_id}.png"
                        img_filepath = os.path.join(temp_dir, img_filename)
                        
                        # Save the first frame as an image
                        import cv2
                        # Convert RGB to BGR for OpenCV
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(img_filepath, frame_bgr)
                        
                        folder_paths.add_temp_file(img_filename)
                        
                        if DEBUG:
                            print(f"Created image preview from first frame: {img_filepath}")
                        
                        # Return image preview instead of video
                        return {"ui": {"images": [{"filename": img_filename, "subfolder": "", "type": "temp"}]}}
                except Exception as e:
                    if DEBUG:
                        print(f"Error creating image preview: {str(e)}")
                
                return {"ui": {"video": []}}
            
            # Return metadata for UI to display video player
            video_info = {"filename": filename, "subfolder": "", "type": "temp"}
            if DEBUG:
                print(f"Returning video preview info: {video_info}")
                
            # Add additional information to facilitate debugging
            if os.path.exists(filepath):
                if DEBUG:
                    print(f"Video file exists at {filepath} with size {os.path.getsize(filepath)} bytes")
            else:
                if DEBUG:
                    print(f"WARNING: Video file {filepath} doesn't exist, though codec reported success")
            
            return {"ui": {"video": [video_info]}}
            
        except Exception as e:
            if DEBUG:
                print(f"Error previewing video: {str(e)}")
            return {"ui": {"video": []}}

class SaveAudio:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "audio/media_url_loader"}),
                "format": (["mp3", "wav", "flac", "ogg"], {"default": "mp3"}),
                "sample_rate": ("INT", {"default": 44100, "min": 8000, "max": 192000, "step": 100}),
                "normalize": (["True", "False"], {"default": "True"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save_audio"
    OUTPUT_NODE = True
    CATEGORY = "MediaURLLoader"
    
    def save_audio(self, audio, filename_prefix, format, sample_rate, normalize, prompt=None, extra_pnginfo=None):
        try:
            # Create output directory if it doesn't exist
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
            
            results = []
            metadata = {}
            
            # Add metadata if enabled
            if not args.disable_metadata:
                if prompt is not None:
                    metadata["prompt"] = json.dumps(prompt)
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata[x] = json.dumps(extra_pnginfo[x])
            
            # Process each waveform in the batch
            for batch_number, waveform in enumerate(audio["waveform"].cpu()):
                # Replace batch placeholder if it exists
                filename_with_batch = filename.replace("%batch_num%", str(batch_number))
                file = f"{filename_with_batch}_{counter:05}.{format}"
                file_path = os.path.join(full_output_folder, file)
                
                # Resample if needed
                if audio["sample_rate"] != sample_rate:
                    waveform = torchaudio.functional.resample(waveform, audio["sample_rate"], sample_rate)
                
                # Normalize audio if enabled
                if normalize == "True":
                    max_val = torch.max(torch.abs(waveform))
                    if max_val > 0:
                        waveform = waveform / max_val * 0.9  # Leave some headroom
                
                # Save audio in the specified format
                if format.lower() == "flac":
                    # FLAC with metadata
                    buff = io.BytesIO()
                    torchaudio.save(buff, waveform, sample_rate, format="FLAC")
                    
                    # Add metadata if available
                    if metadata:
                        from .utils import insert_or_replace_vorbis_comment
                        buff = insert_or_replace_vorbis_comment(buff, metadata)
                    
                    with open(file_path, 'wb') as f:
                        f.write(buff.getbuffer())
                elif format.lower() == "mp3":
                    # Explicitly handle MP3 format
                    print(f"Saving audio as MP3 to {file_path}")
                    torchaudio.save(file_path, waveform, sample_rate, format="MP3")
                    # Verify file was created with correct extension
                    if not os.path.exists(file_path):
                        print(f"Error: MP3 file not created at {file_path}")
                else:
                    # Other formats
                    torchaudio.save(file_path, waveform, sample_rate, format=format.upper())
                
                # Log success and add to results
                print(f"Saved audio as {file_path}")
                results.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": "output"
                })
                
                counter += 1
            
            # Return UI info for audio preview
            return {"ui": {"audio": results}}
            
        except Exception as e:
            print(f"Error saving audio: {str(e)}")
            return {"ui": {"audio": []}}

class SaveVideo:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE",),
                "fps": ("INT", {"default": 30, "min": 1, "max": 120, "step": 1}),
                "filename_prefix": ("STRING", {"default": "video/media_url_loader"}),
                "format": (["mp4", "webm", "avi", "mov", "gif"], {"default": "mp4"}),
                "codec": (["auto", "h264", "vp9", "avc1", "xvid", "mjpg"], {"default": "auto"}),
                "quality": ("INT", {"default": 85, "min": 1, "max": 100, "step": 1}),
                "convert_from_gif": (["True", "False"], {"default": "True"}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "pingpong": (["True", "False"], {"default": "False"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    CATEGORY = "MediaURLLoader"
    
    def save_video(self, frames, fps, filename_prefix, format, codec, quality, convert_from_gif="True", audio=None, loop_count=0, pingpong="False", prompt=None, extra_pnginfo=None):
        try:
            # Handle parameter validation issues
            # Convert convert_from_gif to proper string format
            if isinstance(convert_from_gif, str):
                convert_from_gif = "True" if convert_from_gif.lower() in ["true", "1"] else "False"
            else:
                convert_from_gif = "True" if convert_from_gif else "False"
            
            # Ensure loop_count is an integer
            if isinstance(loop_count, str):
                try:
                    loop_count = int(loop_count)
                except ValueError:
                    print(f"Warning: Invalid loop_count value '{loop_count}', defaulting to 0")
                    loop_count = 0
            
            # Ensure pingpong is proper string format
            if isinstance(pingpong, str):
                pingpong = "True" if pingpong.lower() in ["true", "1"] else "False"
            else:
                pingpong = "True" if pingpong else "False"
                
            if DEBUG:
                print(f"Starting SaveVideo process with format={format}, codec={codec}, fps={fps}")
                print(f"Parameters: convert_from_gif={convert_from_gif}, loop_count={loop_count}, pingpong={pingpong}")
                
            # Create output directory if it doesn't exist
            try:
                full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
                if DEBUG:
                    print(f"Output directory: {full_output_folder}")
            except Exception as e:
                print(f"[ERROR] Failed to create or access output directory: {str(e)}")
                raise
            
            # Track if we used GIF fallback but wanted another format
            used_gif_fallback = False
            original_format = format
            
            # Create full output path (this will be updated if format changes)
            output_file = f"{filename}_{counter:05}.{format}"
            output_path = os.path.join(full_output_folder, output_file)
            
            # Convert frames to numpy array
            try:
                if DEBUG:
                    print(f"Converting frames tensor of shape {frames.shape} to numpy array")
                frames_np = (frames.cpu().numpy() * 255).astype(np.uint8)
                
                # Get dimensions
                height, width = frames_np[0].shape[:2]
                if DEBUG:
                    print(f"Video dimensions: {width}x{height}, frames: {len(frames_np)}")
            except Exception as e:
                print(f"[ERROR] Failed to process input frames: {str(e)}")
                raise
            
            # Apply pingpong effect if requested
            if pingpong == "True":
                try:
                    if DEBUG:
                        print("Applying pingpong effect")
                    reversed_frames = frames_np[::-1]
                    if len(reversed_frames) > 2:
                        frames_np = np.concatenate([frames_np, reversed_frames[1:-1]])
                        if DEBUG:
                            print(f"After pingpong: {len(frames_np)} frames")
                except Exception as e:
                    print(f"[ERROR] Failed to apply pingpong effect: {str(e)}")
            # Apply looping if requested
            if loop_count > 0:
                try:
                    if DEBUG:
                        print(f"Applying loop effect {loop_count} times")
                    frames_np = np.tile(frames_np, (loop_count + 1, 1, 1, 1))
                    if DEBUG:
                        print(f"After looping: {len(frames_np)} frames")
                except Exception as e:
                    print(f"[ERROR] Failed to apply loop effect: {str(e)}")

            # --- CUSTOM MP4 LOGIC ---
            if format == "mp4":
                if DEBUG:
                    print("[MP4 MODE] Creating GIF first, then converting to MP4 via ffmpeg.")
                # Step 1: Save as GIF
                try:
                    import imageio
                    gif_output_file = f"{filename}_{counter:05}.gif"
                    gif_output_path = os.path.join(full_output_folder, gif_output_file)
                    imageio_frames = [frame for frame in frames_np]
                    imageio.mimsave(gif_output_path, imageio_frames, fps=fps, loop=0)
                    if DEBUG:
                        print(f"[MP4 MODE] Successfully saved GIF: {gif_output_path} ({os.path.getsize(gif_output_path)} bytes)")
                except ImportError:
                    try:
                        from PIL import Image
                        gif_output_file = f"{filename}_{counter:05}.gif"
                        gif_output_path = os.path.join(full_output_folder, gif_output_file)
                        pil_frames = [Image.fromarray(frame) for frame in frames_np]
                        duration = int(1000 / fps)
                        if len(pil_frames) > 0:
                            pil_frames[0].save(
                                gif_output_path,
                                format='GIF',
                                append_images=pil_frames[1:],
                                save_all=True,
                                duration=duration,
                                loop=0,
                                optimize=False
                            )
                            if DEBUG:
                                print(f"[MP4 MODE] Successfully saved GIF with PIL: {gif_output_path} ({os.path.getsize(gif_output_path)} bytes)")
                        else:
                            raise ValueError("No frames to save")
                    except ImportError:
                        print("[ERROR] Neither imageio nor PIL is available. Please install one of them with: pip install imageio")
                        return {"ui": {"video": []}}
                    except Exception as e:
                        print(f"[ERROR] Failed to create GIF with PIL: {str(e)}")
                        return {"ui": {"video": []}}
                except Exception as e:
                    print(f"[ERROR] Failed to create GIF: {str(e)}")
                    return {"ui": {"video": []}}
                # Step 2: Convert GIF to MP4 using ffmpeg
                mp4_output_file = f"{filename}_{counter:05}.mp4"
                mp4_output_path = os.path.join(full_output_folder, mp4_output_file)
                try:
                    import subprocess
                    # Check ffmpeg availability
                    test_cmd = ["ffmpeg", "-version"]
                    test_result = subprocess.run(test_cmd, capture_output=True)
                    if test_result.returncode != 0:
                        print("[ERROR] ffmpeg not found in PATH, cannot convert GIF to MP4.")
                        return {"ui": {"video": []}}
                    # Build ffmpeg command
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-i", gif_output_path,
                        "-vf", f"fps={fps}",
                        "-c:v", "libx264",
                        "-preset", "medium",
                        "-crf", "23",
                        "-pix_fmt", "yuv420p",
                        mp4_output_path
                    ]
                    if DEBUG:
                        print(f"[MP4 MODE] Running ffmpeg command: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True)
                    if result.returncode != 0:
                        stderr = result.stderr.decode() if hasattr(result.stderr, 'decode') else str(result.stderr)
                        print(f"[ERROR] ffmpeg failed to convert GIF to MP4: {stderr}")
                        print(f"[INFO] GIF file is available at: {gif_output_path}")
                        return {"ui": {"video": []}}
                    if DEBUG:
                        print(f"[MP4 MODE] Successfully converted GIF to MP4: {mp4_output_path} ({os.path.getsize(mp4_output_path)} bytes)")
                except Exception as e:
                    print(f"[ERROR] Exception during GIF to MP4 conversion: {str(e)}")
                    print(f"[INFO] GIF file is available at: {gif_output_path}")
                    return {"ui": {"video": []}}
                # Step 3: Add audio if provided
                if audio is not None:
                    try:
                        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                        temp_audio_path = temp_audio.name
                        temp_audio.close()
                        torchaudio.save(temp_audio_path, audio["waveform"][0], audio["sample_rate"])
                        temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                        temp_output_path = temp_output.name
                        temp_output.close()
                        cmd = [
                            "ffmpeg",
                            "-y",
                            "-i", mp4_output_path,
                            "-i", temp_audio_path,
                            "-c:v", "copy",
                            "-c:a", "aac",
                            "-shortest",
                            temp_output_path
                        ]
                        if DEBUG:
                            print(f"[MP4 MODE] Adding audio with ffmpeg: {' '.join(cmd)}")
                        result = subprocess.run(cmd, capture_output=True)
                        if result.returncode == 0:
                            import shutil
                            shutil.move(temp_output_path, mp4_output_path)
                            if DEBUG:
                                print(f"[MP4 MODE] Successfully added audio to MP4: {mp4_output_path}")
                        else:
                            stderr = result.stderr.decode() if hasattr(result.stderr, 'decode') else str(result.stderr)
                            print(f"[ERROR] ffmpeg failed to add audio: {stderr}")
                        try:
                            os.unlink(temp_audio_path)
                        except Exception as e:
                            print(f"[WARNING] Failed to clean up temporary audio file: {str(e)}")
                    except Exception as e:
                        print(f"[ERROR] Failed to add audio to MP4: {str(e)}")
                # Step 4: Return mp4 file info
                filename_only = os.path.basename(mp4_output_path)
                file_ext = os.path.splitext(filename_only)[1].lower()
                format_type = 'video/h264-mp4'
                workflow_name = f"{filename_only.split('.')[0]}.png"
                video_info = {
                    "filename": filename_only,
                    "subfolder": subfolder,
                    "type": "output",
                    "format": format_type,
                    "frame_rate": float(fps),
                    "workflow": workflow_name,
                    "fullpath": mp4_output_path
                }
                print(f"Saved video as {mp4_output_path}")
                return {
                    "ui": {"video": [video_info]},
                    "gifs": [video_info]
                }
            # --- END CUSTOM MP4 LOGIC ---

            # Save as GIF directly or if requested
            if format == "gif":
                if DEBUG:
                    print("Saving as GIF format")
                    try:
                        try:
                            import imageio
                            if DEBUG:
                                print("Using imageio for GIF creation")
                            
                            # Convert frames for imageio (RGB order)
                            imageio_frames = [frame for frame in frames_np]
                            
                            # Save as GIF
                            imageio.mimsave(output_path, imageio_frames, fps=fps, loop=0)
                            if DEBUG:
                                print(f"Successfully saved GIF with imageio: {output_path} ({os.path.getsize(output_path)} bytes)")
                            
                        except ImportError:
                            print("WARNING: imageio is required for GIF output. Falling back to PIL if available.")
                            # Try PIL as fallback
                            try:
                                from PIL import Image
                                if DEBUG:
                                    print("Using PIL for GIF creation")
                                    
                                # Convert frames to PIL Images
                                pil_frames = []
                                for frame in frames_np:
                                    pil_frame = Image.fromarray(frame)
                                    pil_frames.append(pil_frame)
                                
                                # Calculate duration (in milliseconds)
                                duration = int(1000 / fps)
                                
                                # Save as GIF
                                if len(pil_frames) > 0:
                                    pil_frames[0].save(
                                        output_path,
                                        format='GIF',
                                        append_images=pil_frames[1:],
                                        save_all=True,
                                        duration=duration,
                                        loop=0,  # 0 means loop forever
                                        optimize=False
                                    )
                                    if DEBUG:
                                        print(f"Successfully saved GIF with PIL: {output_path} ({os.path.getsize(output_path)} bytes)")
                                else:
                                    raise ValueError("No frames to save")
                            except ImportError:
                                print("[ERROR] Neither imageio nor PIL is available. Please install one of them with: pip install imageio")
                                raise
                            except Exception as e:
                                print(f"[ERROR] Failed to create GIF with PIL: {str(e)}")
                                raise
                    except Exception as e:
                        print(f"[ERROR] Failed to create GIF: {str(e)}")
                        return {"ui": {"video": []}}

            # Save as video
            else:
                if DEBUG:
                    print(f"Saving as video format: {format}")
                    
                # Track if any codec succeeded
                success = False
                
                # Map between user-selected codec and fourcc strings
                codec_map = {
                    "h264": "avc1",   # H.264 codec
                    "vp9": "VP90",    # WebM codec
                    "avc1": "avc1",   # H.264 codec (alias)
                    "xvid": "XVID",   # AVI codec
                    "mjpg": "MJPG",   # Motion JPEG
                    "auto": None      # Will use fallback logic
                }
                
                # If the user selected a specific codec, try that first
                if codec != "auto":
                    fourcc_to_try = [(codec_map.get(codec, "avc1"), f".{format}")]
                    if DEBUG:
                        print(f"Using user-selected codec: {codec} (mapped to {codec_map.get(codec, 'avc1')})")
                else:
                    # Start with codec options appropriate for the selected format
                    if format == "mp4":
                        fourcc_to_try = [('avc1', '.mp4'), ('mp4v', '.mp4')]
                    elif format == "webm":
                        fourcc_to_try = [('VP90', '.webm')]
                    elif format == "avi":
                        fourcc_to_try = [('XVID', '.avi'), ('MJPG', '.avi')]
                    elif format == "mov":
                        fourcc_to_try = [('avc1', '.mov'), ('mp4v', '.mov')]
                    else:
                        fourcc_to_try = [('avc1', '.mp4')]  # Default
                    
                    if DEBUG:
                        print(f"Using auto codec selection with format-specific options: {fourcc_to_try}")
                    
                    # Then add fallbacks from global options that have different extensions
                    fallbacks = []
                    for codec_opt, ext in VIDEO_CODEC_OPTIONS:
                        if (codec_opt, ext) not in fourcc_to_try:
                            fallbacks.append((codec_opt, ext))
                    
                    if DEBUG and fallbacks:
                        print(f"Added fallback codecs: {fallbacks}")
                        
                    fourcc_to_try.extend(fallbacks)
                
                # Variables to store the error messages
                codec_errors = []
                
                # Try each codec option until one works
                for codec_index, (fourcc_codec, ext) in enumerate(fourcc_to_try):
                    # Update filename if extension changes
                    current_output_path = output_path
                    if f".{format}" != ext:
                        output_file = f"{filename}_{counter:05}{ext}"
                        current_output_path = os.path.join(full_output_folder, output_file)
                        if DEBUG:
                            print(f"[Attempt {codec_index+1}/{len(fourcc_to_try)}] Trying alternate format: {ext} with codec {fourcc_codec}")
                    else:
                        if DEBUG:
                            print(f"[Attempt {codec_index+1}/{len(fourcc_to_try)}] Trying codec {fourcc_codec} for {current_output_path}")
                    
                    try:
                        # Create video writer with codec
                        fourcc = cv2.VideoWriter_fourcc(*fourcc_codec)
                        writer = cv2.VideoWriter(current_output_path, fourcc, fps, (width, height))
                        
                        # Check if writer was opened successfully
                        if writer.isOpened():
                            if DEBUG:
                                print(f"VideoWriter opened successfully with {fourcc_codec}")
                                
                            # Quality setting (only affects some codecs)
                            if hasattr(writer, 'set'):
                                writer.set(cv2.VIDEOWRITER_PROP_QUALITY, quality)
                                if DEBUG:
                                    print(f"Set quality to {quality}")
                            
                            # Write frames to video
                            frame_count = 0
                            for frame in frames_np:
                                # Convert RGB to BGR for OpenCV
                                try:
                                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                    writer.write(frame_bgr)
                                    frame_count += 1
                                except Exception as e:
                                    print(f"[ERROR] Failed to write frame {frame_count}: {str(e)}")
                                    raise
                            
                            # Release the writer
                            writer.release()
                            success = True
                            
                            if DEBUG:
                                print(f"Successfully saved video with {fourcc_codec} codec at {current_output_path}")
                                print(f"Video size: {os.path.getsize(current_output_path)} bytes, frames written: {frame_count}")
                            
                            # Update output path for any subsequent operations
                            output_path = current_output_path
                            break  # Exit loop if successful
                        else:
                            error_msg = f"Failed to open VideoWriter with {fourcc_codec} codec"
                            codec_errors.append(f"{fourcc_codec}: {error_msg}")
                            if DEBUG:
                                print(f"[ERROR] {error_msg}")
                                
                    except Exception as e:
                        error_msg = f"Error with {fourcc_codec} codec: {str(e)}"
                        codec_errors.append(f"{fourcc_codec}: {str(e)}")
                        if DEBUG:
                            print(f"[ERROR] {error_msg}")
                
                # If all video codecs failed, try GIF as last resort if it wasn't the original format
                if not success and format != "gif":
                    if DEBUG:
                        print(f"All {len(fourcc_to_try)} video codecs failed. Errors: {'; '.join(codec_errors)}")
                        print("Attempting to create GIF as fallback")
                    
                    try:
                        # Try using imageio for GIF fallback
                        try:
                            import imageio
                            
                            # Update filename for GIF
                            gif_output_file = f"{filename}_{counter:05}.gif"
                            gif_output_path = os.path.join(full_output_folder, gif_output_file)
                            
                            if DEBUG:
                                print(f"Creating GIF fallback with imageio: {gif_output_path}")
                            
                            # Convert frames for imageio
                            imageio_frames = [frame for frame in frames_np]
                            
                            # Save as GIF
                            imageio.mimsave(gif_output_path, imageio_frames, fps=fps, loop=0)
                            success = True
                            used_gif_fallback = True
                            
                            # Update output path to the GIF
                            output_path = gif_output_path
                            
                            if DEBUG:
                                print(f"Successfully created GIF fallback at {gif_output_path}")
                                print(f"GIF size: {os.path.getsize(gif_output_path)} bytes")
                                
                        except ImportError:
                            # Try using PIL if imageio is not available
                            try:
                                from PIL import Image
                                
                                # Update filename for GIF
                                gif_output_file = f"{filename}_{counter:05}.gif"
                                gif_output_path = os.path.join(full_output_folder, gif_output_file)
                                
                                if DEBUG:
                                    print(f"Creating GIF fallback with PIL: {gif_output_path}")
                                
                                # Convert frames to PIL Images
                                pil_frames = []
                                for frame in frames_np:
                                    pil_frame = Image.fromarray(frame)
                                    pil_frames.append(pil_frame)
                                
                                # Calculate duration (in milliseconds)
                                duration = int(1000 / fps)
                                
                                # Save as GIF
                                if len(pil_frames) > 0:
                                    pil_frames[0].save(
                                        gif_output_path,
                                        format='GIF',
                                        append_images=pil_frames[1:],
                                        save_all=True,
                                        duration=duration,
                                        loop=0,  # 0 means loop forever
                                        optimize=False
                                    )
                                    success = True
                                    used_gif_fallback = True
                                    
                                    # Update output path to the GIF
                                    output_path = gif_output_path
                                    
                                    if DEBUG:
                                        print(f"Successfully created GIF fallback with PIL: {gif_output_path}")
                                        print(f"GIF size: {os.path.getsize(gif_output_path)} bytes")
                            except ImportError:
                                print("[ERROR] Neither imageio nor PIL available for GIF fallback")
                            except Exception as e:
                                print(f"[ERROR] Failed to create GIF fallback with PIL: {str(e)}")
                    except Exception as e:
                        print(f"[ERROR] Failed to create GIF fallback: {str(e)}")
                
                # If nothing worked, raise an error with detailed information
                if not success:
                    error_detail = "; ".join(codec_errors)
                    error_msg = f"Failed to save video with any available codec. Errors: {error_detail}"
                    print(f"[ERROR] {error_msg}")
                    raise ValueError(f"{error_msg}. Try installing a conda build of OpenCV or use a different format like GIF.")
                
                # Add audio if provided and if video creation was successful
                if audio is not None and success and os.path.splitext(output_path)[1] in [".mp4", ".webm", ".mov"]:
                    if DEBUG:
                        print(f"Adding audio to video {output_path}")
                        
                    try:
                        # Create temporary audio file
                        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                        temp_audio_path = temp_audio.name
                        temp_audio.close()
                        
                        if DEBUG:
                            print(f"Created temporary audio file: {temp_audio_path}")
                        
                        # Save audio to temporary file
                        try:
                            torchaudio.save(temp_audio_path, audio["waveform"][0], audio["sample_rate"])
                            if DEBUG:
                                print(f"Saved audio to temporary file: {os.path.getsize(temp_audio_path)} bytes")
                        except Exception as e:
                            print(f"[ERROR] Failed to save audio to temporary file: {str(e)}")
                            raise
                        
                        # Create temporary output file
                        ext = os.path.splitext(output_path)[1]
                        temp_output = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
                        temp_output_path = temp_output.name
                        temp_output.close()
                        
                        if DEBUG:
                            print(f"Created temporary output file: {temp_output_path}")
                        
                        # Check for ffmpeg
                        try:
                            import subprocess
                            # Test if ffmpeg is available
                            test_cmd = ["ffmpeg", "-version"]
                            test_result = subprocess.run(test_cmd, capture_output=True)
                            if test_result.returncode != 0:
                                print("[WARNING] ffmpeg not found in PATH, audio may not be added")
                        except Exception as e:
                            print(f"[WARNING] Error checking for ffmpeg: {str(e)}")
                        
                        # Use ffmpeg to combine video and audio
                        try:
                            import subprocess
                            cmd = [
                                "ffmpeg",
                                "-y",  # Overwrite output file if it exists
                                "-i", output_path,  # Input video
                                "-i", temp_audio_path,  # Input audio
                                "-c:v", "copy",  # Copy video codec
                                "-c:a", "aac",  # AAC audio codec
                                "-shortest",  # End when shortest input ends
                                temp_output_path  # Output
                            ]
                            
                            if DEBUG:
                                print(f"Running ffmpeg command: {' '.join(cmd)}")
                            
                            # Run ffmpeg
                            result = subprocess.run(cmd, capture_output=True)
                            
                            if result.returncode == 0:
                                # Replace original video with the one with audio
                                import shutil
                                shutil.move(temp_output_path, output_path)
                                if DEBUG:
                                    print(f"Successfully added audio to video: {output_path}")
                            else:
                                stderr = result.stderr.decode() if hasattr(result.stderr, 'decode') else str(result.stderr)
                                print(f"[ERROR] Error adding audio to video: {stderr}")
                        except Exception as e:
                            print(f"[ERROR] Failed to run ffmpeg: {str(e)}")
                        
                        # Clean up temporary audio file
                        try:
                            os.unlink(temp_audio_path)
                            if DEBUG:
                                print(f"Cleaned up temporary audio file: {temp_audio_path}")
                        except Exception as e:
                            print(f"[WARNING] Failed to clean up temporary audio file: {str(e)}")
                            
                    except Exception as e:
                        print(f"[ERROR] Failed to add audio to video: {str(e)}")
                        # Don't fail the whole process if audio addition fails
            
            # If we used GIF fallback but wanted another format, and convert_from_gif is True
            # Try to convert from GIF to the original format using ffmpeg
            if used_gif_fallback and original_format != "gif" and convert_from_gif == "True":
                if DEBUG:
                    print(f"Attempting to convert GIF to {original_format} using ffmpeg")
                
                try:
                    # Check if ffmpeg is available
                    ffmpeg_available = False
                    try:
                        import subprocess
                        test_cmd = ["ffmpeg", "-version"]
                        test_result = subprocess.run(test_cmd, capture_output=True)
                        ffmpeg_available = test_result.returncode == 0
                    except Exception as e:
                        print(f"[WARNING] Error checking for ffmpeg: {str(e)}")
                    
                    if ffmpeg_available:
                        # Define the target path for the converted file
                        converted_output_file = f"{filename}_{counter:05}.{original_format}"
                        converted_output_path = os.path.join(full_output_folder, converted_output_file)
                        
                        # Set ffmpeg params based on format
                        if original_format == "mp4":
                            # Use libx264 for MP4 output with medium preset and high quality
                            vcodec = "libx264"
                            extra_params = ["-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p"]
                        elif original_format == "webm":
                            # Use libvpx for WebM output
                            vcodec = "libvpx"
                            extra_params = ["-b:v", "1M", "-pix_fmt", "yuv420p"]
                        elif original_format == "mov":
                            # Use libx264 for MOV output
                            vcodec = "libx264"
                            extra_params = ["-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p"]
                        elif original_format == "avi":
                            # Use MPEG-4 for AVI
                            vcodec = "mpeg4"
                            extra_params = ["-q:v", "6", "-pix_fmt", "yuv420p"]
                        else:
                            # Default
                            vcodec = "libx264"
                            extra_params = ["-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p"]
                        
                        # Build the ffmpeg command
                        cmd = [
                            "ffmpeg",
                            "-y",  # Overwrite output file if it exists
                            "-i", output_path,  # Input GIF
                            "-vf", f"fps={fps}",  # Set the framerate
                            "-c:v", vcodec,  # Set video codec
                        ]
                        
                        # Add extra parameters
                        cmd.extend(extra_params)
                        
                        # Add output path
                        cmd.append(converted_output_path)
                        
                        if DEBUG:
                            print(f"Running ffmpeg conversion command: {' '.join(cmd)}")
                            
                        # Run ffmpeg
                        result = subprocess.run(cmd, capture_output=True)
                        
                        if result.returncode == 0:
                            if DEBUG:
                                print(f"Successfully converted GIF to {original_format}: {converted_output_path}")
                                print(f"Converted file size: {os.path.getsize(converted_output_path)} bytes")
                            
                            # Update output path to point to the converted file
                            output_path = converted_output_path
                            
                            # Note: we don't delete the GIF, keeping it as a backup
                        else:
                            stderr = result.stderr.decode() if hasattr(result.stderr, 'decode') else str(result.stderr)
                            print(f"[WARNING] Failed to convert GIF to {original_format}: {stderr}")
                            print(f"[INFO] GIF file is still available at: {output_path}")
                    else:
                        print(f"[WARNING] Cannot convert GIF to {original_format}: ffmpeg not available")
                        print(f"[INFO] GIF file is available at: {output_path}")
                except Exception as e:
                    print(f"[WARNING] Error during GIF to {original_format} conversion: {str(e)}")
                    print(f"[INFO] GIF file is still available at: {output_path}")
            
            # Log success
            print(f"Saved video as {output_path}")
            
            # Create results with enhanced metadata for history output
            filename_only = os.path.basename(output_path)
            file_ext = os.path.splitext(filename_only)[1].lower()
            
            # Determine video format type
            if file_ext == '.mp4':
                format_type = 'video/h264-mp4'
            elif file_ext == '.webm':
                format_type = 'video/webm'
            elif file_ext == '.avi':
                format_type = 'video/avi'
            elif file_ext == '.mov':
                format_type = 'video/quicktime'
            elif file_ext == '.gif':
                format_type = 'image/gif'
            else:
                format_type = f'video/{file_ext[1:]}'
            
            # Create workflow image name (for ComfyUI graph thumbnail)
            workflow_name = f"{filename_only.split('.')[0]}.png"
            
            # Enhanced video info dictionary with information needed for history output
            video_info = {
                "filename": filename_only,
                "subfolder": subfolder,
                "type": "output",
                "format": format_type,
                "frame_rate": float(fps),
                "workflow": workflow_name,
                "fullpath": output_path
            }
            
            # Return video metadata for UI preview, plus additional output for history
            return {
                "ui": {"video": [video_info]},
                "gifs": [video_info]  # This is what gets recorded in history
            }
            
        except Exception as e:
            import traceback
            error_with_trace = f"{str(e)}\n{traceback.format_exc()}"
            print(f"[ERROR] Error saving video: {error_with_trace}")
            return {"ui": {"video": []}}

# Register nodes
NODE_CLASS_MAPPINGS = {
    "AudioURLLoader": AudioURLLoader,
    "AudioPreview": AudioPreview,
    "VideoURLLoader": VideoURLLoader,
    "VideoPreview": VideoPreview,
    "SaveAudio": SaveAudio,
    "SaveVideo": SaveVideo,
}

# Node display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioURLLoader": "Load Audio From URL",
    "AudioPreview": "Preview Audio",
    "VideoURLLoader": "Load Video From URL",
    "VideoPreview": "Preview Video",
    "SaveAudio": "Save Audio",
    "SaveVideo": "Save Video",
} 