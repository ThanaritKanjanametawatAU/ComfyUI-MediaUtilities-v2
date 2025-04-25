# ComfyUI-MediaURLLoader

Custom nodes for loading and previewing media from URLs in ComfyUI.

## Features

- **Load Audio From URL**: Download and load audio files from the internet
- **Preview Audio**: Play downloaded audio in the ComfyUI interface
- **Load Video From URL**: Download and extract frames from videos on the internet
- **Preview Video**: Play extracted video frames in the ComfyUI interface

## Installation

### Manual Installation

1. Clone this repository into your ComfyUI custom_nodes directory:
```
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/ComfyUI-MediaURLLoader.git
```

2. Install the required dependencies:
```
cd ComfyUI-MediaURLLoader
pip install -r requirements.txt
```

3. Restart ComfyUI

## Usage

### Load Audio From URL

This node allows you to load audio files directly from a URL.

**Inputs:**
- `url`: URL to the audio file (.mp3, .wav, .flac, .ogg, .m4a)

**Outputs:**
- `AUDIO`: Audio data that can be used with other ComfyUI audio nodes

### Preview Audio

This node creates an audio player in the UI to preview audio data.

**Inputs:**
- `audio`: Audio data from an audio loader node

### Load Video From URL

This node allows you to load video files directly from a URL and extract frames.

**Inputs:**
- `url`: URL to the video file (.mp4, .webm, .mkv, .mov, .avi)
- `force_rate`: Force a specific frame rate (0 to use original)
- `force_size`: Resize video to specific dimensions
- `custom_width`: Width for custom resizing
- `custom_height`: Height for custom resizing
- `frame_load_cap`: Maximum number of frames to load
- `skip_first_frames`: Number of frames to skip from the beginning
- `select_every_nth`: Select every nth frame

**Outputs:**
- `FRAMES`: Extracted video frames
- `frame_count`: Number of extracted frames
- `audio`: Audio track extracted from the video
- `video_info`: Video metadata (fps, duration, etc.)

### Preview Video

This node creates a video player in the UI to preview extracted frames.

**Inputs:**
- `frames`: Frames from a video loader node
- `fps`: Playback frame rate
- `video_info`: Video metadata

## Example Workflows

### Audio Loading and Preview
1. Add a "Load Audio From URL" node
2. Enter the URL of an audio file
3. Connect the output to a "Preview Audio" node
4. Execute the workflow to download and play the audio

### Video Loading and Preview
1. Add a "Load Video From URL" node
2. Enter the URL of a video file
3. Configure the frame rate, size, and other options as needed
4. Connect the "FRAMES" output to a "Preview Video" node
5. Execute the workflow to download the video and preview the frames

## License

MIT

## Credits

This project was inspired by:

- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 