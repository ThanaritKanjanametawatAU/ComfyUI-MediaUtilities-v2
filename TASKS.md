# ComfyUI-MediaUtilities-v2 Custom Nodes

## Project Overview
Create a ComfyUI extension that provides nodes for loading and previewing media from URLs:
- Load audio from URL
- Preview audio
- Load video from URL 
- Preview video
- Save audio to file
- Save video to file

## Tasks

### 1. Project Setup
- [x] Create directory structure for the custom node
- [x] Create __init__.py to register the nodes
- [x] Setup requirements.txt with necessary dependencies

### 2. Audio URL Loader Node
- [x] Research ComfyUI node structure
- [x] Implement AudioURLLoader class
- [x] Add URL input field
- [x] Implement audio downloading functionality
- [x] Define output format compatible with ComfyUI
- [x] Add error handling

### 3. Audio Preview Node
- [x] Research ComfyUI audio preview capabilities
- [x] Implement AudioPreview class
- [x] Create UI for audio player
- [x] Handle audio data from loader node
- [x] Implement playback controls

### 4. Video URL Loader Node
- [x] Implement VideoURLLoader class
- [x] Add URL input field
- [x] Implement video downloading functionality
- [x] Define output format compatible with ComfyUI
- [x] Add error handling for various formats

### 5. Video Preview Node
- [x] Research ComfyUI video preview capabilities
- [x] Implement VideoPreview class
- [x] Create UI for video player
- [x] Handle video data from loader node
- [x] Implement playback controls
- [x] Fix video preview not appearing in workspace issue with the following improvements:
  - [x] Use more browser-compatible codecs (H.264/avc1)
  - [x] Add proper file registration with ComfyUI's file service
  - [x] Add fallback mechanisms for different video formats
  - [x] Add GIF fallback option when video codecs fail
  - [x] Add static image fallback option as last resort
  - [x] Improve error handling and debugging

### 6. Save Audio Node
- [ ] Research ComfyUI save file capabilities
- [ ] Implement SaveAudio class
- [ ] Add output path and filename configuration
- [ ] Add format options (mp3, wav, flac)
- [ ] Implement save functionality
- [ ] Add progress indication
- [ ] Add error handling

### 7. Save Video Node
- [ ] Research ComfyUI save file capabilities
- [ ] Implement SaveVideo class
- [ ] Add output path and filename configuration
- [ ] Add format and codec options
- [ ] Add quality/bitrate settings
- [ ] Implement audio+video merging functionality
- [ ] Add progress indication
- [ ] Add error handling

### 8. Testing
- [ ] Test with various audio URL formats
- [ ] Test with various video URL formats
- [ ] Test error handling
- [ ] Test UI responsiveness
- [ ] Test save functionality with different formats

### 9. Documentation
- [x] Create README.md with installation instructions
- [x] Document usage of each node
- [x] Provide examples
- [x] Add screenshots
- [ ] Update documentation for new save nodes

### 10. Packaging
- [x] Finalize directory structure
- [x] Update requirements.txt
- [x] Create installation script if needed

## Recent Fixes

### Video Preview Enhancement (June 2025)
The VideoPreview node has been enhanced to fix issues where videos wouldn't appear in the workspace despite showing in the queue sidebar. The following improvements were made:

1. Added multiple codec support with automatic fallbacks:
   - H.264 (avc1) for primary compatibility
   - MPEG-4 (mp4v) as first fallback
   - Motion JPEG (MJPG) as secondary fallback
   - GIF creation as tertiary fallback
   - Static image (first frame) as last resort fallback

2. Improved file handling:
   - Better registration with ComfyUI's file system
   - Proper temp file management
   - Added verification that created files exist

3. Enhanced debugging and error reporting:
   - Added detailed logging of codec attempts
   - File size verification
   - Clear error messages for troubleshooting

These changes ensure that video previews will work reliably across different browsers and platforms.
