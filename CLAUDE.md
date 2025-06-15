# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-powered Instagram reel generator that creates professional short-form videos (1080x1920) with AI-generated images, captions, and optional background music. The tool uses OpenAI's GPT models for content generation and image creation.

## Common Commands

### Running the Tool
```bash
# Basic usage
python main.py "Your topic here"

# With background music
python main.py "Ocean mysteries" --music assets/ambient.ogg

# Debug mode (no API calls, uses mock data)
python main.py "Test topic" --debug

# Custom output directory
python main.py "Travel tips" --out-dir my-reels

# Using mock data directory
python main.py "Test" --debug --mock-dir test-data
```

### Setup and Dependencies
```bash
# Install dependencies
pip install -r requirements.txt

# Environment setup
# Create .env file with: OPENAI_API_KEY=your_key_here
```

## Architecture

### Core Components

- **main.py**: Single-file application containing all functionality
- **Content Generation Pipeline**:
  1. `outline_reel()` - Uses GPT-4o-mini to create scene structure
  2. `generate_image()` - Creates images with gpt-image-1 (fallback to DALL-E-3)
  3. `build_clip()` - Assembles video clips with MoviePy
  4. `create_title_card()` - Generates branded intro cards
  5. `make_reel()` - Orchestrates the complete pipeline

### Video Structure
- **Title Card**: 3-second intro with topic branding
- **Scenes**: 6 seconds each, typically 6-8 scenes total
- **Format**: 1080x1920 portrait (Instagram optimized)
- **Content Flow**: Hook → Build → Value → Conclusion

### Key Dependencies
- **OpenAI**: GPT-4o-mini for text, gpt-image-1/DALL-E-3 for images
- **MoviePy**: Video composition, text overlays, audio processing
- **PIL/Pillow**: Image processing and manipulation
- **FFmpeg**: Required for video rendering (h264_nvenc codec)

### Error Handling
- Automatic fallback from gpt-image-1 to DALL-E-3 for image generation
- Font fallback system (Arial-bold.ttf → system default)
- Scene-level error recovery (skips failed scenes, continues processing)
- Debug mode with mock data for development without API costs

### File Structure
- `assets/`: Background music files (.mp3, .ogg)
- `reels/`: Generated video outputs (auto-created)
- `Arial-bold.ttf`: Custom font for text rendering
- `.env`: OpenAI API key configuration

## Development Notes

- The application is designed as a single-file tool for portability
- Uses environment variables for API key management
- Implements comprehensive error handling with graceful degradation
- Debug mode enables development without API costs
- Cost-efficient: ~$0.30-0.50 per 6-scene reel