from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import re
import textwrap
import tempfile
from datetime import datetime
from io import BytesIO
from typing import List

import requests
import numpy as np
from moviepy import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    concatenate_videoclips,
)
from moviepy.video.fx import Resize, Crop
from moviepy.audio.fx import AudioFadeOut, MultiplyVolume
from openai import OpenAI
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path
FONT_PATH = Path(__file__).with_name("Arial-bold.ttf")  # put the TTF in same folder

# Load variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# -----------------------------------------------------------------------------
# OpenAI helpers
# -----------------------------------------------------------------------------

client = OpenAI(api_key=openai_api_key)

_CHAT_DIRECTOR_PROMPT = textwrap.dedent(
    """
    You are a creative director for short‑form social media videos. When given a TOPIC,
    produce an engaging 35–45 second reel broken into 5–7 SCENES. Each scene needs:
    1. an image prompt describing what to show (vivid visual language, no text)
    2. a concise on‑screen caption under 40 characters
    Return ONLY valid JSON in the form:
    {"scenes": [{"prompt": "…", "caption": "…"}, …]}
    """
)


def outline_reel(topic: str,
                 debug: bool = False,
                 mock_dir: str | None = None) -> List[dict]:
    """Ask GPT‑4o to storyboard the reel and return a list of scene dicts."""
    if debug:
        # ── OPTION A: use captions.txt in --mock-dir ───────────────────
        if mock_dir:
            cap_file = Path(mock_dir) / "captions.txt"
            if cap_file.exists():
                with cap_file.open(encoding="utf-8") as fh:
                    captions = [c.strip() for c in fh if c.strip()]
                return [
                    {"prompt": f"mock image {i+1}", "caption": c}
                    for i, c in enumerate(captions)
                ]
        # ── OPTION B: built-in dummy storyboard ───────────────────────
        return [
            {"prompt": "Solid blue background",  "caption": "Hello 🌍"},
            {"prompt": "Solid green background", "caption": "This is debug"},
            {"prompt": "Solid red background",   "caption": "No tokens spent"},
        ]

    try:
        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            max_tokens=800,
            messages=[
                {"role": "system", "content": _CHAT_DIRECTOR_PROMPT},
                {"role": "user", "content": topic},
            ],
        )
        response_content = chat.choices[0].message.content
        parsed_response = json.loads(response_content)
        
        if "scenes" not in parsed_response:
            raise ValueError("Invalid response format: missing 'scenes' key")
            
        return parsed_response["scenes"]
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response from OpenAI: {e}")
    except Exception as e:
        raise ValueError(f"Failed to generate reel outline: {e}")


# -----------------------------------------------------------------------------
# Image generation
# -----------------------------------------------------------------------------


def _download(url: str) -> bytes:
    """Download image from URL with error handling."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        raise ValueError(f"Failed to download image: {e}")

def generate_image(prompt: str,
                   size: str = "1024x1024",
                   debug: bool = False,
                   mock_dir: str | None = None,
                   idx: int | None = None) -> bytes:
    """Generate an image and return the raw bytes.

    We *don't* pass `response_format` because the gpt‑image‑1 endpoint
    already returns base‑64 by default (the extra field triggers a 400).
    For other models (e.g. DALL·E‑3) we fall back to the URL response.
    """
    
    if debug:
        # If user supplied images, pick the matching one
        if mock_dir:
            img_file = Path(mock_dir) / f"img_{idx+1}.jpg"
            if img_file.exists():
                return img_file.read_bytes()

        # Otherwise make a plain-colour placeholder
        from PIL import Image, ImageColor
        colour = ImageColor.getrgb(
            ["#3477eb", "#2ecc71", "#e74c3c", "#9b59b6"][idx % 4])
        img = Image.new("RGB", (1024, 1024), colour)
        buf = BytesIO(); img.save(buf, format="PNG"); return buf.getvalue()

    try:
        img = client.images.generate(
            model="gpt-image-1",  # swap to "dall-e-3" if preferred
            prompt=prompt,
            size=size,
            quality='low',
            n=1,
        )

        data = img.data[0]
        if hasattr(data, "b64_json") and data.b64_json:
            return base64.b64decode(data.b64_json)
        # Older or different model – fetch the image URL
        return _download(data.url)
        
    except Exception as e:
        print(f"Warning: Failed to generate image with gpt-image-1, trying dall-e-3: {e}")
        try:
            # Fallback to DALL-E-3
            img = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                quality='standard',
                n=1,
            )
            data = img.data[0]
            return _download(data.url)
        except Exception as e2:
            raise ValueError(f"Failed to generate image with both models: {e2}")

# -----------------------------------------------------------------------------
# Video assembly helpers
# -----------------------------------------------------------------------------

WIDTH, HEIGHT = 1080, 1920  # Instagram portrait
SCENE_DURATION = 6  # seconds per scene


def build_clip(image_bytes: bytes, caption: str, duration: float = SCENE_DURATION) -> CompositeVideoClip:
    """Create a CompositeVideoClip with the given image and caption."""
    try:
        # Load image and convert to numpy array
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        # Resize image to cover the entire frame (no black bars)
        img_width, img_height = img.size
        aspect_ratio = img_width / img_height
        target_aspect_ratio = WIDTH / HEIGHT
        
        if aspect_ratio > target_aspect_ratio:
            # Image is wider - scale by height and crop width
            new_height = HEIGHT
            new_width = int(HEIGHT * aspect_ratio)
        else:
            # Image is taller - scale by width and crop height
            new_width = WIDTH
            new_height = int(WIDTH / aspect_ratio)
        
        # Resize the image
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert PIL Image to numpy array for MoviePy
        img_array = np.array(img)
        
        # Create image clip from numpy array
        clip = ImageClip(img_array, duration=duration)
        
        # Always resize to fill the frame completely
        if new_width != WIDTH or new_height != HEIGHT:
            # Center crop to exact dimensions
            x_center = new_width // 2
            y_center = new_height // 2
            x1 = max(0, x_center - WIDTH // 2)
            y1 = max(0, y_center - HEIGHT // 2)
            
            clip = clip.with_effects([
                Crop(width=WIDTH, height=HEIGHT, x_center=x_center, y_center=y_center)
            ])
        
        # Ensure clip is exactly the right size
        clip = clip.with_effects([Resize(width=WIDTH, height=HEIGHT)])

        # Create text clip with better error handling
        try:
            text_clip = TextClip(
                text=caption,
                font=str(FONT_PATH) if FONT_PATH.exists() else None,
                font_size=60,
                color="white",
                stroke_color="black",
                stroke_width=2,
                method="caption",
                size=(int(WIDTH * 0.9), None),
                text_align="center",
                bg_color=None,
                transparent=True
            ).with_duration(duration).with_position(("center", HEIGHT - 250))
            
        except Exception as err:
            # fallback to Pillow's default font
            print(f"[TextClip] Falling back to default font: {err}")
            text_clip = TextClip(
                text=caption,
                font_size=60,
                color="white",
                stroke_color="black",
                stroke_width=2,
                method="caption",
                size=(int(WIDTH * 0.9), None),
                text_align="center"
            ).with_duration(duration).with_position(("center", HEIGHT - 250))
        
        return CompositeVideoClip([clip, text_clip], size=(WIDTH, HEIGHT))
        
    except Exception as e:
        raise ValueError(f"Failed to build video clip: {e}")

# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def make_reel(topic: str, music_path: str | None = None,
              out_dir: str = "reels",
              debug: bool = False,
              mock_dir: str | None = None) -> pathlib.Path:
    """Generate the complete reel."""
    print(f"Generating reel for topic: {topic}")
    
    # Generate the reel outline
    print("Creating reel outline...")
    scenes = outline_reel(topic, debug=debug, mock_dir=mock_dir)

    print(f"Generated {len(scenes)} scenes")

    # Create output directory
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    clips = []

    # Generate each scene
    for i, scene in enumerate(tqdm(scenes, desc="Generating scenes")):
        try:
            print(f"Generating scene {i+1}: {scene.get('caption', 'No caption')}")
            img_bytes = generate_image(scene["prompt"] ,  debug=debug,
                               mock_dir=mock_dir,
                               idx=i)
            clip = build_clip(img_bytes, scene["caption"])
            clips.append(clip)
        except Exception as e:
            print(f"Error generating scene {i+1}: {e}")
            print(f"Skipping scene: {scene}")
            continue

    if not clips:
        raise ValueError("No scenes were successfully generated")

    print("Concatenating video clips...")
    video = concatenate_videoclips(clips, method="compose")

    # Optional background music
    if music_path and pathlib.Path(music_path).exists():
        print(f"Adding background music: {music_path}")
        try:
            audio = AudioFileClip(music_path)
            # Loop audio if it's shorter than video, or trim if longer
            if audio.duration < video.duration:
                # Loop the audio to match video duration
                audio = audio.with_effects([]).loop(duration=video.duration)
            else:
                # Trim audio to match video duration
                audio = audio.subclipped(0, video.duration)
            
            # Apply fade out and volume effects
            audio = audio.with_effects([AudioFadeOut(2), MultiplyVolume(0.7)])
            video = video.with_audio(audio)
        except Exception as e:
            print(f"Warning: Failed to add background music: {e}")
            print("Continuing without background music...")

    # Generate output filename
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", topic.lower()).strip("_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = pathlib.Path(out_dir) / f"{slug}_{timestamp}.mp4"

    print(f"Rendering final video to: {output_path}")
    
    try:
        video.write_videofile(
            str(output_path),
            fps=30,
            codec="mpeg4",
            preset="medium",
            threads=os.cpu_count() or 4,
            audio_codec="aac",
            logger=None     # Suppress moviepy logging
        )
    except Exception as e:
        raise ValueError(f"Failed to render video: {e}")
    finally:
        # Clean up video clips to free memory
        video.close()
        for clip in clips:
            clip.close()

    return output_path


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an Instagram reel from a topic")
    parser.add_argument("topic")
    parser.add_argument("--music")
    parser.add_argument("--out-dir", default="reels")
    parser.add_argument("--debug", action="store_true",
                        help="Skip OpenAI calls, use local stubs")
    parser.add_argument("--mock-dir",
                        help="Folder that contains captions.txt + images")
    return parser.parse_args()



if __name__ == "__main__":
    args = _parse_args()
    try:
        video_path = make_reel(
            args.topic, args.music, args.out_dir,
            debug=args.debug, mock_dir=args.mock_dir)
        print(f"\n✅ Reel saved to {video_path}\n")
    except Exception as exc:
        print(f"✖ Error while generating reel:\n{exc}")
        import traceback
        traceback.print_exc()