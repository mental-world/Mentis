from __future__ import annotations

import base64
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def scene_from_story(story: Any, base_dir: str = "") -> dict[str, Any]:
    data = story if isinstance(story, dict) else {}
    images = [str(p) for p in (data.get("images") or []) if p]
    video = str(data.get("video") or "")
    scene = {
        "text": str(data.get("text") or ""),
        "image_paths": [resolve_media_path(p, base_dir) for p in images],
        "video_path": resolve_media_path(video, base_dir) if video else "",
        "scene_context": str(data.get("scene_context") or ""),
    }
    if scene["image_paths"]:
        scene["modality"] = "image"
    elif scene["video_path"]:
        scene["modality"] = "video"
    else:
        scene["modality"] = "text"
    return scene


def resolve_media_path(path: str, base_dir: str = "") -> str:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return str(candidate)
    bases = []
    if base_dir:
        bases.extend([Path(base_dir), Path(base_dir).parent])
    bases.append(Path.cwd())
    for base in bases:
        resolved = base / candidate
        if resolved.exists():
            return str(resolved)
    return path


def image_content_part(path: str | Path) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": file_to_data_url(path)}}


def file_to_data_url(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Media file not found: {path}")
    suffix = source.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else (suffix or "png")
    encoded = base64.b64encode(source.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"


def require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for video inputs but was not found on PATH")
    return ffmpeg


def video_duration_seconds(path: str | Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    command = [
        ffprobe, "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1", str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return float(result.stdout.strip() or 0.0)
    except Exception:
        return 0.0


def sample_video_frames(path: str | Path, max_frames: int) -> tuple[list[dict[str, Any]], list[float]]:
    ffmpeg = require_ffmpeg()
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")
    duration = video_duration_seconds(source)
    count = max(1, int(max_frames))
    if duration > 0 and count > 1:
        timestamps = [round(i * max(duration - 0.15, 0.0) / (count - 1), 2) for i in range(count)]
    else:
        timestamps = [0.0]
    parts: list[dict[str, Any]] = []
    kept: list[float] = []
    with tempfile.TemporaryDirectory() as tmp:
        for index, ts in enumerate(timestamps):
            frame_path = Path(tmp) / f"frame_{index:03d}.png"
            command = [
                ffmpeg, "-hide_banner", "-loglevel", "error",
                "-ss", f"{ts:.3f}", "-i", str(source),
                "-frames:v", "1", str(frame_path),
            ]
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
            except Exception:
                continue
            if frame_path.is_file() and frame_path.stat().st_size > 0:
                parts.append(image_content_part(frame_path))
                kept.append(ts)
    if not parts:
        raise RuntimeError(f"No frames could be decoded from video: {path}")
    return parts, kept


def extract_audio_wav(path: str | Path) -> Path | None:
    ffmpeg = shutil.which("ffmpeg")
    source = Path(path)
    if not ffmpeg or not source.is_file():
        return None
    out_dir = Path(tempfile.mkdtemp(prefix="mentis_audio_"))
    target = out_dir / (source.stem + ".wav")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", str(target),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception:
        return None
    return target if target.is_file() and target.stat().st_size > 0 else None


def frame_preamble(timestamps: list[float], duration: float) -> str:
    stamps = ", ".join(f"{t:.1f}s" for t in timestamps)
    head = f"The following {len(timestamps)} images are frames sampled from one continuous video"
    if duration > 0:
        head += f" of {duration:.1f}s"
    return f"{head}, in temporal order at [{stamps}]."
