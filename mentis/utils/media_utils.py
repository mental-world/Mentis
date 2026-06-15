"""Media helpers for multimodal LLM inputs."""

from __future__ import annotations

import base64
import io
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def file_to_data_url(path: str | Path) -> str:
    p = Path(path)
    suffix = p.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix or "octet-stream"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{data}"


def scene_to_openai_content(
    scene: dict[str, Any], prompt: str, *, max_video_frames: int = 8
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for image_path in scene.get("image_paths", []) or []:
        content.append({"type": "input_image", "image_url": _path_to_data_url(image_path)})
    video_path = scene.get("video_path")
    if video_path:
        frames = scene.get("video_frames")
        errors: list[str] = []
        if not isinstance(frames, list) or not frames:
            frames, errors = video_frames_as_content(video_path, max_video_frames)
        if not frames:
            diagnostics = f" Decoder diagnostics: {'; '.join(errors)}" if errors else ""
            raise ValueError(
                f"Video input could not be decoded into frames: {video_path}. "
                "The OpenAI Responses payload uses sampled video frames as "
                "input_image items; install a local video decoder such as "
                "opencv-python, imageio with ffmpeg, or ffmpeg on PATH."
                f"{diagnostics}"
            )
        content.extend(frames)
    return content


def video_frames_as_content(path: str | Path, max_frames: int) -> tuple[list[dict[str, Any]], list[str]]:
    return _video_frames_as_content(path, max_frames)


def _path_to_data_url(path: str | Path) -> str:
    p = _resolve_path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Image input file not found: {path}")
    return file_to_data_url(p)


def _video_frames_as_content(path: str | Path, max_frames: int) -> tuple[list[dict[str, Any]], list[str]]:
    p = _resolve_path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Video input file not found: {path}")
    if max_frames <= 0:
        return [], ["max_video_frames must be greater than 0"]
    errors: list[str] = []
    frames = _animated_image_frames_as_content(p, max_frames, errors)
    if frames:
        return frames, errors
    frames = _opencv_video_frames_as_content(p, max_frames, errors)
    if frames:
        return frames, errors
    frames = _imageio_video_frames_as_content(p, max_frames, errors)
    if frames:
        return frames, errors
    return _ffmpeg_video_frames_as_content(p, max_frames, errors), errors


def _animated_image_frames_as_content(
    p: Path, max_frames: int, errors: list[str]
) -> list[dict[str, Any]]:
    try:
        from PIL import Image, ImageSequence

        image = Image.open(p)
        frames = []
        for idx, frame in enumerate(ImageSequence.Iterator(image)):
            if idx >= max_frames:
                break
            frames.append({"type": "input_image", "image_url": _pil_to_data_url(frame.convert("RGB"))})
        return frames
    except Exception as exc:
        errors.append(f"PIL/ImageSequence failed: {exc}")
        return []


def _opencv_video_frames_as_content(
    p: Path, max_frames: int, errors: list[str]
) -> list[dict[str, Any]]:
    try:
        import cv2  # type: ignore
        from PIL import Image

        cap = cv2.VideoCapture(str(p))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if not cap.isOpened() or frame_count <= 0:
            cap.release()
            errors.append("OpenCV failed: video could not be opened or has no frames")
            return []
        frames = []
        for idx in _uniform_indices(frame_count, max_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append({"type": "input_image", "image_url": _pil_to_data_url(Image.fromarray(rgb))})
        cap.release()
        return frames
    except Exception as exc:
        errors.append(f"OpenCV failed: {exc}")
        return []


def _imageio_video_frames_as_content(
    p: Path, max_frames: int, errors: list[str]
) -> list[dict[str, Any]]:
    try:
        import imageio.v2 as imageio  # type: ignore
        from PIL import Image

        reader = imageio.get_reader(str(p))
        frame_count = _imageio_frame_count(reader)
        indices = _uniform_indices(frame_count, max_frames) if frame_count else range(max_frames)
        frames = []
        for idx in indices:
            try:
                frame = reader.get_data(idx)
            except Exception:
                break
            frames.append({"type": "input_image", "image_url": _pil_to_data_url(Image.fromarray(frame))})
        reader.close()
        return frames
    except Exception as exc:
        errors.append(f"imageio failed: {exc}")
        return []


def _imageio_frame_count(reader: Any) -> int:
    try:
        count = reader.count_frames()
        return int(count) if count else 0
    except Exception:
        try:
            length = reader.get_length()
            return int(length) if length and length != float("inf") else 0
        except Exception:
            return 0


def _ffmpeg_video_frames_as_content(
    p: Path, max_frames: int, errors: list[str]
) -> list[dict[str, Any]]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        errors.append("ffmpeg failed: ffmpeg executable not found on PATH")
        return []
    frame_count = _ffmpeg_frame_count(p)
    with tempfile.TemporaryDirectory() as tmp:
        out_pattern = str(Path(tmp) / "frame_%03d.png")
        if frame_count:
            indices = _uniform_indices(frame_count, max_frames)
            select_expr = "+".join(f"eq(n\\,{idx})" for idx in indices)
            vf = f"select='{select_expr}',setpts=N/FRAME_RATE/TB"
        else:
            vf = _ffmpeg_uniform_fps_filter(p, max_frames)
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(p),
            "-vf",
            vf,
            "-vsync",
            "0",
            "-frames:v",
            str(max_frames),
            out_pattern,
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            errors.append(f"ffmpeg failed: {stderr or exc}")
            return []
        except Exception as exc:
            errors.append(f"ffmpeg failed: {exc}")
            return []
        frames = []
        for frame_path in sorted(Path(tmp).glob("frame_*.png"))[:max_frames]:
            frames.append({"type": "input_image", "image_url": file_to_data_url(frame_path)})
        return frames


def _ffmpeg_frame_count(p: Path) -> int:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0
    command = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(p),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return int(result.stdout.strip() or 0)
    except Exception:
        return 0


def _ffmpeg_uniform_fps_filter(p: Path, max_frames: int) -> str:
    duration = _ffmpeg_duration_seconds(p)
    if duration <= 0:
        return f"fps={max(1, max_frames)}"
    return f"fps={max_frames / duration:.8f}"


def _ffmpeg_duration_seconds(p: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(p),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return float(result.stdout.strip() or 0)
    except Exception:
        return 0.0


def _uniform_indices(frame_count: int, max_frames: int) -> list[int]:
    count = min(frame_count, max_frames)
    if count <= 1:
        return [0] if frame_count else []
    last = frame_count - 1
    return [round(i * last / (count - 1)) for i in range(count)]


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.exists() or p.is_absolute():
        return p
    return Path.cwd() / p


def _pil_to_data_url(image: Any) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"
