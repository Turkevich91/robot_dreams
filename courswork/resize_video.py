"""
Resize any video to a standard resolution (default 1280x720).
- Mode "fit" keeps aspect ratio with letterboxing (black padding).
- Mode "stretch" resizes to exact size (may distort).

Usage (Windows cmd):
  python courswork/resize_video.py --input VIDEO/20250330_084658_L.MP4 --output VIDEO/20250330_084658_L_720p.mp4
  python courswork/resize_video.py --input <in.mp4> --size 1920x1080 --mode stretch --output <out.mp4>
"""
import argparse
import os
import sys
import cv2
import numpy as np
from typing import Tuple

# Default standard size: 1280x720 (HD 720p, 16:9) — widely supported, lightweight
DEFAULT_SIZE: Tuple[int, int] = (1280, 720)  # (width, height)


def parse_size(s: str) -> Tuple[int, int]:
    """Parse size string like '1280x720' into (width, height)."""
    try:
        w_str, h_str = s.lower().split("x")
        w, h = int(w_str), int(h_str)
        if w <= 0 or h <= 0:
            raise ValueError
        return (w, h)
    except Exception:
        raise argparse.ArgumentTypeError("Size must be in the form WxH, e.g., 1280x720")


def letterbox_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize frame to target_size preserving aspect ratio with black padding (letterbox)."""
    th = target_size[1]
    tw = target_size[0]
    fh, fw = frame.shape[:2]
    if fh == 0 or fw == 0:
        return np.zeros((th, tw, 3), dtype=np.uint8)
    scale = min(tw / fw, th / fh)
    nw, nh = int(round(fw * scale)), int(round(fh * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=resized.dtype)
    x0 = (tw - nw) // 2
    y0 = (th - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


def stretch_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize frame to target_size without preserving aspect ratio (may distort)."""
    th = target_size[1]
    tw = target_size[0]
    return cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)


def ensure_ext(path: str, ext: str) -> str:
    """Ensure output path has the given extension (e.g., '.mp4')."""
    root, old_ext = os.path.splitext(path)
    if old_ext.lower() != ext.lower():
        return root + ext
    return path


def resize_video(input_path: str, output_path: str, size: Tuple[int, int], mode: str = "fit") -> None:
    """
    Resize a video to the target size and save to output path.
    - mode: 'fit' (letterbox) or 'stretch'.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 1e-3:  # fallback if FPS is not available
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = ensure_ext(output_path, ".mp4")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (size[0], size[1]))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output writer: {out_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    transform = letterbox_frame if mode == "fit" else stretch_frame

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame is None:
            continue
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        resized = transform(frame, size)
        writer.write(resized)
        i += 1
        # Optional: print simple progress every ~5%%
        if frame_count > 0 and i % max(1, frame_count // 20) == 0:
            print(f"Progress: {int(100 * i / frame_count)}%")

    cap.release()
    writer.release()
    print(f"Saved: {out_path} ({size[0]}x{size[1]}), mode={mode}, fps={fps:.2f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Resize video to a standard resolution.")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", required=False, default=None, help="Path to output video (default adds _720p)")
    parser.add_argument("--size", type=parse_size, default=f"{DEFAULT_SIZE[0]}x{DEFAULT_SIZE[1]}", help="Target size as WxH (default 1280x720)")
    parser.add_argument("--mode", choices=["fit", "stretch"], default="fit", help="Resize mode: fit (letterbox) or stretch")

    args = parser.parse_args(argv)

    in_path = args.input
    if not os.path.isfile(in_path):
        print(f"Input not found: {in_path}")
        sys.exit(1)

    size = args.size if isinstance(args.size, tuple) else parse_size(args.size)

    if args.output:
        out_path = args.output
    else:
        root, ext = os.path.splitext(in_path)
        suffix = f"_{size[1]}p" if size[0] * 9 == size[1] * 16 else f"_{size[0]}x{size[1]}"
        out_path = root + suffix + ".mp4"

    try:
        resize_video(in_path, out_path, size=size, mode=args.mode)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()

