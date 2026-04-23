#!/usr/bin/env python3

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from machinevisiontoolbox import Image, set_window_title

parser = argparse.ArgumentParser(
    description="Demonstration of morphological operations."
)
parser.add_argument(
    "-o", "--operation", choices=["min", "max", "erode", "dilate"], default="erode"
)
parser.add_argument(
    "-d",
    "--delay",
    type=float,
    default=0.5,
    help="Delay between updates in seconds (default: 0.5). Set to 0 for no delay.",
)
# add -m, --movie option to save the animation as a movie file
parser.add_argument(
    "-m",
    "--movie",
    type=str,
    default=None,
    help="Filename to save the animation as a movie file (e.g. morphdemo.mp4). Requires ffmpeg to be installed.",
)
args = parser.parse_args()


def _encode_movie(frame_dir: Path, movie_file: str, fps: float) -> None:
    """Encode saved PNG frames into a movie file using the ffmpeg executable."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg executable not found in PATH")

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        f"{fps:g}",
        "-i",
        str(frame_dir / "frame_%06d.png"),
        "-pix_fmt",
        "yuv420p",
        "-vcodec",
        "libx264",
        movie_file,
    ]
    subprocess.run(cmd, check=True)


morph_op = {
    "min": np.min,
    "max": np.max,
    "erode": np.min,
    "dilate": np.max,
}

if args.operation in ("min", "max"):
    img = Image.Random(10)
else:
    arr = np.zeros((16, 15), dtype="uint8")
    arr[3:6, 3:6] = 255  # 3x3 block in the top-left corner
    arr[3:6, 9:12] = 255  # 3x3 block in the top-right corner
    arr[4, 10] = 0  # hole in the second block
    arr[9:13, 3:7] = 255  # 4x4 block in the bottom-left corner
    arr[10:12, 10:12] = 255  # 2x2 block in the bottom-right corner
    arr[13, 11] = 255  # single pixel in the bottom-right corner
    img = Image(arr)

# we use a float image for the output so that we can show the "empty" pixels as NaN, which appear as white in the display
result = Image.Constant(np.nan, size=img.size, dtype="float")

# create two adjacent subplots, and display the input image on the left
fig, (input, output) = plt.subplots(1, 2, figsize=(10, 5))
set_window_title("Morphological operation demonstration")

img.showpixels(ax=input)  # display the input image and get a window annotator
window = img.showwindow(
    h=1, ax=input
)  # display the input image and get a window annotator

result.showpixels(
    ax=output, fmt="{:.0f}", badcolor="#FFFF00"
)  # pixels are displayed as integers with no decimal places, so the NaN pixels appear as blank

quit_requested = False


def on_key_press(event) -> None:
    global quit_requested
    if event.key == "q":
        quit_requested = True
        plt.close(fig)


fig.canvas.mpl_connect("key_press_event", on_key_press)

frame_dir: Path | None = None
frame_index = 0
if args.movie is not None:
    frame_dir = Path(tempfile.mkdtemp(prefix="morphdemo_frames_"))


def save_frame() -> None:
    global frame_index
    if frame_dir is not None:
        fig.savefig(frame_dir / f"frame_{frame_index:06d}.png", dpi=120)
        frame_index += 1


save_frame()

# loop over all pixels in the input image, excluding the border
for v in range(1, img.shape[0] - 1):
    if quit_requested or not plt.fignum_exists(fig.number):
        break
    for u in range(1, img.shape[1] - 1):
        if quit_requested or not plt.fignum_exists(fig.number):
            break
        W = window.move(u, v)  # highlight the window

        # window is the 3x3 window centered on (u,v)
        result.array[v, u] = morph_op[args.operation](
            W
        )  # perform the morphological operation

        # update the display of the output image
        output.cla()
        result.showpixels(ax=output, fmt="{:.0f}", badcolor="#FFFF00")

        save_frame()

        if args.movie is None and args.delay > 0:
            plt.pause(args.delay)

if args.movie is None and args.delay > 0:
    plt.show(block=True)

if frame_dir is not None:
    fps = 5.0 if args.delay <= 0 else 1.0 / args.delay
    _encode_movie(frame_dir, args.movie, fps=fps)
    shutil.rmtree(frame_dir)
    print(f"Movie --> {args.movie}")
