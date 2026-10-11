# Basic Task — PyQt Video Inspector

This folder contains a simple video player built with PyQt6, OpenCV, and NumPy.

## Features

- Open common video files (`mp4`, `avi`, `mov`, `mkv`, and `webm`).
- Play, pause, and stop with a `QTimer`-driven playback loop.
- Seek by frame number using the spin box or slider.
- Drag a rectangular region over the displayed frame.
- Show a live RGB histogram for the selected region in a separate window.
- User-facing dialogs for missing dependencies, unreadable videos, invalid frames, and invalid selections.

## Run

From the repository root:

```powershell
.\venv\Scripts\python.exe -m pip install -r .\basic_task\requirements.txt
.\venv\Scripts\python.exe .\basic_task\main.py
```

If you are using another virtual environment, replace the Python executable in those commands.

Frame numbers are zero-based: the first frame is `0` and the last frame is `total frames - 1`.
