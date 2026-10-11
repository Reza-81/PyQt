# Advanced Task — PyQt Video Viewer

This folder contains the advanced video processing task from `Advance Task-PyQt-01.pdf`.

## Features

- Open common video files and show frame count, FPS, and resolution.
- Navigate by slider, frame number, previous/next buttons, or playback.
- Multiply the OpenCV blue channel by a user-selected coefficient.
- Process the current frame and continue applying the coefficient while navigating or playing.
- Save the current processed frame as PNG or JPEG.
- User-facing errors for missing dependencies, unsupported videos, frame failures, dialog failures, and save failures.

## Run

From the repository root:

```powershell
.\venv\Scripts\python.exe -m pip install -r .\advance_task\requirements.txt
.\venv\Scripts\python.exe .\advance_task\main.py
```

The blue-channel coefficient is applied as `blue = clip(blue * coefficient, 0, 255)`.

