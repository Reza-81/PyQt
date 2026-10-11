# Advanced Task — PyQt Video Viewer

This folder contains the advanced video processing task from `Advance Task-PyQt-01.pdf`.

## Features

- Open common video files and show frame count, FPS, and resolution.
- Navigate by slider, previous/next buttons, or playback; the current frame number is shown beside the slider.
- Hover over the timeline slider to preview the corresponding frame in a small popup.
- Multiply the OpenCV blue channel by a user-selected coefficient.
- The coefficient accepts any finite non-negative floating-point value supported by Qt, with six decimal places and adaptive up/down steps.
- Apply coefficient changes instantly to the current frame and continue applying the latest coefficient while navigating or playing.
- Keep the Process button as an explicit manual re-apply control.
- Save the current processed frame as PNG or JPEG.
- User-facing errors for missing dependencies, unsupported videos, frame failures, dialog failures, and save failures.

## Run

From the repository root:

```powershell
.\venv\Scripts\python.exe -m pip install -r .\advance_task\requirements.txt
.\venv\Scripts\python.exe .\advance_task\main.py
```

The blue-channel coefficient is applied as `blue = clip(blue * coefficient, 0, 255)`.
