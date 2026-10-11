"""Advanced video viewer with blue-channel coefficient processing.

PyQt6 is preferred. PySide6 is supported as a local fallback, which makes the
application easier to preview in environments that already provide PySide6.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

try:
    from PyQt6.QtCore import QPoint, QRectF, Qt, QTimer
    from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
    from PyQt6.QtWidgets import (
        QApplication,
        QAbstractSpinBox,
        QDialog,
        QFileDialog,
        QDoubleSpinBox,
        QFrame,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSizePolicy,
        QSlider,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError:  # pragma: no cover - used when PyQt6 is unavailable.
    from PySide6.QtCore import QPoint, QRectF, Qt, QTimer
    from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
    from PySide6.QtWidgets import (
        QApplication,
        QAbstractSpinBox,
        QDialog,
        QFileDialog,
        QDoubleSpinBox,
        QFrame,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSizePolicy,
        QSlider,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None


DEFAULT_BLUE_FACTOR = 1.5


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


class FrameViewer(QWidget):
    """Display a BGR OpenCV frame with aspect-ratio-preserving scaling."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._image = QImage()
        self._scaled_image = QImage()

    def set_frame(self, frame_bgr) -> None:
        if cv2 is None or np is None or frame_bgr is None:
            return
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._image = QImage(
            rgb.data, width, height, int(rgb.strides[0]), QImage.Format.Format_RGB888
        ).copy()
        self._scaled_image = QImage()
        self.update()

    def clear(self) -> None:
        self._image = QImage()
        self._scaled_image = QImage()
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0b1220"))
        if self._image.isNull():
            painter.setPen(QColor("#8394aa"))
            painter.setFont(QFont("Segoe UI", 13))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open a video to begin")
            return

        if self._scaled_image.isNull():
            self._scaled_image = self._image.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        left = (self.width() - self._scaled_image.width()) / 2
        top = (self.height() - self._scaled_image.height()) / 2
        painter.drawImage(QRectF(left, top, self._scaled_image.width(), self._scaled_image.height()), self._scaled_image)

    def resizeEvent(self, event) -> None:
        self._scaled_image = QImage()
        super().resizeEvent(event)


class VideoProcessor(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Video Viewer")
        self.resize(1120, 700)
        self.setMinimumSize(940, 590)

        self._capture = None
        self._video_path: Optional[str] = None
        self._frame_count = 0
        self._fps = 30.0
        self._frame_index = 0
        self._current_raw_frame = None
        self._current_processed_frame = None
        self._processing_enabled = False
        self._blue_factor = DEFAULT_BLUE_FACTOR
        self._updating_controls = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance_frame)

        self.viewer = FrameViewer()
        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        central = QWidget()
        outer = QHBoxLayout(central)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(14)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(255)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(16, 16, 16, 16)
        side.setSpacing(10)

        title = QLabel("Video Viewer")
        title.setObjectName("title")
        side.addWidget(title)

        self.open_button = QPushButton("Open video")
        self.open_button.setObjectName("primary")
        self.open_button.clicked.connect(self.open_video)
        side.addWidget(self.open_button)

        info_title = QLabel("VIDEO INFO")
        info_title.setObjectName("section")
        side.addWidget(info_title)
        self.info_label = QLabel("No video loaded")
        self.info_label.setObjectName("info")
        self.info_label.setWordWrap(True)
        side.addWidget(self.info_label)

        coefficient_title = QLabel("BLUE CHANNEL FACTOR")
        coefficient_title.setObjectName("section")
        side.addWidget(coefficient_title)

        factor_row = QHBoxLayout()
        factor_row.setContentsMargins(0, 0, 0, 0)
        factor_row.setSpacing(4)
        self.factor_spin = QDoubleSpinBox()
        # Keep the coefficient non-negative, but do not impose the previous
        # arbitrary maximum of 10. Values above 255 are valid input; they
        # simply saturate the uint8 blue channel at 255.
        self.factor_spin.setRange(0.0, sys.float_info.max)
        self.factor_spin.setDecimals(6)
        self.factor_spin.setSingleStep(0.1)
        self.factor_spin.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
        # Hide the native spin-box subcontrols. Some Windows Qt styles make
        # the upper native subcontrol behave like the line edit. Explicit
        # tool buttons below guarantee that both arrows receive clicks.
        self.factor_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.factor_spin.setAccelerated(True)
        self.factor_spin.setKeyboardTracking(True)
        self.factor_spin.setValue(DEFAULT_BLUE_FACTOR)
        self.factor_spin.setToolTip(
            "Enter any non-negative coefficient. Use the arrows to increment or decrement it."
        )
        self.factor_spin.valueChanged.connect(self._factor_changed)
        factor_row.addWidget(self.factor_spin, 1)

        factor_arrows = QVBoxLayout()
        factor_arrows.setContentsMargins(0, 0, 0, 0)
        factor_arrows.setSpacing(1)
        self.factor_up_button = QToolButton()
        self.factor_up_button.setObjectName("factorArrow")
        self.factor_up_button.setText("▲")
        self.factor_up_button.setToolTip("Increase coefficient")
        self.factor_up_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.factor_up_button.setFixedSize(34, 16)
        self.factor_up_button.clicked.connect(self.factor_spin.stepUp)
        self.factor_down_button = QToolButton()
        self.factor_down_button.setObjectName("factorArrow")
        self.factor_down_button.setText("▼")
        self.factor_down_button.setToolTip("Decrease coefficient")
        self.factor_down_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.factor_down_button.setFixedSize(34, 16)
        self.factor_down_button.clicked.connect(self.factor_spin.stepDown)
        factor_arrows.addWidget(self.factor_up_button)
        factor_arrows.addWidget(self.factor_down_button)
        factor_row.addLayout(factor_arrows)
        side.addLayout(factor_row)

        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self.process_current_frame)
        side.addWidget(self.process_button)

        self.save_button = QPushButton("Save current frame")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_current_frame)
        side.addWidget(self.save_button)
        side.addStretch(1)

        outer.addWidget(sidebar)

        content = QVBoxLayout()
        content.setSpacing(10)
        self.status_label = QLabel("Open a video file to start")
        self.status_label.setObjectName("status")
        content.addWidget(self.status_label)
        content.addWidget(self.viewer, 1)

        navigation = QHBoxLayout()
        self.previous_button = QPushButton("<<")
        self.previous_button.setToolTip("Previous frame")
        self.previous_button.clicked.connect(self.previous_frame)
        self.toggle_button = QPushButton("▶")
        self.toggle_button.setObjectName("playControl")
        self.toggle_button.setMinimumSize(52, 40)
        self.toggle_button.setMaximumWidth(52)
        self.toggle_button.setToolTip("Play / pause")
        self.toggle_button.clicked.connect(self.toggle_playback)
        self.next_button = QPushButton(">>")
        self.next_button.setToolTip("Next frame")
        self.next_button.clicked.connect(self.next_frame)
        navigation.addWidget(self.previous_button)
        navigation.addWidget(self.toggle_button)
        navigation.addWidget(self.next_button)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self._slider_frame_changed)
        navigation.addWidget(self.frame_slider, 1)

        self.frame_status = QLabel("0 / 0")
        self.frame_status.setMinimumWidth(70)
        self.frame_status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        navigation.addWidget(self.frame_status)
        content.addLayout(navigation)

        outer.addLayout(content, 1)
        self.setCentralWidget(central)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget { background: #172335; color: #e6edf6; font-family: 'Segoe UI'; font-size: 13px; }
            #sidebar { background: #1d2c40; border: 1px solid #30445c; border-radius: 8px; }
            #title { color: #ffffff; font-size: 21px; font-weight: 700; padding-bottom: 5px; }
            #section { color: #a9bdd5; font-size: 10px; font-weight: 700; letter-spacing: 1px; margin-top: 10px; }
            #info { background: #25374d; border: 1px solid #354b64; border-radius: 5px; padding: 10px; color: #d5e2ef; }
            #status { color: #a9bdd5; }
            QPushButton { background: #263a52; color: #e6edf6; border: 1px solid #405974; border-radius: 5px; padding: 8px 12px; }
            QPushButton:hover { background: #315172; border-color: #47a5ff; }
            QPushButton:disabled { color: #6e8095; background: #203044; border-color: #2d4055; }
            QPushButton#primary { background: #1685e5; border-color: #1685e5; color: #ffffff; }
            QPushButton#primary:hover { background: #2c9af0; }
            QPushButton#playControl { background: #1685e5; color: #ffffff; border: 2px solid #5abaff; border-radius: 20px; font-size: 18px; font-weight: 700; padding: 0px; }
            QPushButton#playControl:hover { background: #2c9af0; border-color: #9bd8ff; }
            QPushButton#playControl:pressed { background: #0d6fc2; border-color: #d1efff; }
            QDoubleSpinBox { background: #263a52; color: #ffffff; border: 1px solid #405974; border-radius: 4px; padding: 6px; }
            QToolButton#factorArrow { background: #263a52; color: #e6edf6; border: 1px solid #405974; border-radius: 2px; padding: 0px; font-size: 10px; }
            QToolButton#factorArrow:hover { background: #315172; color: #ffffff; border-color: #47a5ff; }
            QToolButton#factorArrow:pressed { background: #1685e5; border-color: #9bd8ff; }
            QSlider::groove:horizontal { height: 5px; background: #3a5068; border-radius: 2px; }
            QSlider::handle:horizontal { width: 14px; margin: -5px 0; background: #238ff0; border-radius: 7px; }
            """
        )

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def _show_warning(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)

    def _show_file_dialog(self, save: bool = False) -> Optional[str]:
        dialog = QFileDialog(self)
        dialog.setWindowTitle("Save modified frame" if save else "Open video")
        dialog.setFileMode(
            QFileDialog.FileMode.AnyFile if save else QFileDialog.FileMode.ExistingFile
        )
        dialog.setAcceptMode(
            QFileDialog.AcceptMode.AcceptSave if save else QFileDialog.AcceptMode.AcceptOpen
        )
        if save:
            dialog.setNameFilter("PNG image (*.png);;JPEG image (*.jpg *.jpeg);;All files (*.*)")
            dialog.setDefaultSuffix("png")
        else:
            dialog.setNameFilter(
                "Video files (*.mp4 *.avi *.mov *.mkv *.webm);;All files (*.*)"
            )
        dialog.setDirectory(str(Path.cwd()))
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        selected_files = dialog.selectedFiles()
        return selected_files[0] if selected_files else None

    def open_video(self) -> None:
        if cv2 is None or np is None:
            self._show_error("Missing dependency", "OpenCV and NumPy are required. Run: pip install -r requirements.txt")
            return
        try:
            path = self._show_file_dialog()
        except Exception as exc:
            self._show_error("File selection error", f"The video file dialog could not be opened.\n\n{exc}")
            return
        if path:
            self.load_video(path)

    def load_video(self, path: str) -> bool:
        if cv2 is None:
            self._show_error("Missing dependency", "OpenCV is not installed. Run: pip install -r requirements.txt")
            return False
        self.pause_video()
        try:
            capture = cv2.VideoCapture(path)
        except Exception as exc:
            self._show_error("Unable to open video", f"OpenCV could not open the selected file.\n\n{exc}")
            return False
        if not capture.isOpened():
            capture.release()
            self._show_error("Unable to open video", f"OpenCV could not open:\n{path}\n\nCheck the file and its codec.")
            return False

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_count <= 0 or width <= 0 or height <= 0:
            capture.release()
            self._show_error("Invalid video", "The selected file has no readable video frames.")
            return False

        if self._capture is not None:
            self._capture.release()
        self._capture = capture
        self._video_path = path
        self._frame_count = frame_count
        self._fps = fps if fps > 0 else 30.0
        self._frame_index = 0
        # Processing is active as soon as a video is loaded. The coefficient
        # control updates the displayed frame immediately, so Process is no
        # longer required after every coefficient change.
        self._processing_enabled = True
        self._blue_factor = self.factor_spin.value()
        self._current_processed_frame = None
        self.save_button.setEnabled(False)
        self.frame_slider.setRange(0, frame_count - 1)
        self.info_label.setText(
            f"Frames: {frame_count}\n"
            f"FPS: {self._fps:.2f}\n"
            f"Resolution: {width} × {height}\n\n"
            f"File: {Path(path).name}"
        )
        if not self._read_and_display(0):
            self._capture.release()
            self._capture = None
            return False
        self.status_label.setText("Ready — blue-channel updates are automatic")
        return True

    def _display_frame(self, index: int, frame) -> bool:
        self._frame_index = index
        self._current_raw_frame = frame
        if self._processing_enabled:
            self._current_processed_frame = self._apply_blue_factor(frame, self._blue_factor)
            display_frame = self._current_processed_frame
        else:
            self._current_processed_frame = None
            display_frame = frame
        self.viewer.set_frame(display_frame)
        self._sync_frame_controls(index)
        self.save_button.setEnabled(self._current_processed_frame is not None)
        return True

    def _read_and_display(self, index: int) -> bool:
        if self._capture is None:
            return False
        index = clamp(index, 0, max(0, self._frame_count - 1))
        try:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = self._capture.read()
        except Exception as exc:
            self.pause_video()
            self._show_error("Video read error", f"Could not read frame {index}.\n\n{exc}")
            return False
        if not ok or frame is None:
            self.pause_video()
            self._show_error("Video read error", f"Could not read frame {index}.")
            return False
        return self._display_frame(index, frame)

    def _sync_frame_controls(self, index: int) -> None:
        self._updating_controls = True
        try:
            self.frame_slider.setValue(index)
        finally:
            self._updating_controls = False
        self.frame_status.setText(f"{index} / {max(0, self._frame_count - 1)}")

    def _slider_frame_changed(self, value: int) -> None:
        if not self._updating_controls and self._capture is not None:
            self._read_and_display(value)

    def _apply_blue_factor(self, frame_bgr, factor: float):
        if np is None:
            return frame_bgr
        processed = frame_bgr.copy()
        blue_channel = processed[:, :, 0].astype(np.float32) * float(factor)
        processed[:, :, 0] = np.clip(blue_channel, 0, 255).astype(np.uint8)
        return processed

    def process_current_frame(self) -> None:
        if self._capture is None or self._current_raw_frame is None:
            self._show_warning("No video loaded", "Open a video before processing a frame.")
            return
        self._blue_factor = self.factor_spin.value()
        self._processing_enabled = True
        self._display_frame(self._frame_index, self._current_raw_frame)
        self.status_label.setText(f"Blue channel processed with coefficient {self._blue_factor:.2f}")

    def _factor_changed(self, value: float) -> None:
        """Apply a changed coefficient immediately to the current frame."""
        self._blue_factor = float(value)
        if self._capture is None or self._current_raw_frame is None:
            return
        self._processing_enabled = True
        self._display_frame(self._frame_index, self._current_raw_frame)
        self.status_label.setText(f"Updated blue channel coefficient to {self._blue_factor:.2f}")

    def previous_frame(self) -> None:
        if self._capture is None:
            self._show_warning("No video loaded", "Open a video before navigating frames.")
            return
        self.pause_video()
        self._read_and_display(self._frame_index - 1)

    def next_frame(self) -> None:
        if self._capture is None:
            self._show_warning("No video loaded", "Open a video before navigating frames.")
            return
        self.pause_video()
        self._read_and_display(self._frame_index + 1)

    def toggle_playback(self) -> None:
        if self.timer.isActive():
            self.pause_video()
        else:
            self.play_video()

    def play_video(self) -> None:
        if self._capture is None:
            self._show_warning("No video loaded", "Open a video before pressing Play.")
            return
        if self._frame_index >= self._frame_count - 1:
            self._read_and_display(0)
        self.timer.start(max(1, round(1000.0 / self._fps)))
        self.toggle_button.setText("Ⅱ")
        self.toggle_button.setToolTip("Pause")
        self.status_label.setText("Playing")

    def pause_video(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.status_label.setText("Paused")
        self.toggle_button.setText("▶")
        self.toggle_button.setToolTip("Play")

    def _advance_frame(self) -> None:
        next_index = self._frame_index + 1
        if next_index >= self._frame_count:
            self.pause_video()
            self.status_label.setText("Playback finished")
            return
        try:
            ok, frame = self._capture.read()
        except Exception as exc:
            self.pause_video()
            self._show_error("Video read error", f"Could not read frame {next_index}.\n\n{exc}")
            return
        if not ok or frame is None:
            self.pause_video()
            self._show_error("Video read error", f"Could not read frame {next_index}.")
            return
        self._display_frame(next_index, frame)

    def save_current_frame(self) -> None:
        if self._current_processed_frame is None:
            self._show_warning("No processed frame", "Open a video before saving the current frame.")
            return
        try:
            path = self._show_file_dialog(save=True)
        except Exception as exc:
            self._show_error("File selection error", f"The save dialog could not be opened.\n\n{exc}")
            return
        if not path:
            return
        output_path = Path(path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".png")
        try:
            saved = cv2.imwrite(str(output_path), self._current_processed_frame)
        except Exception as exc:
            self._show_error("Save error", f"Could not save the modified frame.\n\n{exc}")
            return
        if not saved:
            self._show_error("Save error", f"OpenCV could not write:\n{output_path}")
            return
        self.status_label.setText(f"Saved modified frame to {output_path.name}")

    def closeEvent(self, event) -> None:
        self.timer.stop()
        if self._capture is not None:
            self._capture.release()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    if cv2 is None or np is None:
        missing = []
        if cv2 is None:
            missing.append("opencv-python")
        if np is None:
            missing.append("numpy")
        QMessageBox.critical(
            None,
            "Missing dependencies",
            "This application needs:\n\n"
            + "\n".join(f"- {package}" for package in missing)
            + "\n\nInstall them with:\npython -m pip install -r requirements.txt",
        )
        return 1
    window = VideoProcessor()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
