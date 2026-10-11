"""A small video inspector built with Qt, OpenCV, and NumPy.

PyQt6 is the primary supported binding.  PySide6 is used as a development
fallback so the application can also be previewed in environments where the
Qt binding has been provided under the LGPL PySide license.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

try:  # The requested binding.
    from PyQt6.QtCore import QLineF, QPoint, QRectF, Qt, QTimer
    from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap
    from PyQt6.QtWidgets import (
        QApplication,
        QAbstractSpinBox,
        QDialog,
        QFileDialog,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSizePolicy,
        QSlider,
        QSpinBox,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )
    from PyQt6.QtCore import pyqtSignal as Signal
except ImportError:  # pragma: no cover - exercised only when PyQt6 is absent.
    from PySide6.QtCore import QLineF, QPoint, QRectF, Qt, QTimer, Signal
    from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QAbstractSpinBox,
        QDialog,
        QFileDialog,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSizePolicy,
        QSlider,
        QSpinBox,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )

try:
    import cv2
except ImportError:  # Give the user a helpful dialog at startup instead of a traceback.
    cv2 = None

try:
    import numpy as np
except ImportError:  # Give the user a helpful dialog at startup instead of a traceback.
    np = None


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


DEFAULT_HISTOGRAM_PERCENTAGE = 25


class PreviewSlider(QSlider):
    """Slider that reports the frame under the mouse without changing value."""

    hover_value_changed = Signal(int)
    hover_left = Signal()

    def __init__(self, orientation: Qt.Orientation, parent: Optional[QWidget] = None) -> None:
        super().__init__(orientation, parent)
        self.setMouseTracking(True)
        self._last_hover_value: Optional[int] = None

    def value_at_x(self, x: int) -> int:
        if self.maximum() <= self.minimum() or self.width() <= 1:
            return self.minimum()
        x = clamp(x, 0, self.width() - 1)
        ratio = x / max(1, self.width() - 1)
        return round(self.minimum() + ratio * (self.maximum() - self.minimum()))

    def mouseMoveEvent(self, event) -> None:
        super().mouseMoveEvent(event)
        value = self.value_at_x(round(event.position().x()))
        if value != self._last_hover_value:
            self._last_hover_value = value
            self.hover_value_changed.emit(value)

    def leaveEvent(self, event) -> None:
        self._last_hover_value = None
        self.hover_left.emit()
        super().leaveEvent(event)


class FramePreviewPopup(QFrame):
    """Small non-interactive thumbnail shown while hovering the frame slider."""

    def __init__(self) -> None:
        super().__init__(None, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFixedSize(204, 140)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(7, 7, 7, 6)
        layout.setSpacing(3)
        self.image_label = QLabel()
        self.image_label.setFixedSize(190, 106)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)
        self.frame_label = QLabel("Frame -")
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.frame_label)
        self.setStyleSheet(
            "QFrame { background: #101b2a; border: 1px solid #4d7398; border-radius: 6px; }"
            "QLabel { background: transparent; color: #e6edf6; border: none; }"
        )

    def set_frame(self, frame_bgr, frame_index: int) -> None:
        if cv2 is None or np is None or frame_bgr is None:
            return
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = QImage(
            rgb.data, width, height, int(rgb.strides[0]), QImage.Format.Format_RGB888
        ).copy()
        image = image.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(QPixmap.fromImage(image))
        self.frame_label.setText(f"Frame {frame_index}")


class VideoCanvas(QWidget):
    """Paint a video frame and let the user drag a rectangular ROI over it."""

    roi_selected = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self._image = QImage()
        self._scaled_image = QImage()
        self._source_width = 0
        self._source_height = 0
        self._roi: Optional[tuple[int, int, int, int]] = None
        self._drag_start: Optional[QPoint] = None
        self._drag_end: Optional[QPoint] = None
        self._display_rect = QRectF()

    def set_frame(self, frame_bgr) -> None:
        if frame_bgr is None or np is None:
            return
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Copy is intentional: the QImage must own its pixel memory after this call.
        self._image = QImage(
            rgb.data, width, height, int(rgb.strides[0]), QImage.Format.Format_RGB888
        ).copy()
        self._scaled_image = QImage()
        self._source_width = width
        self._source_height = height
        self.update()

    def clear_roi(self) -> None:
        self._roi = None
        self._drag_start = None
        self._drag_end = None
        self.roi_selected.emit(None)
        self.update()

    def roi(self) -> Optional[tuple[int, int, int, int]]:
        return self._roi

    def _clamp_to_image(self, point: QPoint) -> QPoint:
        x = clamp(point.x(), int(self._display_rect.left()), int(self._display_rect.right()))
        y = clamp(point.y(), int(self._display_rect.top()), int(self._display_rect.bottom()))
        return QPoint(x, y)

    def _display_to_source(self, point: QPoint) -> tuple[int, int]:
        if not self._display_rect.isValid() or not self._source_width or not self._source_height:
            return 0, 0
        x_ratio = self._source_width / self._display_rect.width()
        y_ratio = self._source_height / self._display_rect.height()
        x = clamp(round((point.x() - self._display_rect.left()) * x_ratio), 0, self._source_width - 1)
        y = clamp(round((point.y() - self._display_rect.top()) * y_ratio), 0, self._source_height - 1)
        return x, y

    def _make_source_roi(self, start: QPoint, end: QPoint) -> Optional[tuple[int, int, int, int]]:
        x1, y1 = self._display_to_source(start)
        x2, y2 = self._display_to_source(end)
        x, right = sorted((x1, x2))
        y, bottom = sorted((y1, y2))
        width = right - x + 1
        height = bottom - y + 1
        if width < 2 or height < 2:
            return None
        return x, y, width, height

    def _source_to_display(self, roi: tuple[int, int, int, int]) -> QRectF:
        x, y, width, height = roi
        scale_x = self._display_rect.width() / self._source_width
        scale_y = self._display_rect.height() / self._source_height
        return QRectF(
            self._display_rect.left() + x * scale_x,
            self._display_rect.top() + y * scale_y,
            width * scale_x,
            height * scale_y,
        )

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0f1724"))

        if self._image.isNull():
            painter.setPen(QColor("#91a4b8"))
            painter.setFont(QFont("Segoe UI", 13))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open a video to begin")
            return

        if self._scaled_image.isNull():
            self._scaled_image = self._image.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        scaled = self._scaled_image
        left = (self.width() - scaled.width()) / 2
        top = (self.height() - scaled.height()) / 2
        self._display_rect = QRectF(left, top, scaled.width(), scaled.height())
        painter.drawImage(self._display_rect, scaled)

        roi = self._roi
        if self._drag_start is not None and self._drag_end is not None:
            roi_rect = QRectF(
                float(self._drag_start.x()),
                float(self._drag_start.y()),
                float(self._drag_end.x() - self._drag_start.x()),
                float(self._drag_end.y() - self._drag_start.y()),
            ).normalized()
        elif roi is not None:
            roi_rect = self._source_to_display(roi)
        else:
            roi_rect = QRectF()

        if roi_rect.isValid() and not roi_rect.isEmpty():
            painter.setPen(QPen(QColor("#22c55e"), 2, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(34, 197, 94, 35))
            painter.drawRect(roi_rect)

    def resizeEvent(self, event) -> None:
        self._scaled_image = QImage()
        super().resizeEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._display_rect.contains(event.position()):
            self._drag_start = self._clamp_to_image(event.position().toPoint())
            self._drag_end = self._drag_start
            self.update()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_start is not None:
            self._drag_end = self._clamp_to_image(event.position().toPoint())
            self.update()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._drag_start is not None:
            self._drag_end = self._clamp_to_image(event.position().toPoint())
            self._roi = self._make_source_roi(self._drag_start, self._drag_end)
            self._drag_start = None
            self._drag_end = None
            self.roi_selected.emit(self._roi)
            self.update()
        else:
            super().mouseReleaseEvent(event)


class HistogramWidget(QWidget):
    """A dependency-free RGB histogram drawn from NumPy arrays."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(560, 350)
        self._histograms: Optional[dict[str, object]] = None
        self._sample_size = 0
        self._total_size = 0
        self._requested_percentage = DEFAULT_HISTOGRAM_PERCENTAGE

    def update_histogram(
        self, frame_bgr, roi: tuple[int, int, int, int], percentage: int
    ) -> None:
        if cv2 is None or np is None or frame_bgr is None or roi is None:
            return
        x, y, width, height = roi
        frame_height, frame_width = frame_bgr.shape[:2]
        x = clamp(x, 0, frame_width - 1)
        y = clamp(y, 0, frame_height - 1)
        right = clamp(x + width, x + 1, frame_width)
        bottom = clamp(y + height, y + 1, frame_height)
        sample = frame_bgr[y:bottom, x:right]
        if sample.size == 0:
            return

        self._total_size = int(sample.shape[0] * sample.shape[1])
        self._requested_percentage = max(1, min(int(percentage), 100))

        if self._requested_percentage < 100:
            # Select evenly distributed rows and columns. This makes the
            # percentage a spatially representative sample rather than just
            # taking the first pixels in the ROI.
            fraction = self._requested_percentage / 100.0
            row_count = max(1, round(sample.shape[0] * np.sqrt(fraction)))
            column_count = max(1, round(sample.shape[1] * np.sqrt(fraction)))
            rows = np.linspace(0, sample.shape[0] - 1, row_count, dtype=np.intp)
            columns = np.linspace(0, sample.shape[1] - 1, column_count, dtype=np.intp)
            sample = sample[rows][:, columns]

        # bincount is faster than calling np.histogram three times for uint8
        # image channels and always produces the required 256 bins.
        self._histograms = {
            "Red": np.bincount(sample[:, :, 2].reshape(-1), minlength=256),
            "Green": np.bincount(sample[:, :, 1].reshape(-1), minlength=256),
            "Blue": np.bincount(sample[:, :, 0].reshape(-1), minlength=256),
        }
        self._sample_size = int(sample.shape[0] * sample.shape[1])
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#111c2b"))
        plot = QRectF(58, 54, max(1, self.width() - 82), max(1, self.height() - 94))

        painter.setPen(QPen(QColor("#30445b"), 1))
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = plot.bottom() - fraction * plot.height()
            painter.drawLine(QLineF(plot.left(), y, plot.right(), y))
        for value in (0, 64, 128, 192, 255):
            x = plot.left() + value / 255 * plot.width()
            painter.drawLine(QLineF(x, plot.top(), x, plot.bottom()))

        painter.setPen(QColor("#a9b8c9"))
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(QRectF(0, plot.bottom() - 10, 50, 20), Qt.AlignmentFlag.AlignRight, "0")
        painter.drawText(QRectF(0, plot.top() - 10, 50, 20), Qt.AlignmentFlag.AlignRight, "max")
        for value in (0, 64, 128, 192, 255):
            x = plot.left() + value / 255 * plot.width()
            painter.drawText(QRectF(x - 15, plot.bottom() + 8, 30, 18), Qt.AlignmentFlag.AlignCenter, str(value))

        legend = (("Red", "#ef4444"), ("Green", "#22c55e"), ("Blue", "#3b82f6"))
        legend_x = plot.left()
        for name, color in legend:
            painter.setBrush(QColor(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(QRectF(legend_x, 18, 12, 12), 2, 2)
            painter.setPen(QColor("#d7e0ea"))
            painter.drawText(QRectF(legend_x + 18, 12, 55, 24), Qt.AlignmentFlag.AlignVCenter, name)
            legend_x += 92
        pixel_label = f"{self._requested_percentage}% ({self._sample_size:,} pixels)"
        if self._sample_size < self._total_size:
            pixel_label += f" / {self._total_size:,}"
        painter.drawText(QRectF(self.width() - 270, 12, 260, 24), Qt.AlignmentFlag.AlignRight, pixel_label)

        if not self._histograms:
            painter.setPen(QColor("#91a4b8"))
            painter.drawText(plot, Qt.AlignmentFlag.AlignCenter, "Select a region to calculate its histogram")
            return

        maximum = max(max(int(values.max()), 1) for values in self._histograms.values())
        for name, color in legend:
            values = self._histograms[name]
            path = QPainterPath()
            for index, count in enumerate(values):
                px = plot.left() + index / 255 * plot.width()
                py = plot.bottom() - (float(count) / maximum) * plot.height()
                if index == 0:
                    path.moveTo(px, py)
                else:
                    path.lineTo(px, py)
            painter.setPen(QPen(QColor(color), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)


class HistogramWindow(QMainWindow):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Histogram — RGB channels")
        self.resize(620, 410)
        self.setCentralWidget(HistogramWidget(self))

    def set_data(self, frame_bgr, roi: tuple[int, int, int, int], percentage: int) -> None:
        widget = self.centralWidget()
        if isinstance(widget, HistogramWidget):
            widget.update_histogram(frame_bgr, roi, percentage)


class VideoPlayer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Video Inspector")
        self.resize(1180, 720)
        self.setMinimumSize(960, 600)

        self._capture = None
        self._video_path: Optional[str] = None
        self._current_frame = None
        self._frame_index = 0
        self._frame_count = 0
        self._fps = 30.0
        self._updating_controls = False
        self._histogram_window: Optional[HistogramWindow] = None
        self._preview_capture = None
        self._pending_preview_index: Optional[int] = None
        self._preview_popup = FramePreviewPopup()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance_frame)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(35)
        self._preview_timer.timeout.connect(self._update_frame_preview)

        self.canvas = VideoCanvas()
        self.canvas.roi_selected.connect(self._on_roi_selected)
        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        central = QWidget()
        outer = QHBoxLayout(central)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(16)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(265)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(16, 16, 16, 16)
        side_layout.setSpacing(10)

        title = QLabel("Video Inspector")
        title.setObjectName("title")
        side_layout.addWidget(title)

        self.open_button = QPushButton("📂  Open video")
        self.open_button.clicked.connect(self.open_video)
        side_layout.addWidget(self.open_button)

        info_title = QLabel("VIDEO INFO")
        info_title.setObjectName("section")
        side_layout.addWidget(info_title)
        self.info_label = QLabel("No video loaded")
        self.info_label.setWordWrap(True)
        self.info_label.setObjectName("info")
        side_layout.addWidget(self.info_label)

        self.frame_slider = PreviewSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self._slider_frame_changed)
        self.frame_slider.hover_value_changed.connect(self._schedule_frame_preview)
        self.frame_slider.hover_left.connect(self._hide_frame_preview)
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, 0)
        self.frame_spin.setFixedWidth(125)
        self.frame_spin.setToolTip("Enter a frame number to display")
        self.frame_spin.valueChanged.connect(self._spin_frame_changed)

        roi_title = QLabel("REGION SELECTION")
        roi_title.setObjectName("section")
        side_layout.addWidget(roi_title)
        roi_help = QLabel("Drag with the left mouse button over the video to select a region.")
        roi_help.setWordWrap(True)
        roi_help.setObjectName("muted")
        side_layout.addWidget(roi_help)

        histogram_percent_title = QLabel("HISTOGRAM PIXELS")
        histogram_percent_title.setObjectName("section")
        side_layout.addWidget(histogram_percent_title)
        percentage_row = QHBoxLayout()
        percentage_row.addWidget(QLabel("Pixels used"))
        self.histogram_percent_spin = QSpinBox()
        self.histogram_percent_spin.setRange(1, 100)
        self.histogram_percent_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.histogram_percent_spin.setAccelerated(True)
        self.histogram_percent_spin.setKeyboardTracking(True)
        self.histogram_percent_spin.setValue(DEFAULT_HISTOGRAM_PERCENTAGE)
        self.histogram_percent_spin.setSuffix(" %")
        self.histogram_percent_spin.setToolTip(
            "Percentage of pixels from the selected region used for each histogram"
        )
        self.histogram_percent_spin.valueChanged.connect(self._histogram_percentage_changed)
        percentage_row.addWidget(self.histogram_percent_spin)

        percentage_arrows = QVBoxLayout()
        percentage_arrows.setContentsMargins(0, 0, 0, 0)
        percentage_arrows.setSpacing(1)
        self.percentage_up_button = QToolButton()
        self.percentage_up_button.setObjectName("percentageArrow")
        self.percentage_up_button.setText("\u25B2")
        self.percentage_up_button.setToolTip("Increase histogram pixel percentage")
        self.percentage_up_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.percentage_up_button.setFixedSize(34, 16)
        self.percentage_up_button.clicked.connect(self.histogram_percent_spin.stepUp)
        self.percentage_down_button = QToolButton()
        self.percentage_down_button.setObjectName("percentageArrow")
        self.percentage_down_button.setText("\u25BC")
        self.percentage_down_button.setToolTip("Decrease histogram pixel percentage")
        self.percentage_down_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.percentage_down_button.setFixedSize(34, 16)
        self.percentage_down_button.clicked.connect(self.histogram_percent_spin.stepDown)
        percentage_arrows.addWidget(self.percentage_up_button)
        percentage_arrows.addWidget(self.percentage_down_button)
        percentage_row.addLayout(percentage_arrows)
        side_layout.addLayout(percentage_row)
        percentage_help = QLabel("100% uses every pixel. The histogram updates on every frame.")
        percentage_help.setWordWrap(True)
        percentage_help.setObjectName("muted")
        side_layout.addWidget(percentage_help)

        self.histogram_button = QPushButton("▥  Show histogram")
        self.histogram_button.clicked.connect(self.show_histogram)
        side_layout.addWidget(self.histogram_button)
        self.clear_roi_button = QPushButton("Clear selected region")
        self.clear_roi_button.clicked.connect(self.canvas.clear_roi)
        side_layout.addWidget(self.clear_roi_button)
        side_layout.addStretch(1)

        outer.addWidget(sidebar)

        content = QVBoxLayout()
        content.setSpacing(12)
        self.status_label = QLabel("Open a video file to start")
        self.status_label.setObjectName("status")
        content.addWidget(self.status_label)
        content.addWidget(self.canvas, 1)

        controls = QHBoxLayout()
        self.previous_button = QPushButton("<<")
        self.previous_button.setToolTip("Previous frame")
        self.previous_button.clicked.connect(self.previous_frame)
        self.toggle_button = QPushButton("▶  Play")
        self.toggle_button.setObjectName("primary")
        self.toggle_button.clicked.connect(self.toggle_playback)
        self.next_button = QPushButton(">>")
        self.next_button.setToolTip("Next frame")
        self.next_button.clicked.connect(self.next_frame)
        self.stop_button = QPushButton("■  Stop")
        self.stop_button.clicked.connect(self.stop_video)
        controls.addWidget(self.previous_button)
        controls.addWidget(self.toggle_button)
        controls.addWidget(self.next_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(QLabel("Frame:"))
        controls.addWidget(self.frame_spin)
        controls.addWidget(self.frame_slider, 1)
        self.frame_status = QLabel("Frame: —")
        self.frame_status.setObjectName("frame_status")
        self.frame_status.setMinimumWidth(90)
        self.frame_status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.frame_status)
        content.addLayout(controls)
        outer.addLayout(content, 1)

        self.setCentralWidget(central)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget { background: #f5f7fa; color: #203047; font-family: 'Segoe UI'; font-size: 13px; }
            #sidebar { background: #ffffff; border: 1px solid #dce4ed; border-radius: 10px; }
            #title { color: #16263b; font-size: 21px; font-weight: 700; padding-bottom: 5px; }
            #section { color: #77869a; font-size: 10px; font-weight: 700; letter-spacing: 1px; margin-top: 10px; }
            #info { background: #f2f5f8; border-radius: 6px; padding: 10px; color: #4c6077; }
            #muted { color: #718198; line-height: 1.3; }
            #status { color: #58708a; }
            #frame_status { color: #58708a; font-weight: 600; }
            QPushButton { background: #ffffff; border: 1px solid #cbd6e2; border-radius: 6px; padding: 9px 12px; }
            QPushButton:hover { background: #edf5ff; border-color: #2584e8; }
            QPushButton#primary { background: #1685e5; color: #ffffff; border-color: #1685e5; }
             QPushButton#primary:hover { background: #0e70c3; }
             QSpinBox { background: #ffffff; border: 1px solid #cbd6e2; border-radius: 5px; padding: 5px; }
             QToolButton#percentageArrow { background: #ffffff; color: #203047; border: 1px solid #cbd6e2; border-radius: 2px; padding: 0px; font-size: 10px; }
             QToolButton#percentageArrow:hover { background: #edf5ff; color: #1685e5; border-color: #2584e8; }
             QToolButton#percentageArrow:pressed { background: #1685e5; color: #ffffff; border-color: #0e70c3; }
             QSlider::groove:horizontal { height: 5px; background: #dbe4ee; border-radius: 2px; }
            QSlider::handle:horizontal { width: 15px; margin: -5px 0; background: #1685e5; border-radius: 7px; }
            """
        )

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def _show_warning(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)

    def open_video(self) -> None:
        if cv2 is None or np is None:
            self._show_error("Missing dependency", "OpenCV and NumPy are required. Run: pip install -r requirements.txt")
            return
        try:
            # Use an explicit Qt dialog instead of the static native-dialog
            # helper. This is more reliable across Windows Qt installations
            # and avoids crashes in some native file-dialog backends.
            dialog = QFileDialog(self)
            dialog.setWindowTitle("Open video")
            dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
            dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
            dialog.setNameFilter(
                "Video files (*.mp4 *.avi *.mov *.mkv *.webm);;All files (*.*)"
            )
            dialog.setDirectory(str(Path.cwd()))
            dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)

            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            selected_files = dialog.selectedFiles()
            if selected_files:
                self.load_video(selected_files[0])
        except Exception as exc:
            self._show_error(
                "File selection error",
                f"The video file dialog could not be opened.\n\n{exc}",
            )

    def load_video(self, path: str) -> bool:
        if cv2 is None:
            self._show_error("Missing dependency", "OpenCV is not installed. Run: pip install -r requirements.txt")
            return False
        self.pause_video()
        new_capture = cv2.VideoCapture(path)
        if not new_capture.isOpened():
            new_capture.release()
            self._show_error("Unable to open video", f"OpenCV could not open:\n{path}\n\nCheck that the file exists and uses a supported codec.")
            return False

        preview_capture = None
        try:
            preview_capture = cv2.VideoCapture(path)
            if not preview_capture.isOpened():
                preview_capture.release()
                preview_capture = None
        except Exception:
            preview_capture = None

        frame_count = int(new_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(new_capture.get(cv2.CAP_PROP_FPS))
        width = int(new_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(new_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_count <= 0 or width <= 0 or height <= 0:
            new_capture.release()
            if preview_capture is not None:
                preview_capture.release()
            self._show_error("Invalid video", "The selected file does not contain readable video frames.")
            return False

        if self._capture is not None:
            self._capture.release()
        if self._preview_capture is not None:
            self._preview_capture.release()
        self._capture = new_capture
        self._preview_capture = preview_capture
        self._hide_frame_preview()
        self._video_path = path
        self._frame_count = frame_count
        self._fps = fps if fps > 0.0 else 30.0
        self.frame_slider.setRange(0, frame_count - 1)
        self.frame_spin.setRange(0, frame_count - 1)
        self.info_label.setText(
            f"Resolution: {width} × {height}\n"
            f"FPS: {self._fps:.2f}\n"
            f"Total frames: {frame_count}\n\n"
            f"File: {Path(path).name}"
        )
        self.canvas.clear_roi()
        self._read_and_display(0)
        self.status_label.setText("Ready — drag on the video to select an ROI")
        return True

    def _read_and_display(self, index: int) -> bool:
        if self._capture is None:
            return False
        index = clamp(index, 0, max(0, self._frame_count - 1))
        try:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = self._capture.read()
        except Exception as exc:  # OpenCV backend errors should become user-facing errors.
            self.pause_video()
            self._show_error("Video read error", f"Could not read frame {index}.\n\n{exc}")
            return False
        if not ok or frame is None:
            self.pause_video()
            self._show_error("Video read error", f"Could not read frame {index} from the selected video.")
            return False

        return self._display_decoded_frame(index, frame)

    def _display_decoded_frame(self, index: int, frame) -> bool:
        """Update the UI for an already decoded frame without seeking the video."""
        self._frame_index = index
        self._current_frame = frame
        self.canvas.set_frame(frame)
        self._sync_frame_controls(index)
        self._update_histogram()
        return True

    def _sync_frame_controls(self, index: int) -> None:
        self._updating_controls = True
        try:
            self.frame_spin.setValue(index)
            self.frame_slider.setValue(index)
        finally:
            self._updating_controls = False
        total = max(0, self._frame_count - 1)
        self.frame_status.setText(f"Frame: {index} / {total}")

    def _spin_frame_changed(self, value: int) -> None:
        if not self._updating_controls and self._capture is not None:
            self._read_and_display(value)

    def _slider_frame_changed(self, value: int) -> None:
        if not self._updating_controls and self._capture is not None:
            self._read_and_display(value)

    def _schedule_frame_preview(self, index: int) -> None:
        if self._preview_capture is None or self._frame_count <= 0:
            return
        self._pending_preview_index = clamp(index, 0, self._frame_count - 1)
        self._preview_timer.start()

    def _hide_frame_preview(self) -> None:
        self._preview_timer.stop()
        self._pending_preview_index = None
        self._preview_popup.hide()

    def _update_frame_preview(self) -> None:
        index = self._pending_preview_index
        if index is None or self._preview_capture is None:
            return
        try:
            self._preview_capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = self._preview_capture.read()
        except Exception:
            return
        if not ok or frame is None:
            return
        self._preview_popup.set_frame(frame, index)

        slider_x = round(
            (index - self.frame_slider.minimum())
            / max(1, self.frame_slider.maximum() - self.frame_slider.minimum())
            * max(1, self.frame_slider.width() - 1)
        )
        anchor = self.frame_slider.mapToGlobal(
            QPoint(slider_x, self.frame_slider.height() + 7)
        )
        self._preview_popup.adjustSize()
        popup_width = self._preview_popup.width()
        popup_height = self._preview_popup.height()
        screen = QApplication.screenAt(anchor) or QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            popup_x = clamp(
                anchor.x() - popup_width // 2,
                available.left(),
                available.right() - popup_width + 1,
            )
            popup_y = anchor.y()
            if popup_y + popup_height > available.bottom() + 1:
                popup_y = self.frame_slider.mapToGlobal(
                    QPoint(slider_x, -popup_height - 7)
                ).y()
            popup_y = clamp(
                popup_y,
                available.top(),
                available.bottom() - popup_height + 1,
            )
        else:
            popup_x = anchor.x() - popup_width // 2
            popup_y = anchor.y()
        self._preview_popup.move(popup_x, popup_y)
        self._preview_popup.show()
        self._preview_popup.raise_()

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
            self._show_warning("No video loaded", "Open a video file before pressing Play.")
            return
        if self._frame_index >= self._frame_count - 1:
            self._read_and_display(0)
        interval = max(1, round(1000.0 / self._fps))
        self.timer.start(interval)
        self.toggle_button.setText("Ⅱ  Pause")
        self.status_label.setText("Playing")

    def pause_video(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.status_label.setText("Paused")
        self.toggle_button.setText("▶  Play")

    def stop_video(self) -> None:
        self.timer.stop()
        self.toggle_button.setText("▶  Play")
        if self._capture is not None:
            self._read_and_display(0)
            self.status_label.setText("Stopped")

    def _advance_frame(self) -> None:
        next_index = self._frame_index + 1
        if next_index >= self._frame_count:
            self.timer.stop()
            self.toggle_button.setText("▶  Play")
            self.status_label.setText("Playback finished")
            return
        # Frames are read sequentially during playback.  Seeking with
        # CAP_PROP_POS_FRAMES on every timer tick makes playback noticeably slow.
        try:
            ok, frame = self._capture.read()
        except Exception as exc:
            self.timer.stop()
            self.toggle_button.setText("▶  Play")
            self._show_error("Video read error", f"Could not read frame {next_index}.\n\n{exc}")
            return
        if not ok or frame is None:
            self.timer.stop()
            self.toggle_button.setText("▶  Play")
            self._show_error("Video read error", f"Could not read frame {next_index} from the selected video.")
            return
        self._display_decoded_frame(next_index, frame)

    def _on_roi_selected(self, roi) -> None:
        if roi is None:
            self.status_label.setText("ROI cleared — drag on the video to select a region")
            return
        x, y, width, height = roi
        self.status_label.setText(f"ROI: x={x}, y={y}, width={width}, height={height}")
        self._update_histogram()

    def _update_histogram(self) -> None:
        if self._histogram_window is not None and self._histogram_window.isVisible():
            roi = self.canvas.roi()
            if roi is not None and self._current_frame is not None:
                # This is intentionally called for every decoded frame.
                self._histogram_window.set_data(
                    self._current_frame, roi, self.histogram_percent_spin.value()
                )

    def _histogram_percentage_changed(self, _value: int) -> None:
        # Recalculate immediately for the current frame when the user changes
        # the sampling percentage.
        self._update_histogram()

    def show_histogram(self) -> None:
        if self._capture is None or self._current_frame is None:
            self._show_warning("No video loaded", "Open a video and select a frame first.")
            return
        roi = self.canvas.roi()
        if roi is None:
            self._show_warning("No region selected", "Drag over a rectangular region in the video first.")
            return
        if self._histogram_window is None:
            self._histogram_window = HistogramWindow(self)
            self._histogram_window.destroyed.connect(lambda: setattr(self, "_histogram_window", None))
        self._histogram_window.set_data(
            self._current_frame, roi, self.histogram_percent_spin.value()
        )
        self._histogram_window.show()
        self._histogram_window.raise_()
        self._histogram_window.activateWindow()

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self._hide_frame_preview()
        if self._capture is not None:
            self._capture.release()
        if self._preview_capture is not None:
            self._preview_capture.release()
        if self._histogram_window is not None:
            self._histogram_window.close()
        self._preview_popup.close()
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
            "This application needs the following package(s):\n\n"
            + "\n".join(f"• {package}" for package in missing)
            + "\n\nInstall them with:\npython -m pip install -r requirements.txt",
        )
        return 1
    window = VideoPlayer()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
