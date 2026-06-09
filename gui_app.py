#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 智能交通灯控制系统 - PyQt6 桌面应用
"""

import os
import time
import math
import threading
import sys
from collections import deque
from pathlib import Path

import cv2

from algorithm import VAController, FrameFeatures

from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QFont

import theme_manager as tm
from gui.theme import *
from gui.widgets import NavButton, StatLabel, TrafficLightIndicator
from gui.live_view import LiveViewMixin
from gui.session_view import SessionViewMixin
from gui.ui_builders import UIBuilderMixin
from gui.detect_controller import DetectControllerMixin

# ─── HiDPI ─────────────────────────────────────────────
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

# ─── 路径 ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
TEST_OUTPUT_DIR = PROJECT_ROOT / "test" / "output"

VEHICLE_CLASSES = {"car", "van", "bus", "truck"}


# ─── 主窗口 ──────────────────────────────────────────────

class MainWindow(UIBuilderMixin, DetectControllerMixin, LiveViewMixin, SessionViewMixin, QMainWindow):
    PROJECT_ROOT = PROJECT_ROOT
    DATA_DIR = DATA_DIR
    TEST_OUTPUT_DIR = TEST_OUTPUT_DIR

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 智能交通灯控制系统")
        self.resize(1200, 750)

        # 状态
        self.sessions = []
        self.selected_session = None
        self.x_light = "off"
        self.y_light = "off"
        self.sim_running = False
        self.sim_paused = False
        self.sim_speed = 5.0
        self.last_tick = 0
        self.detecting = False
        self.detect_progress = None

        # Vehicle-Actuated 控制器
        self.va_controller = VAController()
        self.feature_extractor = FrameFeatures()
        self.va_frames = []
        self.va_fps = 30.0
        self.va_frame_index = 0
        self.va_sim_time = 0.0
        self.va_features = None
        self.va_pair_summary = None
        self.va_pair_wait_x = 0.0
        self.va_pair_wait_y = 0.0

        # 视频播放器
        self.video_cap = None
        self.video_playing = False
        self.video_fps = 30
        self.video_last_frame_time = 0
        self._detect_latest = {"bgr": None, "fps": 0.0, "count": 0, "idx": 0, "video_fps": None}
        self._detect_dirty = False
        self._live_dual_lock = threading.Lock()
        self._live_dual_dirty = False
        self._live_dual_state = {"X": {}, "Y": {}}
        self._live_dual_wait_x = 0.0
        self._live_dual_wait_y = 0.0
        self._live_dual_gap_x = 0.0
        self._live_dual_gap_y = 0.0
        self._live_dual_last_totals = {"X": 0, "Y": 0}

        # 实时检测 → 仿真数据管道
        self._live_frames: deque = deque(maxlen=500)
        self._live_frames_lock = threading.Lock()
        self._sim_live_mode = False

        self._build_ui()
        self._connect_signals()

        # 定时器
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self._sim_tick)
        self.sim_timer.start(33)

        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self._video_tick)
        self.video_timer.start(33)

        self.detect_timer = QTimer()
        self.detect_timer.timeout.connect(self._check_detect_status)
        self.detect_timer.start(100)

        self._load_sessions()
        self.canvas.update_state()

    @staticmethod
    def _backend_label(backend):
        backend = str(backend or "").lower()
        if backend == "onnxruntime":
            return "ONNX"
        if backend == "openvino":
            return "OpenVINO"
        return backend or "未知"

    @staticmethod
    def _new_live_direction_state():
        return {
            "frame_bgr": None,
            "frame_idx": 0,
            "fps": 0.0,
            "num_objects": 0,
            "track_count_total": 0,
            "track_count_keep": 0,
            "track_count_slow": 0,
            "track_count_filtered": 0,
            "line_count_total": 0,
            "line_count_in": 0,
            "line_count_slow": 0,
            "line_count_out": 0,
            "axis": None,
            "axis_ready": False,
            "video_fps": 30.0,
            "backend": "",
            "source": "",
            "done": False,
        }

    def _reset_live_dual_state(self, video_pairs=None):
        state = {"X": self._new_live_direction_state(), "Y": self._new_live_direction_state()}
        if video_pairs:
            for direction, path in video_pairs:
                direction = str(direction).upper()
                if direction in state:
                    state[direction]["source"] = str(path)
        with self._live_dual_lock:
            self._live_dual_state = state
        self._live_dual_dirty = True
        self._live_dual_wait_x = 0.0
        self._live_dual_wait_y = 0.0
        self._live_dual_gap_x = 0.0
        self._live_dual_gap_y = 0.0
        self._live_dual_last_totals = {"X": 0, "Y": 0}

    @staticmethod
    def _set_preview_frame(widget, frame_bgr):
        if widget is None or frame_bgr is None:
            return
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        widget.set_frame(qimg)

    # ── 主题刷新 ─────────────────────────────────────────

    def refresh_theme(self):
        """系统主题变化时调用，刷新所有颜色和样式。"""
        dark = tm.is_dark()
        _init_colors(dark)
        app = QApplication.instance()
        app.setPalette(tm.create_palette(dark))
        app.setStyleSheet(_make_app_stylesheet())
        _refresh_all_styles()
        for w in self.findChildren(NavButton):
            w._update_style()
        for w in self.findChildren(StatLabel):
            w._refresh_style()
        for w in self.findChildren(TrafficLightIndicator):
            w._refresh_style()
        self.update()


# ─── 主入口 ──────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)

    # 安装主题管理器（跟随 Windows 系统主题）
    tm.install(app)
    app.setStyleSheet(_make_app_stylesheet())

    window = MainWindow()
    window.show()

    # 系统主题变化时自动刷新
    def _on_sys_theme_changed(dark: bool):
        window.refresh_theme()

    tm.on_theme_changed(_on_sys_theme_changed)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
