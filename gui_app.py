#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 智能交通灯控制系统 - PyQt6 桌面应用
"""

import json
import csv
import os
import time
import math
import threading
import sys
from pathlib import Path

import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QPushButton, QLabel, QLineEdit, QTextEdit,
    QFileDialog, QSlider, QComboBox, QListWidget, QListWidgetItem,
    QProgressBar, QTableWidget, QTableWidgetItem, QSplitter,
    QGroupBox, QSizePolicy, QAbstractItemView, QHeaderView,
)
from PyQt6.QtCore import Qt, QTimer, QSize, QRectF, pyqtSignal
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QImage, QPixmap,
    QLinearGradient, QRadialGradient,
)

# ─── HiDPI ─────────────────────────────────────────────
# 必须在 QApplication 创建之前设置，且只能设一次
# 用 env var 让 Qt 自己处理，避免与终端冲突
import os
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

# ─── 路径 ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
TEST_OUTPUT_DIR = PROJECT_ROOT / "test" / "output"

VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike", "bicycle"}

# ─── 颜色 ────────────────────────────────────────────────
C_PRIMARY       = QColor(79, 70, 229)
C_PRIMARY_HOVER = QColor(109, 99, 255)
C_RED           = QColor(220, 38, 38)
C_YELLOW        = QColor(202, 138, 4)
C_GREEN         = QColor(22, 163, 74)
C_RED_DIM       = QColor(220, 38, 38, 60)
C_YELLOW_DIM    = QColor(202, 138, 4, 60)
C_GREEN_DIM     = QColor(22, 163, 74, 60)
C_BLUE          = QColor(37, 99, 235)
C_ORANGE        = QColor(234, 88, 12)

C_BG_BASE       = QColor(240, 242, 245)
C_BG_SURFACE    = QColor(255, 255, 255)
C_BG_ELEVATED   = QColor(241, 243, 245)
C_BG_OVERLAY    = QColor(233, 236, 239)
C_CARD_BG       = QColor(255, 255, 255)
C_CARD_BORDER   = QColor(226, 229, 235)

C_TEXT_PRIMARY   = QColor(17, 24, 39)
C_TEXT_SECONDARY = QColor(55, 65, 81)
C_TEXT_MUTED     = QColor(107, 114, 128)

C_BORDER        = QColor(220, 225, 231)
C_BORDER_LIGHT  = QColor(235, 238, 243)

# Canvas
C_ROAD          = QColor(209, 213, 219)
C_GRASS         = QColor(187, 247, 208)
C_INTERSECTION  = QColor(199, 203, 209)
C_LANE          = QColor(156, 163, 175)
C_CROSSWALK     = QColor(107, 114, 128, 40)
C_SIDEWALK      = QColor(156, 163, 175, 120)
C_STOP_LINE     = QColor(107, 114, 128, 80)
C_CENTER_LINE   = QColor(202, 138, 4, 100)


# ─── 数据加载 ────────────────────────────────────────────

def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_frames(session_dir):
    json_path = os.path.join(session_dir, "frames.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    csv_path = os.path.join(session_dir, "frames.csv")
    if not os.path.exists(csv_path):
        return []
    frames = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = int(row["frame"])
            if fn not in frames:
                frames[fn] = {"frame": fn, "timestamp": float(row["timestamp"]), "detections": []}
            frames[fn]["detections"].append({
                "class": row["class"], "class_id": int(row["class_id"]),
                "confidence": float(row["confidence"]),
                "x1": float(row["x1"]), "y1": float(row["y1"]),
                "x2": float(row["x2"]), "y2": float(row["y2"]),
            })
    return [frames[k] for k in sorted(frames.keys())]


def build_timeline(frames, video_width=1280, fps=30.0):
    YELLOW = 3.0; STD = 10.0; MIN_G = 10.0; MAX_G = 30.0; RATIO = 0.2
    mid_x = video_width / 2
    total_dur = len(frames) / fps
    timeline = []; phase = "X_GREEN"; elapsed = 0.0; idx = 0
    while elapsed < total_dur:
        cx_list, cy_list = [], []
        dur = STD
        if timeline:
            last = timeline[-1]
            dur = last["x_green"] if phase == "X_GREEN" else last["y_green"]
        end_t = elapsed + dur
        while idx < len(frames):
            ft = frames[idx]["frame"] / fps
            if ft > end_t:
                break
            left = right = 0
            for d in frames[idx].get("detections", []):
                if d.get("class", "").lower() not in VEHICLE_CLASSES:
                    continue
                cx = (d["x1"] + d["x2"]) / 2
                if cx < mid_x:
                    left += 1
                else:
                    right += 1
            cx_list.append(left); cy_list.append(right); idx += 1
        ax = sum(cx_list) / len(cx_list) if cx_list else 0
        ay = sum(cy_list) / len(cy_list) if cy_list else 0
        xg, yg = STD, STD
        if ax > ay:
            xg = min(STD + STD * RATIO, MAX_G); yg = max(STD - STD * RATIO, MIN_G)
        elif ay > ax:
            yg = min(STD + STD * RATIO, MAX_G); xg = max(STD - STD * RATIO, MIN_G)
        timeline.append({
            "phase": phase, "start_time": round(elapsed, 1),
            "car_x_avg": round(ax, 1), "car_y_avg": round(ay, 1),
            "x_green": round(xg, 1), "x_red": round(yg + YELLOW, 1),
            "y_green": round(yg, 1), "y_red": round(xg + YELLOW, 1),
            "yellow_duration": YELLOW,
        })
        gt = xg if phase == "X_GREEN" else yg
        elapsed += gt + YELLOW
        phase = "Y_GREEN" if phase == "X_GREEN" else "X_GREEN"
    return timeline, round(total_dur, 1)


# ─── 圆角卡片 ────────────────────────────────────────────

class CardWidget(QGroupBox):
    """带圆角边框的卡片容器"""
    def __init__(self, title="", parent=None):
        super().__init__(title, parent)
        self.setObjectName("card")
        self.setStyleSheet("""
            QGroupBox#card {
                background: #ffffff;
                border: 1px solid #e2e5eb;
                border-radius: 10px;
                margin-top: 0px;
                padding: 14px 12px;
                font-weight: bold;
                color: #374151;
            }
            QGroupBox#card::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
        """)


# ─── 十字路口 Canvas ─────────────────────────────────────

class IntersectionCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.x_color = "off"
        self.y_color = "off"
        self.car_x = None
        self.car_y = None
        self.countdown = None
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def update_state(self, x_color="off", y_color="off", car_x=None, car_y=None, countdown=None):
        self.x_color = x_color
        self.y_color = y_color
        self.car_x = car_x
        self.car_y = car_y
        self.countdown = countdown
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        W = self.width()
        H = self.height()
        road_w = int(W * 0.22)
        cx, cy = W / 2, H / 2

        # 四角草地
        corners = [
            (0, 0, cx - road_w/2, cy - road_w/2),
            (cx + road_w/2, 0, W, cy - road_w/2),
            (0, cy + road_w/2, cx - road_w/2, H),
            (cx + road_w/2, cy + road_w/2, W, H),
        ]
        for x1, y1, x2, y2 in corners:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(C_GRASS))
            p.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
            sw = max(4, int(W * 0.014))
            p.setBrush(QBrush(C_SIDEWALK))
            if x1 == 0:
                p.drawRect(QRectF(x2 - sw, y1, sw, y2 - y1))
            if x2 == W:
                p.drawRect(QRectF(x1, y1, sw, y2 - y1))
            if y1 == 0:
                p.drawRect(QRectF(x1, y2 - sw, x2 - x1, sw))
            if y2 == H:
                p.drawRect(QRectF(x1, y1, x2 - x1, sw))

        # 道路
        p.setBrush(QBrush(C_ROAD))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRect(QRectF(0, cy - road_w/2, W, road_w))
        p.drawRect(QRectF(cx - road_w/2, 0, road_w, H))
        p.setBrush(QBrush(C_INTERSECTION))
        p.drawRect(QRectF(cx - road_w/2, cy - road_w/2, road_w, road_w))

        # 中心线（虚线）
        pen = QPen(C_CENTER_LINE, 1, Qt.PenStyle.DashLine)
        p.setPen(pen)
        for offset in [-2, 2]:
            p.drawLine(int(0), int(cy + offset), int(cx - road_w/2), int(cy + offset))
            p.drawLine(int(cx + road_w/2), int(cy + offset), int(W), int(cy + offset))
            p.drawLine(int(cx + offset), int(0), int(cx + offset), int(cy - road_w/2))
            p.drawLine(int(cx + offset), int(cy + road_w/2), int(cx + offset), int(H))

        # 车道虚线
        pen = QPen(C_LANE, 1, Qt.PenStyle.DashLine)
        p.setPen(pen)
        hw = road_w / 4
        for lo in [-hw, hw]:
            p.drawLine(int(0), int(cy + lo), int(cx - road_w/2), int(cy + lo))
            p.drawLine(int(cx + road_w/2), int(cy + lo), int(W), int(cy + lo))
            p.drawLine(int(cx + lo), int(0), int(cx + lo), int(cy - road_w/2))
            p.drawLine(int(cx + lo), int(cy + road_w/2), int(cx + lo), int(H))

        # 人行横道
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(C_CROSSWALK))
        for i in range(8):
            ow = road_w / 8 - 4
            ox = cx - road_w/2 + i * (road_w/8) + 2
            p.drawRect(QRectF(ox, cy - road_w/2 - 14, ow, 12))
            p.drawRect(QRectF(ox, cy + road_w/2 + 2, ow, 12))
            oh = road_w / 8 - 4
            oy = cy - road_w/2 + i * (road_w/8) + 2
            p.drawRect(QRectF(cx - road_w/2 - 14, oy, 12, oh))
            p.drawRect(QRectF(cx + road_w/2 + 2, oy, 12, oh))

        # 停车线
        p.setPen(QPen(C_STOP_LINE, 2))
        p.drawLine(int(cx - road_w/2), int(cy - road_w/2 - 2), int(cx), int(cy - road_w/2 - 2))
        p.drawLine(int(cx), int(cy + road_w/2 + 2), int(cx + road_w/2), int(cy + road_w/2 + 2))
        p.drawLine(int(cx - road_w/2 - 2), int(cy), int(cx - road_w/2 - 2), int(cy + road_w/2))
        p.drawLine(int(cx + road_w/2 + 2), int(cy - road_w/2), int(cx + road_w/2 + 2), int(cy))

        # 道路标签
        fs = max(10, int(W * 0.022))
        font = QFont("Microsoft YaHei", fs)
        p.setFont(font)
        p.setPen(QPen(C_TEXT_SECONDARY))
        p.drawText(int(cx - 18), int(cy - road_w/2 - 36), "X")
        p.drawText(int(cx - 18), int(cy + road_w/2 + 22 + fs), "X")
        p.drawText(int(cx - road_w/2 - 30), int(cy - 8 + fs), "Y")
        p.drawText(int(cx + road_w/2 + 16), int(cy - 8 + fs), "Y")

        # 交通灯
        def draw_light(lx, ly, ac):
            bw = max(20, int(W * 0.048))
            bh = max(56, int(H * 0.14))
            r = max(5, int(W * 0.013))
            # 外框
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(80, 80, 90)))
            p.drawRoundedRect(QRectF(lx - bw/2 - 2, ly - bh/2 - 2, bw + 4, bh + 4), 3, 3)
            p.setBrush(QBrush(QColor(50, 50, 60)))
            p.drawRoundedRect(QRectF(lx - bw/2, ly - bh/2, bw, bh), 2, 2)
            for i, cn in enumerate(["red", "yellow", "green"]):
                by = ly - bh/3 + i * (bh/3)
                is_on = cn == ac
                if cn == "red":
                    fill = C_RED if is_on else C_RED_DIM
                elif cn == "yellow":
                    fill = C_YELLOW if is_on else C_YELLOW_DIM
                else:
                    fill = C_GREEN if is_on else C_GREEN_DIM
                if is_on:
                    glow = QColor(fill)
                    glow.setAlpha(30)
                    p.setBrush(QBrush(glow))
                    p.setPen(Qt.PenStyle.NoPen)
                    p.drawEllipse(QRectF(lx - r - 6, by - r - 6, (r + 6)*2, (r + 6)*2))
                p.setBrush(QBrush(fill))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(QRectF(lx - r, by - r, r*2, r*2))

        off_x = max(20, int(W * 0.04))
        off_y = max(30, int(H * 0.08))
        draw_light(cx - road_w/2 - off_x, cy - road_w/2 - off_y, self.y_color)
        draw_light(cx + road_w/2 + off_x, cy + road_w/2 + off_y, self.y_color)
        draw_light(cx - road_w/2 - off_y, cy + road_w/2 + off_x, self.x_color)
        draw_light(cx + road_w/2 + off_y, cy - road_w/2 - off_x, self.x_color)

        # 车辆数
        if self.car_x is not None:
            p.setPen(QPen(C_BLUE))
            p.setFont(QFont("Microsoft YaHei", fs))
            p.drawText(20, int(cy - 14 + fs), f"X: {self.car_x}")
            p.drawText(W - 80, int(cy - 14 + fs), f"X: {self.car_x}")
        if self.car_y is not None:
            p.setPen(QPen(C_ORANGE))
            p.drawText(int(cx - 14), 14 + fs, f"Y: {self.car_y}")
            p.drawText(int(cx - 14), H - 30 + fs, f"Y: {self.car_y}")

        # 倒计时
        if self.countdown is not None:
            cfs = max(14, int(W * 0.03))
            p.setFont(QFont("Microsoft YaHei", cfs, QFont.Weight.Bold))
            p.setPen(QPen(C_TEXT_PRIMARY))
            txt = str(math.ceil(self.countdown))
            p.drawText(QRectF(cx - cfs, cy - cfs/2, cfs*2, cfs*2), Qt.AlignmentFlag.AlignCenter, txt)

        p.end()


# ─── 视频预览 Widget ─────────────────────────────────────

class VideoPreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_frame = None
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_frame(self, qimage):
        self.current_frame = qimage
        self.update()

    def clear(self):
        self.current_frame = None
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if self.current_frame:
            scaled = self.current_frame.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            p.drawImage(x, y, scaled)
        else:
            p.fillRect(self.rect(), QColor(20, 20, 26))
            p.setPen(QPen(C_TEXT_MUTED))
            p.setFont(QFont("Microsoft YaHei", 12))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "无视频")
        p.end()


# ─── 导航按钮 ────────────────────────────────────────────

class NavButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self._active = False
        self.setFixedHeight(36)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_style()

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, val):
        self._active = val
        self._update_style()

    def _update_style(self):
        if self._active:
            self.setStyleSheet("""
                QPushButton {
                    background: #4f46e5; color: white; border: none;
                    border-radius: 8px; padding: 6px 14px; text-align: left;
                    font-size: 13px; font-weight: bold;
                }
                QPushButton:hover { background: #6d63ff; }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background: #f1f3f5; color: #374151; border: none;
                    border-radius: 8px; padding: 6px 14px; text-align: left;
                    font-size: 13px;
                }
                QPushButton:hover { background: #e9ecef; }
            """)


# ─── 统计数字标签 ────────────────────────────────────────

class StatLabel(QWidget):
    def __init__(self, value="0", label="", color=C_PRIMARY, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.val_label = QLabel(value)
        self.val_label.setStyleSheet(f"color: {color.name()}; font-size: 20px; font-weight: bold;")
        self.val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label = QLabel(label)
        self.name_label.setStyleSheet(f"color: {C_TEXT_MUTED.name()}; font-size: 11px;")
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.val_label)
        layout.addWidget(self.name_label)

    def set_value(self, v):
        self.val_label.setText(str(v))


# ─── 交通灯指示器 ────────────────────────────────────────

class TrafficLightIndicator(QWidget):
    def __init__(self, label="X 方向", color=C_BLUE, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {color.name()}; font-weight: bold; font-size: 12px;")
        layout.addWidget(lbl)
        self.lights = {}
        for name, dim_c in [("red", C_RED_DIM), ("yellow", C_YELLOW_DIM), ("green", C_GREEN_DIM)]:
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {dim_c.name()}; font-size: 18px;")
            layout.addWidget(dot)
            self.lights[name] = (dot, dim_c)

    def set_active(self, color_name):
        on_map = {"red": C_RED, "yellow": C_YELLOW, "green": C_GREEN}
        for name, (dot, dim_c) in self.lights.items():
            if name == color_name:
                dot.setStyleSheet(f"color: {on_map[name].name()}; font-size: 18px;")
            else:
                dot.setStyleSheet(f"color: {dim_c.name()}; font-size: 18px;")


# ─── 主窗口 ──────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 智能交通灯控制系统")
        self.resize(1200, 750)

        # 状态
        self.sessions = []
        self.selected_session = None
        self.timeline = []
        self.total_duration = 0
        self.current_cycle = -1
        self.x_light = "off"
        self.y_light = "off"
        self.sim_running = False
        self.sim_paused = False
        self.sim_speed = 5.0
        self.cycle_elapsed = 0.0
        self.yellow_elapsed = 0.0
        self.in_yellow = False
        self.last_tick = 0
        self.detecting = False
        self.detect_progress = ""

        # 视频播放器
        self.video_cap = None
        self.video_playing = False
        self.video_fps = 30
        self.video_last_frame_time = 0

        self._build_ui()
        self._connect_signals()

        # 定时器
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self._sim_tick)
        self.sim_timer.start(33)  # ~30fps

        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self._video_tick)
        self.video_timer.start(33)

        self.detect_timer = QTimer()
        self.detect_timer.timeout.connect(self._check_detect_status)
        self.detect_timer.start(100)

        self._load_sessions()
        self.canvas.update_state()

    # ── UI 构建 ──────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        central.setStyleSheet("background: #f0f2f5;")

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 主体：侧栏 + 内容
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # ── 左侧导航栏 ──
        sidebar = QWidget()
        sidebar.setFixedWidth(200)
        sidebar.setStyleSheet("background: #ffffff; border-right: 1px solid #e2e5eb;")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 12, 8, 8)
        sidebar_layout.setSpacing(8)

        self.nav_yolo = NavButton("  YOLO 视频分析")
        self.nav_yolo.active = True
        self.nav_traffic = NavButton("  交通灯仿真")
        sidebar_layout.addWidget(self.nav_yolo)
        sidebar_layout.addWidget(self.nav_traffic)

        sidebar_layout.addSpacing(8)
        line = QWidget()
        line.setFixedHeight(1)
        line.setStyleSheet("background: #e2e5eb;")
        sidebar_layout.addWidget(line)
        sidebar_layout.addSpacing(8)

        lbl = QLabel("检测记录")
        lbl.setStyleSheet(f"color: {C_TEXT_SECONDARY.name()}; font-size: 12px; font-weight: bold;")
        sidebar_layout.addWidget(lbl)

        self.session_list = QListWidget()
        self.session_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #e2e5eb; border-radius: 6px;
                background: #f9fafb; font-size: 11px; outline: none;
            }
            QListWidget::item { padding: 4px 8px; border-bottom: 1px solid #f3f4f6; }
            QListWidget::item:selected { background: #eef2ff; color: #4f46e5; }
        """)
        sidebar_layout.addWidget(self.session_list)
        sidebar_layout.addStretch()

        body.addWidget(sidebar)

        # ── 右侧内容区 ──
        self.stack = QStackedWidget()
        self.stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._build_yolo_page()
        self._build_traffic_page()

        body.addWidget(self.stack, 1)
        main_layout.addLayout(body, 1)

        # ── 底部状态栏 ──
        status_bar = QWidget()
        status_bar.setFixedHeight(28)
        status_bar.setStyleSheet("background: #ffffff; border-top: 1px solid #e2e5eb;")
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(12, 0, 12, 0)
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet(f"color: {C_TEXT_MUTED.name()}; font-size: 11px;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        brand = QLabel("OpenVINO + YOLOv26")
        brand.setStyleSheet(f"color: {C_TEXT_MUTED.name()}; font-size: 11px;")
        status_layout.addWidget(brand)
        main_layout.addWidget(status_bar)

    def _build_yolo_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 8)
        layout.setSpacing(8)

        # 上传区卡片
        card_upload = CardWidget("YOLOv26 视频检测")
        upload_layout = QVBoxLayout(card_upload)
        upload_layout.setSpacing(6)
        row = QHBoxLayout()
        self.video_path_input = QLineEdit()
        self.video_path_input.setPlaceholderText("输入视频路径...")
        self.video_path_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid #d1d5db; border-radius: 6px;
                padding: 6px 10px; background: #f9fafb; font-size: 12px;
            }
            QLineEdit:focus { border-color: #4f46e5; background: #fff; }
        """)
        row.addWidget(self.video_path_input, 1)

        btn_browse = QPushButton("浏览")
        btn_browse.setFixedSize(70, 32)
        btn_browse.setStyleSheet(self._btn_style(C_PRIMARY))
        btn_browse.clicked.connect(self._on_browse_video)
        row.addWidget(btn_browse)

        self.btn_detect = QPushButton("开始检测")
        self.btn_detect.setFixedSize(90, 32)
        self.btn_detect.setStyleSheet(self._btn_style(C_GREEN))
        self.btn_detect.clicked.connect(self._on_start_detect)
        row.addWidget(self.btn_detect)
        upload_layout.addLayout(row)

        row2 = QHBoxLayout()
        self.detect_status = QLabel("就绪")
        self.detect_status.setStyleSheet(f"color: {C_TEXT_SECONDARY.name()}; font-size: 12px;")
        row2.addWidget(self.detect_status)
        row2.addStretch()
        self.btn_play = QPushButton("播放")
        self.btn_play.setFixedSize(70, 28)
        self.btn_play.setStyleSheet(self._btn_style(C_PRIMARY, 10))
        self.btn_play.clicked.connect(self._on_play_video)
        row2.addWidget(self.btn_play)
        upload_layout.addLayout(row2)
        layout.addWidget(card_upload)

        # 下方两栏
        bottom = QHBoxLayout()
        bottom.setSpacing(8)

        # 左列
        left_col = QVBoxLayout()
        left_col.setSpacing(8)

        # 统计
        card_stats = CardWidget()
        stats_layout = QHBoxLayout(card_stats)
        stats_layout.setSpacing(16)
        self.stat_frames = StatLabel("0", "帧数", C_BLUE)
        self.stat_detections = StatLabel("0", "检测数", C_PRIMARY)
        self.stat_vehicles = StatLabel("0", "车辆数", C_GREEN)
        self.stat_fps = StatLabel("0", "FPS", C_ORANGE)
        stats_layout.addWidget(self.stat_frames)
        stats_layout.addWidget(self.stat_detections)
        stats_layout.addWidget(self.stat_vehicles)
        stats_layout.addWidget(self.stat_fps)
        left_col.addWidget(card_stats)

        # 详情
        card_detail = CardWidget("检测详情")
        detail_layout = QVBoxLayout(card_detail)
        self.session_detail = QTextEdit()
        self.session_detail.setReadOnly(True)
        self.session_detail.setFixedHeight(70)
        self.session_detail.setPlaceholderText("从左侧选择一条检测记录")
        self.session_detail.setStyleSheet("""
            QTextEdit {
                border: 1px solid #e2e5eb; border-radius: 6px;
                background: #f9fafb; font-size: 11px; padding: 4px;
            }
        """)
        detail_layout.addWidget(self.session_detail)
        left_col.addWidget(card_detail)

        # 类别统计
        card_class = CardWidget("类别统计")
        class_layout = QVBoxLayout(card_class)
        self.class_table = QTableWidget(0, 3)
        self.class_table.setHorizontalHeaderLabels(["类别", "数量", "占比"])
        self.class_table.horizontalHeader().setStretchLastSection(True)
        self.class_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.class_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.class_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.class_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.class_table.setMaximumHeight(200)
        self.class_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #e2e5eb; border-radius: 6px;
                background: #f9fafb; font-size: 11px; gridline-color: #e5e7eb;
            }
            QHeaderView::section {
                background: #f3f4f6; border: none; padding: 4px; font-weight: bold;
            }
        """)
        class_layout.addWidget(self.class_table)
        left_col.addWidget(card_class)
        left_col.addStretch()

        left_w = QWidget()
        left_w.setLayout(left_col)
        left_w.setFixedWidth(340)
        bottom.addWidget(left_w)

        # 右列：视频预览
        right_col = QVBoxLayout()
        card_video = CardWidget("视频预览")
        video_layout = QVBoxLayout(card_video)
        self.video_preview = VideoPreviewWidget()
        video_layout.addWidget(self.video_preview)
        right_col.addWidget(card_video, 1)

        right_w = QWidget()
        right_w.setLayout(right_col)
        bottom.addWidget(right_w, 1)

        layout.addLayout(bottom, 1)
        self.stack.addWidget(page)

    def _build_traffic_page(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # 画布
        card_canvas = CardWidget("路口状态")
        canvas_layout = QVBoxLayout(card_canvas)
        self.canvas = IntersectionCanvas()
        canvas_layout.addWidget(self.canvas, 1)
        layout.addWidget(card_canvas, 1)

        # 控制面板
        card_ctrl = CardWidget("交通灯状态")
        ctrl_layout = QVBoxLayout(card_ctrl)
        ctrl_layout.setSpacing(8)

        # 交通灯指示
        self.xl_indicator = TrafficLightIndicator("X 方向", C_BLUE)
        self.yl_indicator = TrafficLightIndicator("Y 方向", C_ORANGE)
        ctrl_layout.addWidget(self.xl_indicator)
        ctrl_layout.addWidget(self.yl_indicator)

        # 阶段
        phase_row = QHBoxLayout()
        phase_lbl = QLabel("阶段:")
        phase_lbl.setStyleSheet(f"color: {C_TEXT_MUTED.name()}; font-size: 12px;")
        phase_row.addWidget(phase_lbl)
        self.phase_label = QLabel("--")
        self.phase_label.setStyleSheet(f"color: {C_TEXT_PRIMARY.name()}; font-size: 12px; font-weight: bold;")
        phase_row.addWidget(self.phase_label)
        phase_row.addStretch()
        ctrl_layout.addLayout(phase_row)

        # 倒计时
        timer_row = QHBoxLayout()
        timer_lbl = QLabel("倒计时")
        timer_lbl.setStyleSheet(f"color: {C_TEXT_MUTED.name()}; font-size: 12px;")
        timer_row.addWidget(timer_lbl)
        self.timer_text = QLabel("--")
        self.timer_text.setStyleSheet(f"color: {C_GREEN.name()}; font-size: 14px; font-weight: bold;")
        timer_row.addWidget(self.timer_text)
        timer_row.addStretch()
        ctrl_layout.addLayout(timer_row)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar { background: #e5e7eb; border: none; border-radius: 3px; }
            QProgressBar::chunk { background: #22c55e; border-radius: 3px; }
        """)
        ctrl_layout.addWidget(self.progress_bar)

        # 按钮
        btn_row = QHBoxLayout()
        self.btn_start_sim = QPushButton("▶ 开始")
        self.btn_start_sim.setFixedSize(90, 32)
        self.btn_start_sim.setStyleSheet(self._btn_style(C_GREEN))
        self.btn_start_sim.clicked.connect(self._on_start_sim)
        btn_row.addWidget(self.btn_start_sim)

        self.btn_pause_sim = QPushButton("⏸ 暂停")
        self.btn_pause_sim.setFixedSize(90, 32)
        self.btn_pause_sim.setStyleSheet(self._btn_style(C_YELLOW))
        self.btn_pause_sim.clicked.connect(self._on_pause_sim)
        btn_row.addWidget(self.btn_pause_sim)

        self.btn_reset_sim = QPushButton("■ 重置")
        self.btn_reset_sim.setFixedSize(90, 32)
        self.btn_reset_sim.setStyleSheet(self._btn_style(C_RED))
        self.btn_reset_sim.clicked.connect(self._on_reset_sim)
        btn_row.addWidget(self.btn_reset_sim)
        ctrl_layout.addLayout(btn_row)

        # 速度
        speed_lbl = QLabel("速度")
        speed_lbl.setStyleSheet(f"color: {C_TEXT_MUTED.name()}; font-size: 12px;")
        ctrl_layout.addWidget(speed_lbl)
        speed_row = QHBoxLayout()
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 20)
        self.speed_slider.setValue(5)
        self.speed_slider.setStyleSheet("""
            QSlider::groove:horizontal { background: #e5e7eb; height: 6px; border-radius: 3px; }
            QSlider::handle:horizontal { background: #4f46e5; width: 14px; margin: -5px 0; border-radius: 7px; }
        """)
        self.speed_slider.valueChanged.connect(lambda v: setattr(self, 'sim_speed', float(v)))
        speed_row.addWidget(self.speed_slider)
        self.speed_val = QLabel("5x")
        self.speed_val.setStyleSheet(f"color: {C_TEXT_PRIMARY.name()}; font-size: 12px; font-weight: bold;")
        self.speed_slider.valueChanged.connect(lambda v: self.speed_val.setText(f"{v}x"))
        speed_row.addWidget(self.speed_val)
        ctrl_layout.addLayout(speed_row)

        # 数据源
        ds_lbl = QLabel("数据源")
        ds_lbl.setStyleSheet(f"color: {C_TEXT_MUTED.name()}; font-size: 12px;")
        ctrl_layout.addWidget(ds_lbl)
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItem("(默认)")
        self.data_source_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #d1d5db; border-radius: 6px;
                padding: 4px 8px; background: #f9fafb; font-size: 11px;
            }
        """)
        ctrl_layout.addWidget(self.data_source_combo)

        ctrl_layout.addSpacing(4)
        line = QWidget()
        line.setFixedHeight(1)
        line.setStyleSheet("background: #e2e5eb;")
        ctrl_layout.addWidget(line)
        ctrl_layout.addSpacing(4)

        # 周期信息
        ci_lbl = QLabel("周期信息")
        ci_lbl.setStyleSheet(f"color: {C_TEXT_MUTED.name()}; font-size: 12px;")
        ctrl_layout.addWidget(ci_lbl)
        self.cycle_info = QTextEdit()
        self.cycle_info.setReadOnly(True)
        self.cycle_info.setFixedHeight(60)
        self.cycle_info.setPlaceholderText("点击 ▶ 开始")
        self.cycle_info.setStyleSheet("""
            QTextEdit {
                border: 1px solid #e2e5eb; border-radius: 6px;
                background: #f9fafb; font-size: 11px; padding: 4px;
            }
        """)
        ctrl_layout.addWidget(self.cycle_info)

        # 时间线
        tl_lbl = QLabel("时间线")
        tl_lbl.setStyleSheet(f"color: {C_TEXT_MUTED.name()}; font-size: 12px;")
        ctrl_layout.addWidget(tl_lbl)
        self.timeline_table = QTableWidget(0, 4)
        self.timeline_table.setHorizontalHeaderLabels(["相位", "时间", "车辆", "绿灯"])
        self.timeline_table.horizontalHeader().setStretchLastSection(True)
        self.timeline_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.timeline_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.timeline_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.timeline_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.timeline_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.timeline_table.setMaximumHeight(200)
        self.timeline_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #e2e5eb; border-radius: 6px;
                background: #f9fafb; font-size: 11px; gridline-color: #e5e7eb;
            }
            QHeaderView::section {
                background: #f3f4f6; border: none; padding: 4px; font-weight: bold;
            }
        """)
        ctrl_layout.addWidget(self.timeline_table)
        ctrl_layout.addStretch()

        ctrl_w = QWidget()
        ctrl_w.setLayout(ctrl_layout)
        ctrl_w.setFixedWidth(300)
        layout.addWidget(ctrl_w)

        self.stack.addWidget(page)

    # ── 样式工具 ─────────────────────────────────────────

    @staticmethod
    def _btn_style(bg_color, radius=6):
        return f"""
            QPushButton {{
                background: {bg_color.name()}; color: white; border: none;
                border-radius: {radius}px; font-size: 12px; font-weight: bold;
            }}
            QPushButton:hover {{ background: {bg_color.lighter(115).name()}; }}
            QPushButton:disabled {{ background: #9ca3af; }}
        """

    # ── 信号连接 ─────────────────────────────────────────

    def _connect_signals(self):
        self.nav_yolo.clicked.connect(lambda: self._switch_page(0))
        self.nav_traffic.clicked.connect(lambda: self._switch_page(1))
        self.session_list.currentRowChanged.connect(self._on_session_select)

    def _switch_page(self, idx):
        self.stack.setCurrentIndex(idx)
        self.nav_yolo.active = (idx == 0)
        self.nav_traffic.active = (idx == 1)

    # ── YOLO 检测 ────────────────────────────────────────

    def _on_browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi);;所有文件 (*.*)"
        )
        if path:
            self.video_path_input.setText(path)

    def _on_start_detect(self):
        video_path = self.video_path_input.text().strip()
        if not video_path or not os.path.exists(video_path):
            self.detect_status.setText("错误: 视频路径不存在")
            self.detect_status.setStyleSheet(f"color: {C_RED.name()}; font-size: 12px;")
            return
        if self.detecting:
            return

        model_path = PROJECT_ROOT / "public" / "yolo-v26" / "ir_model" / "yolo26n.xml"
        if not model_path.exists():
            self.detect_status.setText("错误: YOLOv26 模型文件不存在")
            self.detect_status.setStyleSheet(f"color: {C_RED.name()}; font-size: 12px;")
            return

        os.makedirs(str(TEST_OUTPUT_DIR), exist_ok=True)
        basename = Path(video_path).stem
        output_path = str(TEST_OUTPUT_DIR / f"output_{basename}.mp4")

        self.detecting = True
        self._stop_video()
        self.detect_status.setText("检测中... (YOLOv26)")
        self.detect_status.setStyleSheet(f"color: {C_PRIMARY.name()}; font-size: 12px;")
        self.btn_detect.setEnabled(False)

        def run_detect():
            orig = (cv2.imshow, cv2.waitKey, cv2.destroyAllWindows)
            try:
                cv2.imshow = lambda *_a, **_k: None
                cv2.waitKey = lambda *_a, **_k: -1
                cv2.destroyAllWindows = lambda: None

                import yolov26 as yolo
                cwd = os.getcwd()
                os.chdir(str(PROJECT_ROOT))
                yolo.detect_video(video_path, output_path)
                os.chdir(cwd)
                self.detect_progress = output_path
            except Exception as e:
                import traceback
                self.detect_progress = f"FAIL:{str(e)[:200]}\n{traceback.format_exc()[:300]}"
            finally:
                cv2.imshow, cv2.waitKey, cv2.destroyAllWindows = orig
                self.detecting = False

        threading.Thread(target=run_detect, daemon=True).start()

    def _check_detect_status(self):
        if not self.detecting and self.detect_progress:
            prog = self.detect_progress
            self.detect_progress = ""
            self.btn_detect.setEnabled(True)
            if prog.startswith("FAIL:"):
                self.detect_status.setText(prog[5:])
                self.detect_status.setStyleSheet(f"color: {C_RED.name()}; font-size: 12px;")
            else:
                self.detect_status.setText("检测完成，正在加载视频...")
                self.detect_status.setStyleSheet(f"color: {C_GREEN.name()}; font-size: 12px;")
                if self._load_video(prog):
                    self.btn_play.setText("⏸ 暂停")
                    self.detect_status.setText(f"播放: {Path(prog).name}")
                    self.detect_status.setStyleSheet(f"color: {C_GREEN.name()}; font-size: 12px;")
                else:
                    self.detect_status.setText("检测完成，但视频无法播放")
                    self.detect_status.setStyleSheet(f"color: {C_ORANGE.name()}; font-size: 12px;")
            self._load_sessions()

    # ── 视频播放 ─────────────────────────────────────────

    def _load_video(self, path):
        self._stop_video()
        self.video_cap = cv2.VideoCapture(path)
        if not self.video_cap.isOpened():
            self.video_cap = None
            return False
        self.video_fps = max(1, self.video_cap.get(cv2.CAP_PROP_FPS) or 30)
        self.video_playing = True
        self.video_last_frame_time = time.time()
        self._read_video_frame()
        return True

    def _read_video_frame(self):
        if not self.video_cap:
            return False
        ret, frame = self.video_cap.read()
        if not ret:
            self.video_playing = False
            self.detect_status.setText("播放结束")
            self.detect_status.setStyleSheet(f"color: {C_TEXT_MUTED.name()}; font-size: 12px;")
            self.btn_play.setText("▶ 播放")
            return False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_preview.set_frame(qimg)
        return True

    def _video_tick(self):
        if not self.video_playing or not self.video_cap:
            return
        now = time.time()
        interval = 1.0 / self.video_fps
        if now - self.video_last_frame_time < interval:
            return
        self.video_last_frame_time = now
        self._read_video_frame()

    def _on_play_video(self):
        if self.video_cap:
            self.video_playing = not self.video_playing
            if self.video_playing:
                self.video_last_frame_time = time.time()
                self.btn_play.setText("⏸ 暂停")
            else:
                self.btn_play.setText("▶ 播放")

    def _stop_video(self):
        self.video_playing = False
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        self.video_preview.clear()
        self.btn_play.setText("▶ 播放")

    # ── 检测记录 ─────────────────────────────────────────

    def _load_sessions(self):
        self.sessions = []
        self.session_list.clear()
        if not DATA_DIR.exists():
            return
        for d in sorted(DATA_DIR.iterdir(), reverse=True):
            if d.is_dir() and d.name.startswith("detection_"):
                summary = load_json(d / "summary.json")
                self.sessions.append({"name": d.name, "path": str(d), "summary": summary})
                self.session_list.addItem(d.name)
        # 更新数据源下拉
        self.data_source_combo.clear()
        self.data_source_combo.addItem("(默认)")
        for s in self.sessions:
            self.data_source_combo.addItem(s["name"])

    def _on_session_select(self, row):
        if row < 0 or row >= len(self.sessions):
            return
        s = self.sessions[row]
        self.selected_session = s
        summary = s.get("summary", {})

        classes = summary.get("class_counts", {})
        total = summary.get("total_detections", 0)
        frames = summary.get("total_frames", 0)
        fps_val = summary.get("avg_fps", 0)
        vi = summary.get("video_info", {})
        # 优先使用去重统计（unique_class_counts），兼容旧数据
        unique_counts = summary.get("unique_class_counts", classes)
        vehicle_count = summary.get("unique_vehicle_count",
                                    sum(v for k, v in unique_counts.items() if k.lower() in VEHICLE_CLASSES))

        self.stat_frames.set_value(str(frames))
        self.stat_detections.set_value(str(total))
        self.stat_vehicles.set_value(str(vehicle_count))
        self.stat_fps.set_value(str(round(fps_val, 1)))

        # 类别表格 — 使用去重统计
        self.class_table.setRowCount(0)
        unique_total = sum(unique_counts.values())
        for cls, count in sorted(unique_counts.items(), key=lambda x: -x[1]):
            pct = f"{(count / unique_total * 100):.1f}%" if unique_total else "0%"
            row_idx = self.class_table.rowCount()
            self.class_table.insertRow(row_idx)
            self.class_table.setItem(row_idx, 0, QTableWidgetItem(cls))
            self.class_table.setItem(row_idx, 1, QTableWidgetItem(str(count)))
            self.class_table.setItem(row_idx, 2, QTableWidgetItem(pct))

        info = (
            f"视频源: {summary.get('source', 'N/A')}\n"
            f"分辨率: {vi.get('width', '?')}x{vi.get('height', '?')}\n"
            f"总帧数: {frames}  总检测: {total}\n"
            f"车辆数: {vehicle_count}  FPS: {fps_val:.1f}"
        )
        self.session_detail.setText(info)

    # ── 交通灯仿真 ───────────────────────────────────────

    def _default_timeline(self):
        self.timeline = []
        for i in range(20):
            is_x = i % 2 == 0
            self.timeline.append({
                "phase": "X_GREEN" if is_x else "Y_GREEN",
                "start_time": i * 13,
                "car_x_avg": round(3 + (i % 3) * 1.5, 1),
                "car_y_avg": round(2 + (i % 4) * 1.2, 1),
                "x_green": 10, "x_red": 13, "y_green": 10, "y_red": 13,
                "yellow_duration": 3,
            })
        self.total_duration = 260

    def _load_timeline(self):
        sel = self.data_source_combo.currentText()
        if sel and sel != "(默认)":
            session_dir = os.path.join(DATA_DIR, sel)
            frames = load_frames(session_dir)
            if frames:
                summary = load_json(os.path.join(session_dir, "summary.json"))
                vi = summary.get("video_info", {})
                vw = vi.get("width", 1280)
                fps_val = vi.get("fps", 30)
                self.timeline, self.total_duration = build_timeline(frames, vw, fps_val)
            else:
                self._default_timeline()
        else:
            self._default_timeline()

        # 时间线表格
        self.timeline_table.setRowCount(0)
        for t in self.timeline:
            phase_text = "X绿" if t["phase"] == "X_GREEN" else "Y绿"
            row_idx = self.timeline_table.rowCount()
            self.timeline_table.insertRow(row_idx)
            self.timeline_table.setItem(row_idx, 0, QTableWidgetItem(phase_text))
            self.timeline_table.setItem(row_idx, 1, QTableWidgetItem(f"{t['start_time']}s"))
            self.timeline_table.setItem(row_idx, 2, QTableWidgetItem(f"X:{t['car_x_avg']} Y:{t['car_y_avg']}"))
            green = t["x_green"] if t["phase"] == "X_GREEN" else t["y_green"]
            self.timeline_table.setItem(row_idx, 3, QTableWidgetItem(f"{green}s"))

        self.current_cycle = -1
        self.cycle_elapsed = 0
        self.in_yellow = False

    def _on_start_sim(self):
        if self.sim_running and not self.sim_paused:
            return
        if not self.sim_running:
            self._load_timeline()
            self.sim_running = True
            self.sim_paused = False
            self.current_cycle = 0
            self.cycle_elapsed = 0
            self.in_yellow = False
            self.yellow_elapsed = 0
            self.last_tick = time.time()
        else:
            self.sim_paused = False
            self.last_tick = time.time()

    def _on_pause_sim(self):
        self.sim_paused = True

    def _on_reset_sim(self):
        self.sim_running = False
        self.sim_paused = False
        self.current_cycle = -1
        self.x_light = "off"
        self.y_light = "off"
        self.canvas.update_state()
        self.timer_text.setText("--")
        self.progress_bar.setValue(0)
        self.cycle_info.setText("点击 ▶ 开始模拟")

    def _sim_tick(self):
        if not self.sim_running or self.sim_paused:
            return
        now = time.time()
        dt = (now - self.last_tick) * self.sim_speed
        self.last_tick = now

        tl = self.timeline
        if not tl or self.current_cycle >= len(tl):
            self._on_reset_sim()
            self.cycle_info.setText("模拟结束")
            return

        cycle = tl[self.current_cycle]
        is_x = cycle["phase"] == "X_GREEN"
        green_dur = cycle["x_green"] if is_x else cycle["y_green"]
        yellow_dur = cycle.get("yellow_duration", 3)

        if not self.in_yellow:
            self.cycle_elapsed += dt
            remaining = max(0, green_dur - self.cycle_elapsed)
            self.timer_text.setText(f"{math.ceil(remaining)}s")
            self.progress_bar.setValue(int(self.cycle_elapsed / green_dur * 100) if green_dur > 0 else 0)

            if is_x:
                self.x_light = "green"; self.y_light = "red"
                self.canvas.update_state("green", "red", round(cycle["car_x_avg"]), round(cycle["car_y_avg"]), remaining)
            else:
                self.x_light = "red"; self.y_light = "green"
                self.canvas.update_state("red", "green", round(cycle["car_x_avg"]), round(cycle["car_y_avg"]), remaining)

            self.xl_indicator.set_active(self.x_light)
            self.yl_indicator.set_active(self.y_light)

            phase_label = "X路绿灯 / Y路红灯" if is_x else "Y路绿灯 / X路红灯"
            self.phase_label.setText(phase_label)
            self.phase_label.setStyleSheet(f"color: {C_BLUE.name() if is_x else C_ORANGE.name()}; font-size: 12px; font-weight: bold;")
            self.cycle_info.setText(
                f"周期 {self.current_cycle+1}/{len(tl)}\n"
                f"X路: {cycle['car_x_avg']}辆(均) | Y路: {cycle['car_y_avg']}辆(均)\n"
                f"绿灯: {green_dur}s | 红灯: {cycle['x_red'] if is_x else cycle['y_red']}s"
            )

            if self.cycle_elapsed >= green_dur:
                self.in_yellow = True
                self.yellow_elapsed = 0
        else:
            self.yellow_elapsed += dt
            remaining = max(0, yellow_dur - self.yellow_elapsed)
            self.timer_text.setText(f"黄灯 {math.ceil(remaining)}s")
            self.progress_bar.setValue(int(self.yellow_elapsed / yellow_dur * 100) if yellow_dur > 0 else 0)

            flash = int(self.yellow_elapsed * 3) % 2 == 0
            xc = "yellow" if flash else "off"
            yc = "yellow" if flash else "off"
            self.x_light = xc; self.y_light = yc
            self.canvas.update_state(xc, yc, round(cycle["car_x_avg"]), round(cycle["car_y_avg"]), remaining)
            self.xl_indicator.set_active(xc)
            self.yl_indicator.set_active(yc)

            self.phase_label.setText("黄灯过渡")
            self.phase_label.setStyleSheet(f"color: {C_YELLOW.name()}; font-size: 12px; font-weight: bold;")
            self.cycle_info.setText(
                f"周期 {self.current_cycle+1} -> {self.current_cycle+2}\n"
                f"双向黄灯 {yellow_dur}s"
            )

            if self.yellow_elapsed >= yellow_dur:
                self.current_cycle += 1
                self.cycle_elapsed = 0
                self.in_yellow = False


# ─── 主入口 ──────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 全局字体
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)

    # 全局样式
    app.setStyleSheet("""
        QMainWindow { background: #f0f2f5; }
        QWidget { font-family: "Microsoft YaHei", "SimHei", sans-serif; }
        QToolTip { background: #1f2937; color: white; border: none; border-radius: 4px; padding: 4px 8px; }
        QScrollBar:vertical {
            background: #f3f4f6; width: 8px; border-radius: 4px;
        }
        QScrollBar::handle:vertical {
            background: #d1d5db; border-radius: 4px; min-height: 30px;
        }
        QScrollBar::handle:vertical:hover { background: #9ca3af; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
