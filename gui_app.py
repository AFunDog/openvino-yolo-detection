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
from collections import deque
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2

from algorithm import VAController, FrameFeatures, load_frames

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QPushButton, QLabel, QLineEdit, QTextEdit,
    QFileDialog, QSlider, QComboBox, QListWidget, QListWidgetItem,
    QProgressBar, QTableWidget, QTableWidgetItem, QSplitter,
    QGroupBox, QSizePolicy, QAbstractItemView, QHeaderView,
    QMenu, QMessageBox, QStyledItemDelegate, QStyle, QStyleOptionViewItem,
)
from PyQt6.QtCore import Qt, QTimer, QSize, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QImage, QPixmap,
    QLinearGradient,
)

import theme_manager as tm

# ─── HiDPI ─────────────────────────────────────────────
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

# ─── 路径 ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
TEST_OUTPUT_DIR = PROJECT_ROOT / "test" / "output"

VEHICLE_CLASSES = {"car", "van", "bus", "truck"}


# ─── 颜色（跟随系统主题） ─────────────────────────────────
def _init_colors(dark: bool):
    """根据 is_dark 重新设定所有 C_* 模块级全局变量。"""
    t = tm.DARK if dark else tm.LIGHT
    for key, val in t.items():
        globals()[f"C_{key.upper()}"] = val
    globals()["C_IS_DARK"] = dark


# 首次初始化（install 之前用 LIGHT 兜底，install 后会自动同步）
_init_colors(tm.is_dark())


# ─── 动态样式表 ────────────────────────────────────────
_dynamic_styles: dict = {}  # id(widget) -> (widget, style_func)


def _ds(widget, style_func):
    """注册控件样式表，主题切换时自动通过 style_func() 重新生成。"""
    _dynamic_styles[id(widget)] = (widget, style_func)
    widget.setStyleSheet(style_func())


def _refresh_all_styles():
    """主题切换后刷新所有已注册的动态样式表。"""
    for key, (widget, style_func) in list(_dynamic_styles.items()):
        try:
            widget.setStyleSheet(style_func())
        except RuntimeError:
            del _dynamic_styles[key]


# ─── 全局应用样式表（每次主题切换时重新生成）──────────────
def _make_app_stylesheet() -> str:
    """生成全局 QSS，覆盖大多数通用控件样式。"""
    return f"""
        QMainWindow {{ background: {C_BG_BASE.name()}; }}
        QWidget {{ font-family: "Microsoft YaHei", "SimHei", sans-serif; }}
        QToolTip {{
            background: {C_TOOLTIP_BG}; color: {C_TOOLTIP_TEXT};
            border: none; border-radius: 4px; padding: 4px 8px;
        }}
        QScrollBar:vertical {{
            background: {C_SCROLLBAR_BG}; width: 8px; border-radius: 4px;
        }}
        QScrollBar::handle:vertical {{
            background: {C_SCROLLBAR_HANDLE}; border-radius: 4px; min-height: 30px;
        }}
        QScrollBar::handle:vertical:hover {{ background: {C_SCROLLBAR_HANDLE_HOVER}; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

        QGroupBox#card {{
            background: {C_CARD_BG.name()};
            border: 1px solid {C_CARD_BORDER.name()};
            border-radius: 10px;
            margin-top: 12px;
            padding: 14px 12px;
            font-weight: bold;
            color: {C_TEXT_SECONDARY.name()};
        }}
        QGroupBox#card::title {{
            subcontrol-origin: margin; left: 12px; top: 2px; padding: 0 6px;
        }}

        QTableWidget {{
            border: none; border-radius: 6px;
            background: {C_TABLE_BG}; font-size: 12px; gridline-color: transparent;
        }}
        QTableWidget::item {{
            padding: 4px 8px; border-bottom: 1px solid {C_TABLE_BORDER};
        }}
        QTableWidget::item:alternate {{ background: {C_TABLE_ALT_BG}; }}
        QHeaderView::section {{
            background: {C_TABLE_HEADER_BG}; color: {C_TABLE_HEADER_TEXT};
            border: none; padding: 6px 8px; font-weight: bold; font-size: 11px;
        }}
        QTableWidget::item:selected {{ background: {C_TABLE_SELECTED_BG}; }}

        QListWidget {{
            border: 1px solid {C_SIDEBAR_BORDER}; border-radius: 6px;
            background: {C_SIDEBAR_ITEM_BG}; font-size: 11px; outline: none;
        }}
        QListWidget::item {{
            padding: 4px 8px; border-bottom: 1px solid {C_TABLE_BORDER};
        }}
        QListWidget::item:selected {{
            background: {C_SIDEBAR_ITEM_SELECTED_BG};
            color: {C_SIDEBAR_ITEM_SELECTED_TEXT};
        }}

        QTextEdit {{
            border: 1px solid {C_DETAIL_BORDER}; border-radius: 6px;
            background: {C_DETAIL_BG}; font-size: 11px; padding: 4px;
        }}

        QComboBox {{
            border: 1px solid {C_INPUT_BORDER}; border-radius: 6px;
            padding: 4px 8px; background: {C_INPUT_BG}; font-size: 11px;
        }}

        QMenu {{
            background: {C_MENU_BG}; border: 1px solid {C_MENU_BORDER};
            border-radius: 6px; padding: 4px;
        }}
        QMenu::item {{ padding: 6px 20px; border-radius: 4px; font-size: 12px; }}
        QMenu::item:selected {{ background: {C_MENU_DELETE_BG}; color: {C_MENU_DELETE_TEXT}; }}

        QMessageBox {{ background: {C_QMESSAGEBOX_BG}; color: {C_QMESSAGEBOX_TEXT}; }}

        QProgressBar {{ background: {C_PROGRESS_BG}; border: none; border-radius: 3px; }}

        QSlider::groove:horizontal {{
            background: {C_PROGRESS_BG}; height: 6px; border-radius: 3px;
        }}
        QSlider::handle:horizontal {{
            background: {C_PRIMARY.name()}; width: 14px; margin: -5px 0; border-radius: 7px;
        }}
    """


# ─── 数据加载 ────────────────────────────────────────────

def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ─── 圆角卡片 ────────────────────────────────────────────

class CardWidget(QGroupBox):
    """带圆角边框的卡片容器（样式由全局 QSS 控制，此处仅设 objectName）"""
    def __init__(self, title="", parent=None):
        super().__init__(title, parent)
        self.setObjectName("card")


# ─── 十字路口 Canvas ─────────────────────────────────────

class IntersectionCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.x_color = "off"
        self.y_color = "off"
        self.car_x = None
        self.car_y = None
        self.countdown = None
        self.setMinimumSize(350, 350)
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
        road_w = int(W * 0.24)
        lane_w = road_w / 4
        curb_w = max(4, int(W * 0.012))
        cx, cy = W / 2, H / 2

        # ── 草地背景 ──
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(C_GRASS))
        p.drawRect(QRectF(0, 0, W, H))

        # ── 人行道（四角） ──
        sw = max(10, int(W * 0.032))
        corners = [
            (0, 0, cx - road_w/2, cy - road_w/2),
            (cx + road_w/2, 0, W, cy - road_w/2),
            (0, cy + road_w/2, cx - road_w/2, H),
            (cx + road_w/2, cy + road_w/2, W, H),
        ]
        for x1, y1, x2, y2 in corners:
            p.setBrush(QBrush(C_SIDEWALK))
            p.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
            p.setBrush(QBrush(C_CURB))
            if x1 == 0:
                p.drawRect(QRectF(x2 - curb_w, y1, curb_w, y2 - y1))
            if x2 == W:
                p.drawRect(QRectF(x1, y1, curb_w, y2 - y1))
            if y1 == 0:
                p.drawRect(QRectF(x1, y2 - curb_w, x2 - x1, curb_w))
            if y2 == H:
                p.drawRect(QRectF(x1, y1, x2 - x1, curb_w))

        # ── 道路 ──
        p.setBrush(QBrush(C_ROAD))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRect(QRectF(0, cy - road_w/2, W, road_w))
        p.drawRect(QRectF(cx - road_w/2, 0, road_w, H))
        p.setBrush(QBrush(C_INTERSECTION))
        p.drawRect(QRectF(cx - road_w/2, cy - road_w/2, road_w, road_w))

        # ── 车道分隔线 ──
        pen_lane = QPen(C_LANE, max(1, int(W * 0.004)), Qt.PenStyle.DashLine)
        p.setPen(pen_lane)
        hw = lane_w
        for lo in [-hw, hw]:
            p.drawLine(int(0), int(cy + lo), int(cx - road_w/2), int(cy + lo))
            p.drawLine(int(cx + road_w/2), int(cy + lo), int(W), int(cy + lo))
        for lo in [-hw, hw]:
            p.drawLine(int(cx + lo), int(0), int(cx + lo), int(cy - road_w/2))
            p.drawLine(int(cx + lo), int(cy + road_w/2), int(cx + lo), int(H))

        # ── 中心线 ──
        pen_center = QPen(C_CENTER_LINE, max(1, int(W * 0.005)))
        p.setPen(pen_center)
        for offset in [-2, 2]:
            p.drawLine(int(0), int(cy + offset), int(cx - road_w/2), int(cy + offset))
            p.drawLine(int(cx + road_w/2), int(cy + offset), int(W), int(cy + offset))
            p.drawLine(int(cx + offset), int(0), int(cx + offset), int(cy - road_w/2))
            p.drawLine(int(cx + offset), int(cy + road_w/2), int(cx + offset), int(H))

        # ── 人行横道 ──
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(C_CROSSWALK))
        stripe_w = max(3, int(road_w * 0.06))
        gap = max(3, int(road_w * 0.04))
        cw_len = max(12, int(road_w * 0.3))
        yb = cy - road_w/2 - cw_len
        for i in range(int(road_w / (stripe_w + gap))):
            p.drawRect(QRectF(cx - road_w/2 + i * (stripe_w + gap), yb, stripe_w, cw_len))
        yb = cy + road_w/2
        for i in range(int(road_w / (stripe_w + gap))):
            p.drawRect(QRectF(cx - road_w/2 + i * (stripe_w + gap), yb, stripe_w, cw_len))
        xb = cx - road_w/2 - cw_len
        for i in range(int(road_w / (stripe_w + gap))):
            p.drawRect(QRectF(xb, cy - road_w/2 + i * (stripe_w + gap), cw_len, stripe_w))
        xb = cx + road_w/2
        for i in range(int(road_w / (stripe_w + gap))):
            p.drawRect(QRectF(xb, cy - road_w/2 + i * (stripe_w + gap), cw_len, stripe_w))

        # ── 停车线 ──
        p.setPen(QPen(C_STOP_LINE, max(2, int(W * 0.005))))
        p.drawLine(int(cx - road_w/2), int(cy - road_w/2 - 2), int(cx), int(cy - road_w/2 - 2))
        p.drawLine(int(cx), int(cy + road_w/2 + 2), int(cx + road_w/2), int(cy + road_w/2 + 2))
        p.drawLine(int(cx - road_w/2 - 2), int(cy), int(cx - road_w/2 - 2), int(cy + road_w/2))
        p.drawLine(int(cx + road_w/2 + 2), int(cy - road_w/2), int(cx + road_w/2 + 2), int(cy))

        # ── 转向箭头 ──
        self._draw_arrow(p, cx - lane_w * 1.5, cy - road_w/2 - cw_len - lane_w, "right", W)
        self._draw_arrow(p, cx + lane_w * 0.5, cy + road_w/2 + cw_len + lane_w, "right", W)
        self._draw_arrow(p, cx - road_w/2 - cw_len - lane_w, cy + lane_w * 0.5, "down", W)
        self._draw_arrow(p, cx + road_w/2 + cw_len + lane_w, cy - lane_w * 1.5, "down", W)

        # ── 交通灯 ──
        pole_h = max(28, int(H * 0.065))
        arm_len = max(14, int(W * 0.03))
        corner_off = max(8, int(W * 0.018))

        bx1 = cx - road_w/2 - corner_off
        by1 = cy - road_w/2 - corner_off
        self._draw_traffic_light(p, bx1, by1, self.x_color, pole_h, arm_len, "right", W)

        bx2 = cx + road_w/2 + corner_off
        by2 = cy + road_w/2 + corner_off
        self._draw_traffic_light(p, bx2, by2, self.x_color, pole_h, arm_len, "left", W)

        bx3 = cx + road_w/2 + corner_off
        by3 = cy - road_w/2 - corner_off
        self._draw_traffic_light(p, bx3, by3, self.y_color, pole_h, arm_len, "left", W)

        bx4 = cx - road_w/2 - corner_off
        by4 = cy + road_w/2 + corner_off
        self._draw_traffic_light(p, bx4, by4, self.y_color, pole_h, arm_len, "right", W)

        # ── 车辆图标 ──
        if self.car_x is not None:
            self._draw_vehicles(p, cx, cy, road_w, lane_w, W)

        # ── 倒计时 ──
        if self.countdown is not None:
            cfs = max(14, int(W * 0.032))
            p.setFont(QFont("Microsoft YaHei", cfs, QFont.Weight.Bold))
            p.setBrush(QBrush(C_CANVAS_COUNTDOWN_BG))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QRectF(cx - cfs * 1.1, cy - cfs * 1.1, cfs * 2.2, cfs * 2.2))
            p.setPen(QPen(C_CANVAS_COUNTDOWN_TEXT))
            p.drawText(QRectF(cx - cfs, cy - cfs, cfs * 2, cfs * 2), Qt.AlignmentFlag.AlignCenter,
                       str(math.ceil(self.countdown)))

        # ── X/Y 路标注（放在各自路段中间） ──
        label_fs = max(12, int(W * 0.03))
        label_font = QFont("Microsoft YaHei", label_fs, QFont.Weight.Bold)
        p.setFont(label_font)
        lbl_w = label_fs * 3.0
        lbl_h = label_fs * 1.6

        # X 路 — 左侧水平路段正中
        xr = QRectF(cx / 2 - road_w / 4 - lbl_w / 2, cy - lbl_h / 2, lbl_w, lbl_h)
        p.setBrush(QBrush(QColor(0, 0, 0, 90)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(xr, 4, 4)
        p.setPen(QPen(QColor(255, 255, 255, 220)))
        p.drawText(xr, Qt.AlignmentFlag.AlignCenter, "X 路")

        # Y 路 — 上方垂直路段正中
        yr = QRectF(cx - lbl_w / 2, cy / 2 - road_w / 4 - lbl_h / 2, lbl_w, lbl_h)
        p.drawRoundedRect(yr, 4, 4)
        p.drawText(yr, Qt.AlignmentFlag.AlignCenter, "Y 路")

        p.end()

    def _draw_arrow(self, p, x, y, direction, W):
        size = max(6, int(W * 0.02))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(C_ARROW_COLOR))
        if direction == "right":
            pts = [
                QPointF(x - size, y - size/2), QPointF(x + size/2, y - size/2),
                QPointF(x + size/2, y - size), QPointF(x + size, y),
                QPointF(x + size/2, y + size), QPointF(x + size/2, y + size/2),
                QPointF(x - size, y + size/2),
            ]
        elif direction == "down":
            pts = [
                QPointF(x - size/2, y - size), QPointF(x + size/2, y - size),
                QPointF(x + size/2, y + size/2), QPointF(x + size, y + size/2),
                QPointF(x, y + size), QPointF(x - size, y + size/2),
                QPointF(x - size/2, y + size/2),
            ]
        else:
            return
        p.drawPolygon(*pts)

    def _draw_traffic_light(self, p, bx, by, active_color, pole_h, arm_len, facing, W):
        bw = max(14, int(W * 0.03))
        bh = max(38, int(W * 0.075))
        r = max(4, int(W * 0.008))
        pole_w = max(3, int(W * 0.005))
        arm_h = max(2, int(W * 0.003))

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(C_POLE))
        pole_top = by - pole_h
        p.drawRect(QRectF(bx - pole_w/2, pole_top, pole_w, pole_h))

        if facing == "right":
            p.drawRect(QRectF(bx, pole_top - arm_h/2, arm_len, arm_h))
            self._draw_light_box(p, bx + arm_len, pole_top, bw, bh, r, active_color)
        else:
            p.drawRect(QRectF(bx - arm_len, pole_top - arm_h/2, arm_len, arm_h))
            self._draw_light_box(p, bx - arm_len, pole_top, bw, bh, r, active_color)

    def _draw_light_box(self, p, lx, ly, bw, bh, r, active_color):
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(C_LIGHT_BOX_OUTER))
        p.drawRoundedRect(QRectF(lx - bw/2 - 1, ly - bh/2 - 1, bw + 2, bh + 2), 4, 4)
        p.setBrush(QBrush(C_LIGHT_BOX_BODY))
        p.drawRoundedRect(QRectF(lx - bw/2, ly - bh/2, bw, bh), 3, 3)
        inner_m = max(2, int(bw * 0.12))
        p.setBrush(QBrush(C_LIGHT_BOX_INNER))
        p.drawRoundedRect(QRectF(lx - bw/2 + inner_m, ly - bh/2 + inner_m,
                                   bw - inner_m * 2, bh - inner_m * 2), 2, 2)

        for i, cn in enumerate(["red", "yellow", "green"]):
            by = ly - bh/3 + i * (bh/3)
            is_on = cn == active_color
            if cn == "red":
                fill = C_RED if is_on else C_RED_DIM
            elif cn == "yellow":
                fill = C_YELLOW if is_on else C_YELLOW_DIM
            else:
                fill = C_GREEN if is_on else C_GREEN_DIM
            if is_on:
                glow = QColor(fill)
                glow.setAlpha(40)
                p.setBrush(QBrush(glow))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(QRectF(lx - r - 8, by - r - 8, (r + 8)*2, (r + 8)*2))
                glow2 = QColor(fill)
                glow2.setAlpha(80)
                p.setBrush(QBrush(glow2))
                p.drawEllipse(QRectF(lx - r - 3, by - r - 3, (r + 3)*2, (r + 3)*2))
            p.setBrush(QBrush(fill))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QRectF(lx - r, by - r, r*2, r*2))

    def _draw_vehicles(self, p, cx, cy, road_w, lane_w, W):
        car_w = max(8, int(road_w * 0.15))
        car_h = max(5, int(road_w * 0.09))

        p.setPen(Qt.PenStyle.NoPen)
        for i in range(min(self.car_x, 5)):
            x_off = cx - road_w/2 - car_w * 2 - i * (car_w + 4)
            y_off = cy - lane_w * 0.5
            p.setBrush(QBrush(C_VEHICLE_BLUE))
            p.drawRoundedRect(QRectF(x_off, y_off - car_h/2, car_w, car_h), 2, 2)
            p.setBrush(QBrush(C_VEHICLE_WINDOW_BLUE))
            p.drawRect(QRectF(x_off + car_w * 0.55, y_off - car_h/2 + 1, car_w * 0.35, car_h - 2))

        p.setPen(Qt.PenStyle.NoPen)
        for i in range(min(self.car_x, 5)):
            x_off = cx + road_w/2 + car_w * 0.5 + i * (car_w + 4)
            y_off = cy + lane_w * 0.5
            p.setBrush(QBrush(C_VEHICLE_BLUE))
            p.drawRoundedRect(QRectF(x_off, y_off - car_h/2, car_w, car_h), 2, 2)
            p.setBrush(QBrush(C_VEHICLE_WINDOW_BLUE))
            p.drawRect(QRectF(x_off + car_w * 0.1, y_off - car_h/2 + 1, car_w * 0.35, car_h - 2))

        p.setPen(Qt.PenStyle.NoPen)
        for i in range(min(self.car_y, 5)):
            x_off = cx + lane_w * 0.5
            y_off = cy - road_w/2 - car_w * 2 - i * (car_w + 4)
            p.setBrush(QBrush(C_VEHICLE_ORANGE))
            p.drawRoundedRect(QRectF(x_off - car_h/2, y_off, car_h, car_w), 2, 2)
            p.setBrush(QBrush(C_VEHICLE_WINDOW_ORANGE))
            p.drawRect(QRectF(x_off - car_h/2 + 1, y_off + car_w * 0.55, car_h - 2, car_w * 0.35))

        p.setPen(Qt.PenStyle.NoPen)
        for i in range(min(self.car_y, 5)):
            x_off = cx - lane_w * 0.5
            y_off = cy + road_w/2 + car_w * 0.5 + i * (car_w + 4)
            p.setBrush(QBrush(C_VEHICLE_ORANGE))
            p.drawRoundedRect(QRectF(x_off - car_h/2, y_off, car_h, car_w), 2, 2)
            p.setBrush(QBrush(C_VEHICLE_WINDOW_ORANGE))
            p.drawRect(QRectF(x_off - car_h/2 + 1, y_off + car_w * 0.1, car_h - 2, car_w * 0.35))


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
            p.fillRect(self.rect(), C_VIDEO_BG)
            p.setPen(QPen(C_VIDEO_PLACEHOLDER_TEXT))
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
            self.setStyleSheet(f"""
                QPushButton {{
                    background: {C_NAV_ACTIVE_BG}; color: {C_NAV_ACTIVE_TEXT};
                    border: none; border-radius: 8px; padding: 6px 14px;
                    text-align: left; font-size: 13px; font-weight: bold;
                }}
                QPushButton:hover {{ background: {C_NAV_ACTIVE_HOVER_BG}; }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: {C_NAV_INACTIVE_BG}; color: {C_NAV_INACTIVE_TEXT};
                    border: none; border-radius: 8px; padding: 6px 14px;
                    text-align: left; font-size: 13px;
                }}
                QPushButton:hover {{ background: {C_NAV_HOVER_BG}; }}
            """)


# ─── 进度条表格委托 ──────────────────────────────────────

class ProgressBarDelegate(QStyledItemDelegate):
    """在表格单元格内绘制迷你进度条 + 百分比文字"""

    def paint(self, painter, option, index):
        text = index.data(Qt.ItemDataRole.DisplayRole) or ""
        pct = 0.0
        try:
            pct = float(str(text).replace("%", "").strip())
        except ValueError:
            pass

        painter.save()
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, QColor(C_TABLE_SELECTED_BG))
        elif index.row() % 2 == 1:
            painter.fillRect(option.rect, QColor(C_TABLE_ALT_BG))

        r = option.rect.adjusted(6, 4, -6, -4)
        bar_w = int(r.width() * pct / 100)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(C_PROGRESS_BG)))
        painter.drawRoundedRect(r, 3, 3)

        if bar_w > 0:
            fill_rect = QRectF(r.x(), r.y(), bar_w, r.height())
            grad = QLinearGradient(fill_rect.topLeft(), fill_rect.topRight())
            grad.setColorAt(0, C_PRIMARY)
            grad.setColorAt(1, C_PRIMARY_HOVER)
            painter.setBrush(QBrush(grad))
            painter.drawRoundedRect(fill_rect, 3, 3)

        painter.setPen(QPen(C_TEXT_PRIMARY))
        painter.setFont(QFont("Microsoft YaHei", 9))
        painter.drawText(option.rect, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()


# ─── 统计数字标签 ────────────────────────────────────────

class StatLabel(QWidget):
    def __init__(self, value="0", label="", color_key="primary", parent=None):
        super().__init__(parent)
        self._color_key = color_key
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.val_label = QLabel(value)
        self.val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label = QLabel(label)
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.val_label)
        layout.addWidget(self.name_label)
        self._refresh_style()

    def _refresh_style(self):
        c = globals().get(f"C_{self._color_key.upper()}", C_PRIMARY)
        self.val_label.setStyleSheet(
            f"color: {c.name()}; font-size: 20px; font-weight: bold;"
        )
        self.name_label.setStyleSheet(
            f"color: {C_TEXT_MUTED.name()}; font-size: 11px;"
        )

    def set_value(self, v):
        self.val_label.setText(str(v))


# ─── 交通灯指示器 ────────────────────────────────────────

class TrafficLightIndicator(QWidget):
    def __init__(self, label="X 方向", color_key="blue", parent=None):
        super().__init__(parent)
        self._color_key = color_key  # 存储在主题词典里的 key，延迟解析
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._lbl = QLabel(label)
        layout.addWidget(self._lbl)
        self._dots = {}  # name -> QLabel
        for name in ("red", "yellow", "green"):
            dot = QLabel("●")
            dot.setStyleSheet(f"font-size: 18px;")
            layout.addWidget(dot)
            self._dots[name] = dot
        self._refresh_style()

    def _refresh_style(self):
        c = globals().get(f"C_{self._color_key.upper()}", C_BLUE)
        self._lbl.setStyleSheet(f"color: {c.name()}; font-weight: bold; font-size: 12px;")
        dims = {"red": C_RED_DIM, "yellow": C_YELLOW_DIM, "green": C_GREEN_DIM}
        for name, dot in self._dots.items():
            dot.setStyleSheet(f"color: {dims[name].name()}; font-size: 18px;")

    def set_active(self, color_name):
        on_map = {"red": C_RED, "yellow": C_YELLOW, "green": C_GREEN}
        dim_map = {"red": C_RED_DIM, "yellow": C_YELLOW_DIM, "green": C_GREEN_DIM}
        for name, dot in self._dots.items():
            c = on_map[name] if name == color_name else dim_map[name]
            dot.setStyleSheet(f"color: {c.name()}; font-size: 18px;")


# ─── 主窗口 ──────────────────────────────────────────────

class MainWindow(QMainWindow):
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
            "line_count_total": 0,
            "line_count_in": 0,
            "line_count_out": 0,
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

    # ── UI 构建 ──────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        _ds(central, lambda: f"background: {C_BG_BASE.name()};")

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # ── 左侧导航栏 ──
        sidebar = QWidget()
        sidebar.setFixedWidth(200)
        _ds(sidebar, lambda: f"background: {C_SIDEBAR_BG}; border-right: 1px solid {C_SIDEBAR_BORDER};")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 12, 8, 8)
        sidebar_layout.setSpacing(8)

        self.nav_yolo = NavButton("  YOLO 视频分析")
        self.nav_yolo.active = True
        self.nav_traffic = NavButton("  实时联动")
        sidebar_layout.addWidget(self.nav_yolo)
        sidebar_layout.addWidget(self.nav_traffic)

        sidebar_layout.addSpacing(8)
        line = QWidget()
        line.setFixedHeight(1)
        _ds(line, lambda: f"background: {C_SIDEBAR_BORDER};")
        sidebar_layout.addWidget(line)
        sidebar_layout.addSpacing(8)

        lbl = QLabel("检测记录")
        _ds(lbl, lambda: f"color: {C_TEXT_SECONDARY.name()}; font-size: 12px; font-weight: bold;")
        sidebar_layout.addWidget(lbl)

        self.session_list = QListWidget()
        self.session_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.session_list.customContextMenuRequested.connect(self._on_session_context_menu)
        sidebar_layout.addWidget(self.session_list, 1)

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
        _ds(status_bar, lambda: f"background: {C_STATUS_BG}; border-top: 1px solid {C_STATUS_BORDER};")
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(12, 0, 12, 0)
        self.status_label = QLabel("就绪")
        _ds(self.status_label, lambda: f"color: {C_TEXT_MUTED.name()}; font-size: 11px;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        brand = QLabel("OpenVINO + YOLOv26")
        _ds(brand, lambda: f"color: {C_TEXT_MUTED.name()}; font-size: 11px;")
        status_layout.addWidget(brand)
        main_layout.addWidget(status_bar)

    def _build_yolo_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 8)
        layout.setSpacing(8)

        card_upload = CardWidget("YOLOv26 视频检测")
        upload_layout = QVBoxLayout(card_upload)
        upload_layout.setSpacing(6)
        row = QVBoxLayout()
        row.setSpacing(6)

        x_row = QHBoxLayout()
        x_lbl = QLabel("X方向视频")
        _ds(x_lbl, lambda: f"color: {C_TEXT_SECONDARY.name()}; font-size: 12px;")
        x_row.addWidget(x_lbl)
        self.video_path_x_input = QLineEdit()
        self.video_path_x_input.setPlaceholderText("上传同一路口 X 方向监控视频...")
        _ds(self.video_path_x_input, lambda: f"""
            QLineEdit {{
                border: 1px solid {C_INPUT_BORDER}; border-radius: 6px;
                padding: 6px 10px; background: {C_INPUT_BG}; font-size: 12px;
            }}
            QLineEdit:focus {{ border-color: {C_INPUT_FOCUS_BORDER}; background: {C_INPUT_FOCUS_BG}; }}
        """)
        x_row.addWidget(self.video_path_x_input, 1)
        btn_browse_x = QPushButton("浏览X")
        btn_browse_x.setFixedSize(70, 32)
        _ds(btn_browse_x, lambda: MainWindow._btn_style(C_PRIMARY))
        btn_browse_x.clicked.connect(lambda: self._on_browse_video("x"))
        x_row.addWidget(btn_browse_x)
        row.addLayout(x_row)

        y_row = QHBoxLayout()
        y_lbl = QLabel("Y方向视频")
        _ds(y_lbl, lambda: f"color: {C_TEXT_SECONDARY.name()}; font-size: 12px;")
        y_row.addWidget(y_lbl)
        self.video_path_y_input = QLineEdit()
        self.video_path_y_input.setPlaceholderText("上传同一路口 Y 方向监控视频...")
        _ds(self.video_path_y_input, lambda: f"""
            QLineEdit {{
                border: 1px solid {C_INPUT_BORDER}; border-radius: 6px;
                padding: 6px 10px; background: {C_INPUT_BG}; font-size: 12px;
            }}
            QLineEdit:focus {{ border-color: {C_INPUT_FOCUS_BORDER}; background: {C_INPUT_FOCUS_BG}; }}
        """)
        y_row.addWidget(self.video_path_y_input, 1)
        btn_browse_y = QPushButton("浏览Y")
        btn_browse_y.setFixedSize(70, 32)
        _ds(btn_browse_y, lambda: MainWindow._btn_style(C_PRIMARY))
        btn_browse_y.clicked.connect(lambda: self._on_browse_video("y"))
        y_row.addWidget(btn_browse_y)

        self.btn_detect = QPushButton("开始检测")
        self.btn_detect.setFixedSize(90, 32)
        _ds(self.btn_detect, lambda: MainWindow._btn_style(C_GREEN))
        self.btn_detect.clicked.connect(self._on_start_detect)
        y_row.addWidget(self.btn_detect)
        row.addLayout(y_row)

        upload_layout.addLayout(row)

        row2 = QHBoxLayout()
        self.detect_status = QLabel("就绪")
        _ds(self.detect_status, lambda: f"color: {C_TEXT_SECONDARY.name()}; font-size: 12px;")
        row2.addWidget(self.detect_status)
        row2.addStretch()
        self.btn_play = QPushButton("播放")
        self.btn_play.setFixedSize(70, 28)
        _ds(self.btn_play, lambda: MainWindow._btn_style(C_PRIMARY, 10))
        self.btn_play.clicked.connect(self._on_play_video)
        row2.addWidget(self.btn_play)
        upload_layout.addLayout(row2)
        layout.addWidget(card_upload)

        bottom = QHBoxLayout()
        bottom.setSpacing(8)

        left_col = QVBoxLayout()
        left_col.setSpacing(8)

        card_stats = CardWidget()
        stats_layout = QHBoxLayout(card_stats)
        stats_layout.setSpacing(16)
        self.stat_frames = StatLabel("0", "帧数", "blue")
        self.stat_detections = StatLabel("0", "检测数", "primary")
        self.stat_vehicles = StatLabel("0", "总过线", "green")
        self.stat_x_count = StatLabel("0", "X过线", "blue")
        self.stat_y_count = StatLabel("0", "Y过线", "orange")
        self.stat_fps = StatLabel("0", "FPS", "orange")
        stats_layout.addWidget(self.stat_frames)
        stats_layout.addWidget(self.stat_detections)
        stats_layout.addWidget(self.stat_vehicles)
        stats_layout.addWidget(self.stat_x_count)
        stats_layout.addWidget(self.stat_y_count)
        stats_layout.addWidget(self.stat_fps)
        left_col.addWidget(card_stats)

        card_detail = CardWidget("检测详情")
        detail_layout = QVBoxLayout(card_detail)
        self.session_detail = QTextEdit()
        self.session_detail.setReadOnly(True)
        self.session_detail.setFixedHeight(70)
        self.session_detail.setPlaceholderText("从左侧选择一条检测记录")
        detail_layout.addWidget(self.session_detail)
        left_col.addWidget(card_detail)

        card_class = CardWidget("类别统计")
        class_layout = QVBoxLayout(card_class)
        self.class_table = QTableWidget(0, 3)
        self.class_table.setHorizontalHeaderLabels(["类别", "数量", "占比"])
        self.class_table.horizontalHeader().setStretchLastSection(True)
        self.class_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.class_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.class_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.class_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.class_table.verticalHeader().setVisible(False)
        self.class_table.verticalHeader().setDefaultSectionSize(32)
        self.class_table.setAlternatingRowColors(True)
        self.class_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        class_layout.setContentsMargins(0, 0, 0, 0)
        class_layout.setSpacing(0)
        class_layout.addWidget(self.class_table)
        left_col.addWidget(card_class, 1)

        left_w = QWidget()
        left_w.setLayout(left_col)
        left_w.setMinimumWidth(320)
        bottom.addWidget(left_w, 2)

        right_col = QVBoxLayout()
        preview_row = QHBoxLayout()
        preview_row.setSpacing(8)

        card_video_x = CardWidget("X方向实时画面")
        video_layout_x = QVBoxLayout(card_video_x)
        self.video_preview_x = VideoPreviewWidget()
        video_layout_x.addWidget(self.video_preview_x)
        preview_row.addWidget(card_video_x, 1)

        card_video_y = CardWidget("Y方向实时画面")
        video_layout_y = QVBoxLayout(card_video_y)
        self.video_preview_y = VideoPreviewWidget()
        video_layout_y.addWidget(self.video_preview_y)
        preview_row.addWidget(card_video_y, 1)

        right_col.addLayout(preview_row, 3)

        card_traffic = CardWidget("实时交通灯联动")
        traffic_layout = QHBoxLayout(card_traffic)
        traffic_layout.setSpacing(8)

        traffic_left = QVBoxLayout()
        self.canvas = IntersectionCanvas()
        traffic_left.addWidget(self.canvas, 1)
        traffic_layout.addLayout(traffic_left, 3)

        traffic_right = QVBoxLayout()
        traffic_right.setSpacing(8)

        self.xl_indicator = TrafficLightIndicator("X 方向", "blue")
        self.yl_indicator = TrafficLightIndicator("Y 方向", "orange")
        traffic_right.addWidget(self.xl_indicator)
        traffic_right.addWidget(self.yl_indicator)

        phase_row = QHBoxLayout()
        phase_lbl = QLabel("阶段:")
        _ds(phase_lbl, lambda: f"color: {C_TEXT_MUTED.name()}; font-size: 12px;")
        phase_row.addWidget(phase_lbl)
        self.phase_label = QLabel("--")
        _ds(self.phase_label, lambda: f"color: {C_TEXT_PRIMARY.name()}; font-size: 12px; font-weight: bold;")
        phase_row.addWidget(self.phase_label)
        phase_row.addStretch()
        traffic_right.addLayout(phase_row)

        timer_row = QHBoxLayout()
        timer_lbl = QLabel("倒计时")
        _ds(timer_lbl, lambda: f"color: {C_TEXT_MUTED.name()}; font-size: 12px;")
        timer_row.addWidget(timer_lbl)
        self.timer_text = QLabel("--")
        _ds(self.timer_text, lambda: f"color: {C_GREEN.name()}; font-size: 14px; font-weight: bold;")
        timer_row.addWidget(self.timer_text)
        timer_row.addStretch()
        traffic_right.addLayout(timer_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        _ds(self.progress_bar, lambda: f"""
            QProgressBar {{ background: {C_PROGRESS_BG}; border: none; border-radius: 3px; }}
            QProgressBar::chunk {{ background: {C_PROGRESS_CHUNK}; border-radius: 3px; }}
        """)
        traffic_right.addWidget(self.progress_bar)

        btn_row = QHBoxLayout()
        self.btn_start_sim = QPushButton("▶ 启动联动")
        self.btn_start_sim.setFixedSize(96, 32)
        _ds(self.btn_start_sim, lambda: MainWindow._btn_style(C_GREEN))
        self.btn_start_sim.clicked.connect(self._on_start_sim)
        btn_row.addWidget(self.btn_start_sim)

        self.btn_pause_sim = QPushButton("⏸ 暂停")
        self.btn_pause_sim.setFixedSize(90, 32)
        _ds(self.btn_pause_sim, lambda: MainWindow._btn_style(C_YELLOW))
        self.btn_pause_sim.clicked.connect(self._on_pause_sim)
        btn_row.addWidget(self.btn_pause_sim)

        self.btn_reset_sim = QPushButton("■ 重置")
        self.btn_reset_sim.setFixedSize(90, 32)
        _ds(self.btn_reset_sim, lambda: MainWindow._btn_style(C_RED))
        self.btn_reset_sim.clicked.connect(self._on_reset_sim)
        btn_row.addWidget(self.btn_reset_sim)
        traffic_right.addLayout(btn_row)

        speed_lbl = QLabel("速度")
        _ds(speed_lbl, lambda: f"color: {C_TEXT_MUTED.name()}; font-size: 12px;")
        traffic_right.addWidget(speed_lbl)
        speed_row = QHBoxLayout()
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 20)
        self.speed_slider.setValue(5)
        self.speed_slider.valueChanged.connect(lambda v: setattr(self, 'sim_speed', float(v)))
        speed_row.addWidget(self.speed_slider)
        self.speed_val = QLabel("5x")
        _ds(self.speed_val, lambda: f"color: {C_TEXT_PRIMARY.name()}; font-size: 12px; font-weight: bold;")
        self.speed_slider.valueChanged.connect(lambda v: self.speed_val.setText(f"{v}x"))
        speed_row.addWidget(self.speed_val)
        traffic_right.addLayout(speed_row)

        ds_lbl = QLabel("数据源")
        _ds(ds_lbl, lambda: f"color: {C_TEXT_MUTED.name()}; font-size: 12px;")
        traffic_right.addWidget(ds_lbl)
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItem("(默认)")
        self.data_source_combo.addItem("实时检测")
        traffic_right.addWidget(self.data_source_combo)

        ci_lbl = QLabel("感应控制状态")
        _ds(ci_lbl, lambda: f"color: {C_TEXT_MUTED.name()}; font-size: 12px;")
        traffic_right.addWidget(ci_lbl)
        self.cycle_info = QTextEdit()
        self.cycle_info.setReadOnly(True)
        self.cycle_info.setFixedHeight(88)
        self.cycle_info.setPlaceholderText("导入 X/Y 视频后点击开始检测")
        traffic_right.addWidget(self.cycle_info)

        hist_lbl = QLabel("切换记录")
        _ds(hist_lbl, lambda: f"color: {C_TEXT_MUTED.name()}; font-size: 12px;")
        traffic_right.addWidget(hist_lbl)
        self.history_table = QTableWidget(0, 3)
        self.history_table.setHorizontalHeaderLabels(["相位", "时长", "原因"])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.history_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.history_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.verticalHeader().setDefaultSectionSize(28)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setFixedHeight(160)
        traffic_right.addWidget(self.history_table, 1)

        traffic_layout.addLayout(traffic_right, 2)
        right_col.addWidget(card_traffic, 2)

        self.video_preview = self.video_preview_x

        right_w = QWidget()
        right_w.setLayout(right_col)
        bottom.addWidget(right_w, 3)

        layout.addLayout(bottom, 1)
        self.stack.addWidget(page)

    def _build_traffic_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        card = CardWidget("页面说明")
        card_layout = QVBoxLayout(card)
        tip = QLabel("交通灯仿真已经合并到“YOLO 视频分析”页面。\n导入 X/Y 两个方向视频后，检测与信号灯联动会在同一页实时展示。")
        tip.setWordWrap(True)
        _ds(tip, lambda: f"color: {C_TEXT_SECONDARY.name()}; font-size: 13px;")
        card_layout.addWidget(tip)
        layout.addWidget(card)
        layout.addStretch()

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
        self.nav_traffic.clicked.connect(lambda: self._switch_page(0))
        self.session_list.currentRowChanged.connect(self._on_session_select)
        self.data_source_combo.currentTextChanged.connect(self._on_data_source_changed)

    def _switch_page(self, idx):
        self.stack.setCurrentIndex(idx)
        self.nav_yolo.active = (idx == 0)
        self.nav_traffic.active = (idx == 1)

    # ── YOLO 检测 ────────────────────────────────────────

    def _on_browse_video(self, direction):
        path, _ = QFileDialog.getOpenFileName(
            self, f"选择{direction.upper()}方向监控视频", "",
            "视频文件 (*.mp4 *.avi);;所有文件 (*.*)"
        )
        if not path:
            return
        if direction.lower() == "x":
            self.video_path_x_input.setText(path)
        else:
            self.video_path_y_input.setText(path)

    def _on_start_detect(self):
        video_x = self.video_path_x_input.text().strip()
        video_y = self.video_path_y_input.text().strip()
        if not video_x or not video_y:
            self.detect_status.setText("错误: 请同时上传 X / Y 两个方向的视频")
            self.detect_status.setStyleSheet(f"color: {C_RED.name()}; font-size: 12px;")
            return
        if os.path.normcase(video_x) == os.path.normcase(video_y):
            self.detect_status.setText("错误: X / Y 方向不能使用同一个视频")
            self.detect_status.setStyleSheet(f"color: {C_RED.name()}; font-size: 12px;")
            return

        video_pairs = [("X", video_x), ("Y", video_y)]
        missing = [p for _, p in video_pairs if not os.path.exists(p)]
        if missing:
            self.detect_status.setText(f"错误: 视频路径不存在 {Path(missing[0]).name}")
            self.detect_status.setStyleSheet(f"color: {C_RED.name()}; font-size: 12px;")
            return
        if self.detecting:
            return

        model_onnx_path = PROJECT_ROOT / "public" / "yolo-v26" / "yolo26n.onnx"
        model_xml_path = PROJECT_ROOT / "public" / "yolo-v26" / "ir_model" / "yolo26n.xml"
        model_bin_path = PROJECT_ROOT / "public" / "yolo-v26" / "ir_model" / "yolo26n.bin"
        has_onnx = model_onnx_path.exists()
        has_openvino_ir = model_xml_path.exists() and model_bin_path.exists()
        if not (has_onnx or has_openvino_ir):
            self.detect_status.setText("错误: YOLOv26 模型文件不存在（需 ONNX 或 OpenVINO IR）")
            self.detect_status.setStyleSheet(f"color: {C_RED.name()}; font-size: 12px;")
            return

        os.makedirs(str(TEST_OUTPUT_DIR), exist_ok=True)

        self.detecting = True
        self.detect_progress = None
        self.detect_status.setText("检测中... 0/2")
        self.detect_status.setStyleSheet(f"color: {C_PRIMARY.name()}; font-size: 12px;")
        self.btn_detect.setEnabled(False)

        self._detect_latest = {"bgr": None, "fps": 0.0, "count": 0, "idx": 0, "video_fps": None}
        self._detect_dirty = False
        self._start_live_sim()
        self._reset_live_dual_state(video_pairs)
        self.sim_running = True
        self.sim_paused = False
        self.last_tick = time.time()
        self.data_source_combo.blockSignals(True)
        self.data_source_combo.setCurrentText("实时检测")
        self.data_source_combo.blockSignals(False)

        def on_detect_frame(direction, frame_bgr, frame_idx, avg_fps, num_objects, video_fps, line_counts, backend):
            direction = str(direction).upper()
            with self._live_dual_lock:
                state = self._live_dual_state.setdefault(direction, self._new_live_direction_state())
                state["frame_bgr"] = frame_bgr
                state["frame_idx"] = frame_idx
                state["fps"] = avg_fps
                state["num_objects"] = num_objects
                state["line_count_total"] = int((line_counts or {}).get("line_count_total", 0))
                state["line_count_in"] = int((line_counts or {}).get("line_count_in", 0))
                state["line_count_out"] = int((line_counts or {}).get("line_count_out", 0))
                state["video_fps"] = float(video_fps or state.get("video_fps") or 30.0)
                state["backend"] = backend
            self._live_dual_dirty = True

        def run_detect():
            outputs = []
            streams = {}
            try:
                import main as yolo
                cwd = os.getcwd()
                os.chdir(str(PROJECT_ROOT))
                try:
                    detector = yolo.load_detector()
                    backend = detector["backend"]

                    for direction, video_path in video_pairs:
                        cap = cv2.VideoCapture(video_path)
                        if not cap.isOpened():
                            raise FileNotFoundError(f"无法打开视频源: {video_path}")
                        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                        basename = Path(video_path).stem
                        output_path = str(TEST_OUTPUT_DIR / f"output_{direction}_{basename}.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(output_path, fourcc, max(1.0, fps), (width, height))
                        streams[direction] = {
                            "source_path": video_path,
                            "output_path": output_path,
                            "cap": cap,
                            "writer": writer,
                            "fps": max(1.0, fps),
                            "width": width,
                            "height": height,
                            "frame_count": 0,
                            "fps_list": [],
                            "total_detections": 0,
                            "tracker": yolo.SimpleTracker(),
                            "line_counter": yolo.LineCounter(),
                            "last_boxes": [],
                            "last_confidences": [],
                            "last_class_ids": [],
                            "done": False,
                            "next_due": 0.0,
                            "backend": backend,
                        }
                        with self._live_dual_lock:
                            state = self._live_dual_state.setdefault(direction, self._new_live_direction_state())
                            state["source"] = video_path
                            state["video_fps"] = max(1.0, fps)
                            state["backend"] = backend

                    start_ts = time.time()
                    while True:
                        any_pending = False
                        clock = time.time() - start_ts
                        for direction, stream in streams.items():
                            if stream["done"]:
                                continue
                            any_pending = True
                            if clock + 1e-6 < stream["next_due"]:
                                continue

                            ret, frame = stream["cap"].read()
                            if not ret:
                                stream["done"] = True
                                with self._live_dual_lock:
                                    self._live_dual_state[direction]["done"] = True
                                self._live_dual_dirty = True
                                continue

                            stream["next_due"] += 1.0 / max(stream["fps"], 1.0)
                            frame_no = stream["frame_count"]
                            step_start = time.time()

                            is_real_detection = (frame_no % yolo.SKIP_FRAMES == 0)
                            if is_real_detection:
                                boxes, confidences, class_ids = yolo.process_frame(frame, detector)
                                stream["last_boxes"] = boxes
                                stream["last_confidences"] = confidences
                                stream["last_class_ids"] = class_ids
                            else:
                                boxes = stream["last_boxes"]
                                confidences = stream["last_confidences"]
                                class_ids = stream["last_class_ids"]

                            track_ids = stream["tracker"].update(boxes, class_ids)
                            stream["line_counter"].update(
                                boxes, class_ids, track_ids, stream["width"], stream["height"]
                            )
                            line_info = stream["line_counter"].get_line_info(stream["width"], stream["height"])

                            if boxes:
                                result_frame = yolo.draw_detections(
                                    frame.copy(), boxes, confidences, class_ids, yolo.CLASS_NAMES, track_ids=track_ids
                                )
                            else:
                                result_frame = frame.copy()
                            result_frame = yolo.draw_counting_line(
                                result_frame,
                                line_info,
                                stream["line_counter"].total_count,
                                stream["line_counter"].count_in,
                                stream["line_counter"].count_out,
                            )
                            cv2.putText(
                                result_frame,
                                f"{direction} FPS: {0.0:.1f}",
                                (10, 30),
                                yolo.DISPLAY_LABEL_FONT,
                                0.8,
                                (0, 0, 255),
                                2,
                            )
                            cv2.putText(
                                result_frame,
                                f"Objects: {len(boxes)}",
                                (10, 60),
                                yolo.DISPLAY_LABEL_FONT,
                                0.8,
                                (0, 0, 255),
                                2,
                            )

                            elapsed = time.time() - step_start
                            avg_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                            stream["fps_list"].append(avg_fps)
                            if len(stream["fps_list"]) > 30:
                                stream["fps_list"].pop(0)
                            avg_fps = sum(stream["fps_list"]) / len(stream["fps_list"])
                            cv2.putText(
                                result_frame,
                                f"{direction} FPS: {avg_fps:.1f}",
                                (10, 30),
                                yolo.DISPLAY_LABEL_FONT,
                                0.8,
                                (0, 0, 255),
                                2,
                            )

                            if stream["writer"]:
                                stream["writer"].write(result_frame)

                            stream["frame_count"] += 1
                            stream["total_detections"] += len(boxes)
                            on_detect_frame(
                                direction,
                                result_frame,
                                stream["frame_count"],
                                avg_fps,
                                len(boxes),
                                stream["fps"],
                                {
                                    "line_count_total": stream["line_counter"].total_count,
                                    "line_count_in": stream["line_counter"].count_in,
                                    "line_count_out": stream["line_counter"].count_out,
                                },
                                backend,
                            )

                        if not any_pending:
                            break
                        time.sleep(0.001)

                    for direction, stream in streams.items():
                        final_avg_fps = sum(stream["fps_list"]) / len(stream["fps_list"]) if stream["fps_list"] else 0.0
                        summary = {
                            "source": stream["source_path"],
                            "total_frames": stream["frame_count"],
                            "avg_fps": round(final_avg_fps, 1),
                            "total_detections": stream["total_detections"],
                            "count_method": "line_crossing",
                            "line_count_total": stream["line_counter"].total_count,
                            "line_count_in": stream["line_counter"].count_in,
                            "line_count_out": stream["line_counter"].count_out,
                            "crossed_class_counts": stream["line_counter"].crossed_class_counts,
                            "video_info": {
                                "width": stream["width"],
                                "height": stream["height"],
                                "fps": stream["fps"],
                            },
                            "model": detector["model_path"],
                            "backend": stream["backend"],
                        }
                        outputs.append({
                            "direction": direction,
                            "source_path": stream["source_path"],
                            "output_path": stream["output_path"],
                            "session_dir": None,
                            "summary": summary,
                        })
                finally:
                    os.chdir(cwd)
                pair_session = self._save_direction_pair_session(outputs)
                self.detect_progress = {"status": "done", "outputs": outputs, "pair_session": pair_session}
            except Exception as e:
                import traceback
                self.detect_progress = {
                    "status": "fail",
                    "message": f"{str(e)[:200]}\n{traceback.format_exc()[:300]}"
                }
            finally:
                for stream in streams.values():
                    try:
                        if stream.get("cap"):
                            stream["cap"].release()
                    except Exception:
                        pass
                    try:
                        if stream.get("writer"):
                            stream["writer"].release()
                    except Exception:
                        pass
                self.detecting = False

        threading.Thread(target=run_detect, daemon=True).start()

    def _check_detect_status(self):
        prog = self.detect_progress
        if self.detecting and isinstance(prog, dict) and prog.get("status") == "running":
            self.detect_status.setText(
                f"检测中... {prog.get('direction', '?')}方向 {prog.get('current', 0)}/{prog.get('total', 0)}"
            )
            self.detect_status.setStyleSheet(f"color: {C_PRIMARY.name()}; font-size: 12px;")
            self.detect_progress = None
            return

        if not self.detecting and prog:
            self.detect_progress = None
            self.btn_detect.setEnabled(True)
            self._detect_latest = {"bgr": None, "fps": 0.0, "count": 0, "idx": 0, "video_fps": None}
            self._detect_dirty = False
            pair_session = None
            if isinstance(prog, dict) and prog.get("status") == "fail":
                self.detect_status.setText(prog.get("message", "检测失败"))
                self.detect_status.setStyleSheet(f"color: {C_RED.name()}; font-size: 12px;")
                self.sim_running = False
                self._sim_live_mode = False
            else:
                outputs = prog.get("outputs", []) if isinstance(prog, dict) else []
                pair_session = prog.get("pair_session") if isinstance(prog, dict) else None
                backend_names = []
                for item in outputs:
                    summary = item.get("summary", {}) or {}
                    direction = str(item.get("direction", "")).upper() or "?"
                    backend_names.append(f"{direction}:{self._backend_label(summary.get('backend'))}")
                backend_text = f"  后端: {' / '.join(backend_names)}" if backend_names else ""
                self.detect_status.setText(f"检测完成，已生成 X/Y 双方向统计{backend_text}")
                self.detect_status.setStyleSheet(f"color: {C_GREEN.name()}; font-size: 12px;")
                last_output = outputs[-1]["output_path"] if outputs else None
                if last_output and self._load_video(last_output):
                    self.btn_play.setText("⏸ 暂停")
                    self.detect_status.setText(f"播放: {Path(last_output).name}{backend_text}")
                    self.detect_status.setStyleSheet(f"color: {C_GREEN.name()}; font-size: 12px;")
                else:
                    self.detect_status.setText(f"检测完成，但结果视频无法播放{backend_text}")
                    self.detect_status.setStyleSheet(f"color: {C_ORANGE.name()}; font-size: 12px;")
                self.sim_running = False
                self._sim_live_mode = False
            # 标出检测结束但不停止仿真
            if self.cycle_info.toPlainText():
                self.cycle_info.append("\n检测结束，实时联动停止")
            self._load_sessions()
            if pair_session:
                self.status_label.setText(f"已生成双方向会话: {Path(pair_session).name}")

    def _save_direction_pair_session(self, outputs):
        """将 X/Y 两个方向的视频检测结果聚合为同一路口的一条会话记录。"""
        by_direction = {}
        for item in outputs:
            direction = str(item.get("direction", "")).upper()
            summary = item.get("summary", {}) or {}
            by_direction[direction] = {
                "source": item.get("source_path"),
                "session_dir": item.get("session_dir"),
                "output_path": item.get("output_path"),
                "line_count_total": summary.get("line_count_total", 0),
                "line_count_in": summary.get("line_count_in", 0),
                "line_count_out": summary.get("line_count_out", 0),
                "crossed_class_counts": summary.get("crossed_class_counts", {}),
                "total_frames": summary.get("total_frames", 0),
                "total_detections": summary.get("total_detections", 0),
                "avg_fps": summary.get("avg_fps", 0),
                "video_info": summary.get("video_info", {}),
                "backend": summary.get("backend", ""),
            }

        if "X" not in by_direction or "Y" not in by_direction:
            raise ValueError("双方向会话生成失败：缺少 X 或 Y 方向结果")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        x_name = Path(by_direction["X"]["source"]).stem
        y_name = Path(by_direction["Y"]["source"]).stem
        session_name = f"detection_pair_{timestamp}_{x_name[:16]}_{y_name[:16]}"
        session_dir = DATA_DIR / session_name
        session_dir.mkdir(parents=True, exist_ok=True)

        line_count_x = int(by_direction["X"]["line_count_total"])
        line_count_y = int(by_direction["Y"]["line_count_total"])
        summary = {
            "session_type": "direction_pair",
            "source": "same_intersection_xy_pair",
            "description": "同一路口两段垂直方向监控视频的聚合过线统计",
            "count_method": "line_crossing_by_direction",
            "line_count_x": line_count_x,
            "line_count_y": line_count_y,
            "line_count_total": line_count_x + line_count_y,
            "direction_videos": by_direction,
            "preview_output": by_direction["Y"].get("output_path") or by_direction["X"].get("output_path"),
        }
        with open(session_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return str(session_dir)

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
        if self._live_dual_dirty:
            with self._live_dual_lock:
                live_state = {
                    direction: dict(values)
                    for direction, values in self._live_dual_state.items()
                }
            x_state = live_state.get("X", {})
            y_state = live_state.get("Y", {})
            self._set_preview_frame(self.video_preview_x, x_state.get("frame_bgr"))
            self._set_preview_frame(self.video_preview_y, y_state.get("frame_bgr"))

            x_count = int(x_state.get("line_count_total", 0))
            y_count = int(y_state.get("line_count_total", 0))
            total_count = x_count + y_count
            total_frames = int(x_state.get("frame_idx", 0)) + int(y_state.get("frame_idx", 0))
            total_objects = int(x_state.get("num_objects", 0)) + int(y_state.get("num_objects", 0))
            avg_fps = (float(x_state.get("fps", 0.0)) + float(y_state.get("fps", 0.0))) / 2.0

            self.stat_frames.set_value(str(total_frames))
            self.stat_detections.set_value(str(total_objects))
            self.stat_vehicles.set_value(str(total_count))
            self.stat_x_count.set_value(str(x_count))
            self.stat_y_count.set_value(str(y_count))
            self.stat_fps.set_value(f"{avg_fps:.1f}")
            self.session_detail.setText(
                f"实时双视频联动\n"
                f"X视频: {Path(str(x_state.get('source', 'X'))).name}  后端: {self._backend_label(x_state.get('backend'))}\n"
                f"Y视频: {Path(str(y_state.get('source', 'Y'))).name}  后端: {self._backend_label(y_state.get('backend'))}\n"
                f"X过线:{x_count}  Y过线:{y_count}  总过线:{total_count}"
            )
            if self.detecting:
                self.detect_status.setText(
                    f"实时检测中... X过线:{x_count}  Y过线:{y_count}  当前目标:{total_objects}"
                )
                self.detect_status.setStyleSheet(f"color: {C_PRIMARY.name()}; font-size: 12px;")
            self._live_dual_dirty = False
            return

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
        if hasattr(self, "video_preview_x"):
            self.video_preview_x.clear()
        if hasattr(self, "video_preview_y"):
            self.video_preview_y.clear()
        self.btn_play.setText("▶ 播放")

    # ── 检测记录 ─────────────────────────────────────────

    def _load_sessions(self):
        self.sessions = []
        self.session_list.clear()
        if not DATA_DIR.exists():
            return
        for d in sorted(DATA_DIR.iterdir(), reverse=True):
            if d.is_dir() and (d.name.startswith("detection_") or d.name.startswith("detection_pair_")):
                summary = load_json(d / "summary.json")
                self.sessions.append({"name": d.name, "path": str(d), "summary": summary})
                self.session_list.addItem(d.name)
        self.data_source_combo.blockSignals(True)
        self.data_source_combo.clear()
        self.data_source_combo.addItem("(默认)")
        self.data_source_combo.addItem("实时检测")
        for s in self.sessions:
            self.data_source_combo.addItem(s["name"])
        self.data_source_combo.blockSignals(False)

    def _on_session_context_menu(self, pos):
        item = self.session_list.itemAt(pos)
        if item is None:
            return
        row = self.session_list.row(item)
        if row < 0 or row >= len(self.sessions):
            return
        menu = QMenu()
        delete_action = menu.addAction("🗑  删除记录")
        action = menu.exec(self.session_list.mapToGlobal(pos))
        if action == delete_action:
            self._delete_session(row)

    def _delete_session(self, row):
        if row < 0 or row >= len(self.sessions):
            return
        s = self.sessions[row]
        session_path = s.get("path", "")
        name = s.get("name", "")
        reply = QMessageBox.question(
            self, "删除确认",
            f"确定要删除检测记录吗？\n\n{name}\n\n此操作不可撤销。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        import shutil
        if os.path.isdir(session_path):
            shutil.rmtree(session_path)
        self._load_sessions()

    def _on_session_select(self, row):
        if row < 0 or row >= len(self.sessions):
            return
        s = self.sessions[row]
        self.selected_session = s
        summary = s.get("summary", {})

        if summary.get("session_type") == "direction_pair":
            self._render_direction_pair_session(summary)
            return

        classes = summary.get("class_counts", {})
        total = summary.get("total_detections", 0)
        frames = summary.get("total_frames", 0)
        fps_val = summary.get("avg_fps", 0)
        vi = summary.get("video_info", {})
        crossed_counts = summary.get("crossed_class_counts")
        unique_counts = crossed_counts or summary.get("unique_class_counts", classes)
        vehicle_count = summary.get("line_count_total")
        if vehicle_count is None:
            vehicle_count = summary.get(
                "unique_vehicle_count",
                sum(v for k, v in unique_counts.items() if k.lower() in VEHICLE_CLASSES)
            )
        count_in = summary.get("line_count_in", 0)
        count_out = summary.get("line_count_out", 0)
        backend_label = self._backend_label(summary.get("backend"))

        self.stat_frames.set_value(str(frames))
        self.stat_detections.set_value(str(total))
        self.stat_vehicles.set_value(str(vehicle_count))
        self.stat_x_count.set_value(str(vehicle_count))
        self.stat_y_count.set_value("--")
        self.stat_fps.set_value(str(round(fps_val, 1)))

        self.class_table.setRowCount(0)
        unique_total = sum(unique_counts.values())
        for cls, count in sorted(unique_counts.items(), key=lambda x: -x[1]):
            pct_val = (count / unique_total * 100) if unique_total else 0
            pct = f"{pct_val:.1f}%"
            row_idx = self.class_table.rowCount()
            self.class_table.insertRow(row_idx)

            name_item = QTableWidgetItem(cls)
            name_item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            self.class_table.setItem(row_idx, 0, name_item)

            count_item = QTableWidgetItem(str(count))
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignCenter)
            self.class_table.setItem(row_idx, 1, count_item)

            pct_item = QTableWidgetItem(pct)
            pct_item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
            if pct_val >= 50:
                pct_item.setForeground(QBrush(QColor(22, 163, 74)))
            elif pct_val >= 20:
                pct_item.setForeground(QBrush(QColor(37, 99, 235)))
            elif pct_val >= 5:
                pct_item.setForeground(QBrush(QColor(202, 138, 4)))
            else:
                pct_item.setForeground(QBrush(QColor(107, 114, 128)))
            self.class_table.setItem(row_idx, 2, pct_item)

        info = (
            f"视频源: {summary.get('source', 'N/A')}\n"
            f"推理后端: {backend_label}\n"
            f"分辨率: {vi.get('width', '?')}x{vi.get('height', '?')}\n"
            f"总帧数: {frames}  总检测: {total}\n"
            f"过线总数: {vehicle_count}  进线:{count_in}  出线:{count_out}  FPS: {fps_val:.1f}"
        )
        self.session_detail.setText(info)

    def _render_direction_pair_session(self, summary):
        direction_videos = summary.get("direction_videos", {})
        x_info = direction_videos.get("X", {})
        y_info = direction_videos.get("Y", {})
        x_count = int(summary.get("line_count_x", x_info.get("line_count_total", 0)))
        y_count = int(summary.get("line_count_y", y_info.get("line_count_total", 0)))
        total_count = int(summary.get("line_count_total", x_count + y_count))
        total_frames = int(x_info.get("total_frames", 0)) + int(y_info.get("total_frames", 0))
        total_detections = int(x_info.get("total_detections", 0)) + int(y_info.get("total_detections", 0))
        x_backend = self._backend_label(x_info.get("backend"))
        y_backend = self._backend_label(y_info.get("backend"))

        self.stat_frames.set_value(str(total_frames))
        self.stat_detections.set_value(str(total_detections))
        self.stat_vehicles.set_value(str(total_count))
        self.stat_x_count.set_value(str(x_count))
        self.stat_y_count.set_value(str(y_count))
        self.stat_fps.set_value("--")

        self.class_table.setRowCount(0)
        rows = [("X方向过线", x_count), ("Y方向过线", y_count)]
        for label, count in rows:
            pct_val = (count / total_count * 100) if total_count else 0.0
            row_idx = self.class_table.rowCount()
            self.class_table.insertRow(row_idx)

            name_item = QTableWidgetItem(label)
            name_item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            self.class_table.setItem(row_idx, 0, name_item)

            count_item = QTableWidgetItem(str(count))
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignCenter)
            self.class_table.setItem(row_idx, 1, count_item)

            pct_item = QTableWidgetItem(f"{pct_val:.1f}%")
            pct_item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
            self.class_table.setItem(row_idx, 2, pct_item)

        info = (
            f"类型: 同一路口双方向视频\n"
            f"X视频: {Path(str(x_info.get('source', 'N/A'))).name}\n"
            f"Y视频: {Path(str(y_info.get('source', 'N/A'))).name}\n"
            f"推理后端: X={x_backend}  Y={y_backend}\n"
            f"X方向车辆数: {x_count}  Y方向车辆数: {y_count}\n"
            f"总过线数: {total_count}\n"
            f"可直接作为交通灯仿真的 X / Y 决策输入"
        )
        self.session_detail.setText(info)

    # ── 交通灯仿真 (Vehicle-Actuated) ────────────────────

    def _on_data_source_changed(self, text):
        """数据源切换：选"实时检测"时自动启停仿真。"""
        if text == "实时检测":
            self._start_live_sim()
        elif self._sim_live_mode:
            self._stop_live_sim()

    def _start_live_sim(self):
        """武装实时模式——等待视频检测开始后才真正启动仿真。"""
        self._sim_live_mode = True
        self.va_controller.reset()
        self.va_features = None
        self.va_frame_index = 0
        self.va_sim_time = 0.0
        with self._live_frames_lock:
            self._live_frames.clear()
        self._reset_live_dual_state()
        self.btn_start_sim.setEnabled(True)
        self.btn_pause_sim.setEnabled(True)
        self.speed_slider.setEnabled(False)
        self.speed_val.setText("1x（实时）")
        self.cycle_info.setPlaceholderText("等待双视频实时检测开始...")

    def _stop_live_sim(self):
        """停止实时仿真并重置 UI。"""
        self._sim_live_mode = False
        self.sim_running = False
        self.sim_paused = False
        self.x_light = "off"
        self.y_light = "off"
        with self._live_frames_lock:
            self._live_frames.clear()
        self.canvas.update_state()
        self.timer_text.setText("--")
        self.progress_bar.setValue(0)
        self.cycle_info.setText("已停止实时仿真")
        self.cycle_info.setPlaceholderText("选择数据源后点击 ▶ 开始")
        self.history_table.setRowCount(0)
        self.speed_slider.setEnabled(True)
        self.speed_val.setText(f"{int(self.sim_speed)}x")
        self.btn_start_sim.setEnabled(True)
        self.btn_pause_sim.setEnabled(True)
        # 回退 combo 到默认，避免 UI 不一致
        self.data_source_combo.blockSignals(True)
        self.data_source_combo.setCurrentIndex(0)
        self.data_source_combo.blockSignals(False)

    def _start_va_sim(self):
        """初始化离线回放仿真。"""
        sel = self.data_source_combo.currentText()
        self._sim_live_mode = False
        self.va_pair_summary = None
        if sel and sel != "(默认)" and sel != "实时检测":
            session_dir = os.path.join(DATA_DIR, sel)
            summary = load_json(Path(session_dir) / "summary.json")
            if summary.get("session_type") == "direction_pair":
                self.va_frames = []
                self.va_fps = 1.0
                self.va_pair_summary = summary
            else:
                self.va_frames, self.va_fps = load_frames(session_dir)
        else:
            self.va_frames = []
            self.va_fps = 30.0

        self.va_controller.reset()
        self.feature_extractor.reset()
        self.va_frame_index = 0
        self.va_sim_time = 0.0
        self.va_pair_wait_x = 0.0
        self.va_pair_wait_y = 0.0

    def _on_start_sim(self):
        # 实时模式：只处理暂停恢复
        if self._sim_live_mode:
            if self.sim_paused:
                self.sim_paused = False
                self.last_tick = time.time()
            return
        if self.sim_running and not self.sim_paused:
            return
        if not self.sim_running:
            self._start_va_sim()
            self.sim_running = True
            self.sim_paused = False
            self.last_tick = time.time()
        else:
            self.sim_paused = False
            self.last_tick = time.time()

    def _on_pause_sim(self):
        self.sim_paused = True

    def _on_reset_sim(self):
        if self._sim_live_mode:
            self._stop_live_sim()
            return
        self.sim_running = False
        self.sim_paused = False
        self.x_light = "off"
        self.y_light = "off"
        self.canvas.update_state()
        self.timer_text.setText("--")
        self.progress_bar.setValue(0)
        self.cycle_info.setText("点击 ▶ 开始模拟")
        self.history_table.setRowCount(0)
        self.va_controller.reset()
        self.feature_extractor.reset()
        self.va_pair_summary = None
        self.va_pair_wait_x = 0.0
        self.va_pair_wait_y = 0.0

    def _build_live_dual_features(self, dt):
        with self._live_dual_lock:
            x_state = dict(self._live_dual_state.get("X", {}))
            y_state = dict(self._live_dual_state.get("Y", {}))

        queue_x = int(x_state.get("num_objects", 0))
        queue_y = int(y_state.get("num_objects", 0))
        total_x = int(x_state.get("line_count_total", 0))
        total_y = int(y_state.get("line_count_total", 0))
        delta_x = max(0, total_x - self._live_dual_last_totals["X"])
        delta_y = max(0, total_y - self._live_dual_last_totals["Y"])
        self._live_dual_last_totals["X"] = total_x
        self._live_dual_last_totals["Y"] = total_y

        if queue_x > 0 or delta_x > 0:
            self._live_dual_gap_x = 0.0
        else:
            self._live_dual_gap_x += dt
        if queue_y > 0 or delta_y > 0:
            self._live_dual_gap_y = 0.0
        else:
            self._live_dual_gap_y += dt

        phase = self.va_controller.get_state().get("phase", "X")
        if phase == "X":
            self._live_dual_wait_x = max(0.0, self._live_dual_wait_x - dt * max(queue_x, 1) * 0.4)
            self._live_dual_wait_y += dt * (queue_y + delta_y)
        else:
            self._live_dual_wait_y = max(0.0, self._live_dual_wait_y - dt * max(queue_y, 1) * 0.4)
            self._live_dual_wait_x += dt * (queue_x + delta_x)

        return {
            "queue_x": queue_x,
            "queue_y": queue_y,
            "wait_x": self._live_dual_wait_x,
            "wait_y": self._live_dual_wait_y,
            "gap_x": self._live_dual_gap_x,
            "gap_y": self._live_dual_gap_y,
            "arrival_x": float(delta_x) / max(dt, 1e-3),
            "arrival_y": float(delta_y) / max(dt, 1e-3),
            "line_count_x": total_x,
            "line_count_y": total_y,
            "calibrated": True,
        }

    def _build_pair_features(self, dt):
        """基于双方向视频聚合计数构造控制器输入。"""
        summary = self.va_pair_summary or {}
        queue_x = int(summary.get("line_count_x", 0))
        queue_y = int(summary.get("line_count_y", 0))

        phase = self.va_controller.get_state().get("phase", "X")
        if phase == "X":
            self.va_pair_wait_x = max(0.0, self.va_pair_wait_x - dt * max(queue_x, 1) * 0.5)
            self.va_pair_wait_y += dt * queue_y
        else:
            self.va_pair_wait_y = max(0.0, self.va_pair_wait_y - dt * max(queue_y, 1) * 0.5)
            self.va_pair_wait_x += dt * queue_x

        return {
            "queue_x": queue_x,
            "queue_y": queue_y,
            "wait_x": self.va_pair_wait_x,
            "wait_y": self.va_pair_wait_y,
            "gap_x": 0.0 if queue_x > 0 else self.va_controller.gap_seconds,
            "gap_y": 0.0 if queue_y > 0 else self.va_controller.gap_seconds,
            "arrival_x": float(queue_x),
            "arrival_y": float(queue_y),
            "calibrated": True,
        }

    def _sim_tick(self):
        if not self.sim_running or self.sim_paused:
            return

        # ── 实时检测模式 ──
        if self._sim_live_mode:
            now = time.time()
            dt = now - self.last_tick
            self.last_tick = now
            self.va_features = self._build_live_dual_features(dt)
            feats = self.va_features

            x_light, y_light, countdown = self.va_controller.step(
                feats["queue_x"], feats["queue_y"],
                feats["wait_x"], feats["wait_y"],
                feats["gap_x"], feats["gap_y"],
                dt,
            )
            self._update_sim_ui(x_light, y_light, countdown)
            return

        # ── 离线回放模式 ──
        now = time.time()
        dt = (now - self.last_tick) * self.sim_speed
        self.last_tick = now

        self.va_sim_time += dt

        if self.va_pair_summary is not None:
            self.va_features = self._build_pair_features(dt)
            feats = self.va_features
            x_light, y_light, countdown = self.va_controller.step(
                feats["queue_x"], feats["queue_y"],
                feats["wait_x"], feats["wait_y"],
                feats["gap_x"], feats["gap_y"],
                dt,
            )
            self._update_sim_ui(x_light, y_light, countdown)
            return

        n_frames = max(1, int(dt * max(self.va_fps, 1.0)))
        for _ in range(min(n_frames, 10)):
            frame_idx = int(self.va_sim_time * self.va_fps)
            if self.va_frames:
                frame_idx = frame_idx % len(self.va_frames)
                frame_data = self.va_frames[frame_idx]
            else:
                frame_data = {"frame": frame_idx, "detections": []}
            self.va_features = self.feature_extractor.process_frame(frame_data)

        feats = self.va_features or {
            "queue_x": 0, "queue_y": 0,
            "wait_x": 0.0, "wait_y": 0.0,
            "gap_x": 0.0, "gap_y": 0.0,
            "arrival_x": 0.0, "arrival_y": 0.0,
        }

        x_light, y_light, countdown = self.va_controller.step(
            feats["queue_x"], feats["queue_y"],
            feats["wait_x"], feats["wait_y"],
            feats["gap_x"], feats["gap_y"],
            dt,
        )
        self._update_sim_ui(x_light, y_light, countdown)

    def _update_sim_ui(self, x_light, y_light, countdown):
        """刷新仿真 UI 控件"""
        state = self.va_controller.get_state()
        feats = self.va_features or {
            "queue_x": 0, "queue_y": 0,
            "wait_x": 0.0, "wait_y": 0.0,
            "gap_x": 0.0, "gap_y": 0.0,
            "arrival_x": 0.0, "arrival_y": 0.0,
        }
        is_yellow = state["in_yellow"]

        self.x_light = x_light
        self.y_light = y_light

        if countdown is not None:
            self.timer_text.setText(f"{math.ceil(countdown)}s")

        if is_yellow:
            yd = self.va_controller.yellow_duration
            self.progress_bar.setValue(
                int(state["yellow_elapsed"] / yd * 100) if yd > 0 else 0
            )
            self.progress_bar.setStyleSheet(f"""
                QProgressBar {{ background: {C_PROGRESS_BG}; border: none; border-radius: 3px; }}
                QProgressBar::chunk {{ background: {C_PROGRESS_CHUNK_YELLOW}; border-radius: 3px; }}
            """)
        else:
            m = self.va_controller.max_green
            self.progress_bar.setValue(
                int(state["phase_elapsed"] / m * 100) if m > 0 else 0
            )
            self.progress_bar.setStyleSheet(f"""
                QProgressBar {{ background: {C_PROGRESS_BG}; border: none; border-radius: 3px; }}
                QProgressBar::chunk {{ background: {C_PROGRESS_CHUNK}; border-radius: 3px; }}
            """)

        self.canvas.update_state(
            x_light, y_light, feats["queue_x"], feats["queue_y"], countdown
        )

        self.xl_indicator.set_active(x_light)
        self.yl_indicator.set_active(y_light)

        if is_yellow:
            self.phase_label.setText("黄灯过渡")
            self.phase_label.setStyleSheet(
                f"color: {C_YELLOW.name()}; font-size: 12px; font-weight: bold;"
            )
        elif state["phase"] == "X":
            self.phase_label.setText("X路绿灯 / Y路红灯")
            self.phase_label.setStyleSheet(
                f"color: {C_BLUE.name()}; font-size: 12px; font-weight: bold;"
            )
        else:
            self.phase_label.setText("Y路绿灯 / X路红灯")
            self.phase_label.setStyleSheet(
                f"color: {C_ORANGE.name()}; font-size: 12px; font-weight: bold;"
            )

        mode_tag = "实时" if self._sim_live_mode else ("双视频" if self.va_pair_summary is not None else "VA")
        if self._sim_live_mode:
            self.cycle_info.setText(
                f"[{mode_tag}] 周期 #{state['cycle_num'] + 1}\n"
                f"X过线:{feats.get('line_count_x', 0)}  当前目标:{feats['queue_x']}  等待:{feats['wait_x']:.0f}s\n"
                f"Y过线:{feats.get('line_count_y', 0)}  当前目标:{feats['queue_y']}  等待:{feats['wait_y']:.0f}s\n"
                f"X到达:{feats['arrival_x']:.1f}/s  Y到达:{feats['arrival_y']:.1f}/s  绿灯已过:{state['phase_elapsed']:.1f}s"
            )
        else:
            self.cycle_info.setText(
                f"[{mode_tag}] 周期 #{state['cycle_num'] + 1}\n"
                f"X数量:{feats['queue_x']}  等待:{feats['wait_x']:.0f}s  "
                f"清空:{feats['gap_x']:.1f}s  到达:{feats['arrival_x']:.1f}/s\n"
                f"Y数量:{feats['queue_y']}  等待:{feats['wait_y']:.0f}s  "
                f"清空:{feats['gap_y']:.1f}s  到达:{feats['arrival_y']:.1f}/s\n"
                f"绿灯已过:{state['phase_elapsed']:.1f}s  "
                f"标定:{'✓' if feats.get('calibrated') else '…'}"
            )

        history = self.va_controller.get_history()
        if history:
            self.history_table.setRowCount(0)
            for rec in reversed(history[-20:]):
                r = self.history_table.rowCount()
                self.history_table.insertRow(0)
                self.history_table.setItem(0, 0, QTableWidgetItem(rec.phase))
                self.history_table.setItem(0, 1, QTableWidgetItem(f"{rec.duration:.1f}s"))
                self.history_table.setItem(0, 2, QTableWidgetItem(rec.reason))

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
