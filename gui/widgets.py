"""Reusable Qt widgets for the traffic-light GUI."""

from __future__ import annotations

import math

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QSize, QRectF, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QImage, QLinearGradient
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QSizePolicy, QStyledItemDelegate, QStyle,
)

import gui.theme as gui_theme

__all__ = [
    "CardWidget",
    "IntersectionCanvas",
    "VideoPreviewWidget",
    "NavButton",
    "ProgressBarDelegate",
    "StatLabel",
    "TrafficLightIndicator",
]


class CardWidget(QGroupBox):
    """带圆角边框的卡片容器（样式由全局 QSS 控制，此处仅设 objectName）"""

    def __init__(self, title="", parent=None):
        super().__init__(title, parent)
        self.setObjectName("card")


class IntersectionCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.x_color = "off"
        self.y_color = "off"
        self.car_x = None
        self.car_y = None
        self.countdown = None
        self.setMinimumSize(220, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self):
        return QSize(360, 360)

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
        p.fillRect(self.rect(), gui_theme.C_CARD_BG)

        side = min(self.width(), self.height())
        offset_x = (self.width() - side) / 2
        offset_y = (self.height() - side) / 2
        p.translate(offset_x, offset_y)

        W = side
        H = side
        road_w = int(W * 0.24)
        lane_w = road_w / 4
        curb_w = max(4, int(W * 0.012))
        cx, cy = W / 2, H / 2

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(gui_theme.C_GRASS))
        p.drawRect(QRectF(0, 0, W, H))

        sw = max(10, int(W * 0.032))
        corners = [
            (0, 0, cx - road_w / 2, cy - road_w / 2),
            (cx + road_w / 2, 0, W, cy - road_w / 2),
            (0, cy + road_w / 2, cx - road_w / 2, H),
            (cx + road_w / 2, cy + road_w / 2, W, H),
        ]
        for x1, y1, x2, y2 in corners:
            p.setBrush(QBrush(gui_theme.C_SIDEWALK))
            p.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
            p.setBrush(QBrush(gui_theme.C_CURB))
            if x1 == 0:
                p.drawRect(QRectF(x2 - curb_w, y1, curb_w, y2 - y1))
            if x2 == W:
                p.drawRect(QRectF(x1, y1, curb_w, y2 - y1))
            if y1 == 0:
                p.drawRect(QRectF(x1, y2 - curb_w, x2 - x1, curb_w))
            if y2 == H:
                p.drawRect(QRectF(x1, y1, x2 - x1, curb_w))

        p.setBrush(QBrush(gui_theme.C_ROAD))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRect(QRectF(0, cy - road_w / 2, W, road_w))
        p.drawRect(QRectF(cx - road_w / 2, 0, road_w, H))
        p.setBrush(QBrush(gui_theme.C_INTERSECTION))
        p.drawRect(QRectF(cx - road_w / 2, cy - road_w / 2, road_w, road_w))

        pen_lane = QPen(gui_theme.C_LANE, max(1, int(W * 0.004)), Qt.PenStyle.DashLine)
        p.setPen(pen_lane)
        hw = lane_w
        for lo in [-hw, hw]:
            p.drawLine(int(0), int(cy + lo), int(cx - road_w / 2), int(cy + lo))
            p.drawLine(int(cx + road_w / 2), int(cy + lo), int(W), int(cy + lo))
        for lo in [-hw, hw]:
            p.drawLine(int(cx + lo), int(0), int(cx + lo), int(cy - road_w / 2))
            p.drawLine(int(cx + lo), int(cy + road_w / 2), int(cx + lo), int(H))

        pen_center = QPen(gui_theme.C_CENTER_LINE, max(1, int(W * 0.005)))
        p.setPen(pen_center)
        for offset in [-2, 2]:
            p.drawLine(int(0), int(cy + offset), int(cx - road_w / 2), int(cy + offset))
            p.drawLine(int(cx + road_w / 2), int(cy + offset), int(W), int(cy + offset))
            p.drawLine(int(cx + offset), int(0), int(cx + offset), int(cy - road_w / 2))
            p.drawLine(int(cx + offset), int(cy + road_w / 2), int(cx + offset), int(H))

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(gui_theme.C_CROSSWALK))
        stripe_w = max(3, int(road_w * 0.06))
        gap = max(3, int(road_w * 0.04))
        cw_len = max(12, int(road_w * 0.3))
        yb = cy - road_w / 2 - cw_len
        for i in range(int(road_w / (stripe_w + gap))):
            p.drawRect(QRectF(cx - road_w / 2 + i * (stripe_w + gap), yb, stripe_w, cw_len))
        yb = cy + road_w / 2
        for i in range(int(road_w / (stripe_w + gap))):
            p.drawRect(QRectF(cx - road_w / 2 + i * (stripe_w + gap), yb, stripe_w, cw_len))
        xb = cx - road_w / 2 - cw_len
        for i in range(int(road_w / (stripe_w + gap))):
            p.drawRect(QRectF(xb, cy - road_w / 2 + i * (stripe_w + gap), cw_len, stripe_w))
        xb = cx + road_w / 2
        for i in range(int(road_w / (stripe_w + gap))):
            p.drawRect(QRectF(xb, cy - road_w / 2 + i * (stripe_w + gap), cw_len, stripe_w))

        p.setPen(QPen(gui_theme.C_STOP_LINE, max(2, int(W * 0.005))))
        p.drawLine(int(cx - road_w / 2), int(cy - road_w / 2 - 2), int(cx), int(cy - road_w / 2 - 2))
        p.drawLine(int(cx), int(cy + road_w / 2 + 2), int(cx + road_w / 2), int(cy + road_w / 2 + 2))
        p.drawLine(int(cx - road_w / 2 - 2), int(cy), int(cx - road_w / 2 - 2), int(cy + road_w / 2))
        p.drawLine(int(cx + road_w / 2 + 2), int(cy - road_w / 2), int(cx + road_w / 2 + 2), int(cy))

        self._draw_arrow(p, cx - lane_w * 1.5, cy - road_w / 2 - cw_len - lane_w, "right", W)
        self._draw_arrow(p, cx + lane_w * 0.5, cy + road_w / 2 + cw_len + lane_w, "right", W)
        self._draw_arrow(p, cx - road_w / 2 - cw_len - lane_w, cy + lane_w * 0.5, "down", W)
        self._draw_arrow(p, cx + road_w / 2 + cw_len + lane_w, cy - lane_w * 1.5, "down", W)

        pole_h = max(28, int(H * 0.065))
        arm_len = max(14, int(W * 0.03))
        corner_off = max(8, int(W * 0.018))

        bx1 = cx - road_w / 2 - corner_off
        by1 = cy - road_w / 2 - corner_off
        self._draw_traffic_light(p, bx1, by1, self.x_color, pole_h, arm_len, "right", W)

        bx2 = cx + road_w / 2 + corner_off
        by2 = cy + road_w / 2 + corner_off
        self._draw_traffic_light(p, bx2, by2, self.x_color, pole_h, arm_len, "left", W)

        bx3 = cx + road_w / 2 + corner_off
        by3 = cy - road_w / 2 - corner_off
        self._draw_traffic_light(p, bx3, by3, self.y_color, pole_h, arm_len, "left", W)

        bx4 = cx - road_w / 2 - corner_off
        by4 = cy + road_w / 2 + corner_off
        self._draw_traffic_light(p, bx4, by4, self.y_color, pole_h, arm_len, "right", W)

        if self.car_x is not None:
            self._draw_vehicles(p, cx, cy, road_w, lane_w, W)

        if self.countdown is not None:
            cfs = max(14, int(W * 0.032))
            p.setFont(QFont("Microsoft YaHei", cfs, QFont.Weight.Bold))
            p.setBrush(QBrush(gui_theme.C_CANVAS_COUNTDOWN_BG))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QRectF(cx - cfs * 1.1, cy - cfs * 1.1, cfs * 2.2, cfs * 2.2))
            p.setPen(QPen(gui_theme.C_CANVAS_COUNTDOWN_TEXT))
            p.drawText(QRectF(cx - cfs, cy - cfs, cfs * 2, cfs * 2), Qt.AlignmentFlag.AlignCenter, str(math.ceil(self.countdown)))

        label_fs = max(12, int(W * 0.03))
        label_font = QFont("Microsoft YaHei", label_fs, QFont.Weight.Bold)
        p.setFont(label_font)
        lbl_w = label_fs * 3.0
        lbl_h = label_fs * 1.6

        # 根据交通灯状态设置路名标签颜色
        if self.x_color == "green":
            x_label_color = gui_theme.C_GREEN
            x_bg_color = QColor(gui_theme.C_GREEN.red(), gui_theme.C_GREEN.green(), gui_theme.C_GREEN.blue(), 120)
        elif self.x_color == "red":
            x_label_color = gui_theme.C_RED
            x_bg_color = QColor(gui_theme.C_RED.red(), gui_theme.C_RED.green(), gui_theme.C_RED.blue(), 120)
        else:
            x_label_color = QColor(255, 255, 255, 220)
            x_bg_color = QColor(0, 0, 0, 90)

        if self.y_color == "green":
            y_label_color = gui_theme.C_GREEN
            y_bg_color = QColor(gui_theme.C_GREEN.red(), gui_theme.C_GREEN.green(), gui_theme.C_GREEN.blue(), 120)
        elif self.y_color == "red":
            y_label_color = gui_theme.C_RED
            y_bg_color = QColor(gui_theme.C_RED.red(), gui_theme.C_RED.green(), gui_theme.C_RED.blue(), 120)
        else:
            y_label_color = QColor(255, 255, 255, 220)
            y_bg_color = QColor(0, 0, 0, 90)

        xr = QRectF(cx / 2 - road_w / 4 - lbl_w / 2, cy - lbl_h / 2, lbl_w, lbl_h)
        p.setBrush(QBrush(x_bg_color))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(xr, 4, 4)
        p.setPen(QPen(x_label_color))
        p.drawText(xr, Qt.AlignmentFlag.AlignCenter, "X 路")

        yr = QRectF(cx - lbl_w / 2, cy / 2 - road_w / 4 - lbl_h / 2, lbl_w, lbl_h)
        p.setBrush(QBrush(y_bg_color))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(yr, 4, 4)
        p.setPen(QPen(y_label_color))
        p.drawText(yr, Qt.AlignmentFlag.AlignCenter, "Y 路")

        p.end()

    def _draw_arrow(self, p, x, y, direction, W):
        size = max(6, int(W * 0.02))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(gui_theme.C_ARROW_COLOR))
        if direction == "right":
            pts = [
                QPointF(x - size, y - size / 2), QPointF(x + size / 2, y - size / 2),
                QPointF(x + size / 2, y - size), QPointF(x + size, y),
                QPointF(x + size / 2, y + size), QPointF(x + size / 2, y + size / 2),
                QPointF(x - size, y + size / 2),
            ]
        elif direction == "down":
            pts = [
                QPointF(x - size / 2, y - size), QPointF(x + size / 2, y - size),
                QPointF(x + size / 2, y + size / 2), QPointF(x + size, y + size / 2),
                QPointF(x, y + size), QPointF(x - size, y + size / 2),
                QPointF(x - size / 2, y + size / 2),
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
        p.setBrush(QBrush(gui_theme.C_POLE))
        pole_top = by - pole_h
        p.drawRect(QRectF(bx - pole_w / 2, pole_top, pole_w, pole_h))

        if facing == "right":
            p.drawRect(QRectF(bx, pole_top - arm_h / 2, arm_len, arm_h))
            self._draw_light_box(p, bx + arm_len, pole_top, bw, bh, r, active_color)
        else:
            p.drawRect(QRectF(bx - arm_len, pole_top - arm_h / 2, arm_len, arm_h))
            self._draw_light_box(p, bx - arm_len, pole_top, bw, bh, r, active_color)

    def _draw_light_box(self, p, lx, ly, bw, bh, r, active_color):
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(gui_theme.C_LIGHT_BOX_OUTER))
        p.drawRoundedRect(QRectF(lx - bw / 2 - 1, ly - bh / 2 - 1, bw + 2, bh + 2), 4, 4)
        p.setBrush(QBrush(gui_theme.C_LIGHT_BOX_BODY))
        p.drawRoundedRect(QRectF(lx - bw / 2, ly - bh / 2, bw, bh), 3, 3)
        inner_m = max(2, int(bw * 0.12))
        p.setBrush(QBrush(gui_theme.C_LIGHT_BOX_INNER))
        p.drawRoundedRect(QRectF(lx - bw / 2 + inner_m, ly - bh / 2 + inner_m, bw - inner_m * 2, bh - inner_m * 2), 2, 2)

        for i, cn in enumerate(["red", "yellow", "green"]):
            by = ly - bh / 3 + i * (bh / 3)
            is_on = cn == active_color
            if cn == "red":
                fill = gui_theme.C_RED if is_on else gui_theme.C_RED_DIM
            elif cn == "yellow":
                fill = gui_theme.C_YELLOW if is_on else gui_theme.C_YELLOW_DIM
            else:
                fill = gui_theme.C_GREEN if is_on else gui_theme.C_GREEN_DIM
            if is_on:
                glow = QColor(fill)
                glow.setAlpha(40)
                p.setBrush(QBrush(glow))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(QRectF(lx - r - 8, by - r - 8, (r + 8) * 2, (r + 8) * 2))
                glow2 = QColor(fill)
                glow2.setAlpha(80)
                p.setBrush(QBrush(glow2))
                p.drawEllipse(QRectF(lx - r - 3, by - r - 3, (r + 3) * 2, (r + 3) * 2))
            p.setBrush(QBrush(fill))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QRectF(lx - r, by - r, r * 2, r * 2))

    def _draw_vehicles(self, p, cx, cy, road_w, lane_w, W):
        car_w = max(8, int(road_w * 0.15))
        car_h = max(5, int(road_w * 0.09))

        p.setPen(Qt.PenStyle.NoPen)
        for i in range(min(self.car_x, 5)):
            x_off = cx - road_w / 2 - car_w * 2 - i * (car_w + 4)
            y_off = cy - lane_w * 0.5
            p.setBrush(QBrush(gui_theme.C_VEHICLE_BLUE))
            p.drawRoundedRect(QRectF(x_off, y_off - car_h / 2, car_w, car_h), 2, 2)
            p.setBrush(QBrush(gui_theme.C_VEHICLE_WINDOW_BLUE))
            p.drawRect(QRectF(x_off + car_w * 0.55, y_off - car_h / 2 + 1, car_w * 0.35, car_h - 2))

        p.setPen(Qt.PenStyle.NoPen)
        for i in range(min(self.car_x, 5)):
            x_off = cx + road_w / 2 + car_w * 0.5 + i * (car_w + 4)
            y_off = cy + lane_w * 0.5
            p.setBrush(QBrush(gui_theme.C_VEHICLE_BLUE))
            p.drawRoundedRect(QRectF(x_off, y_off - car_h / 2, car_w, car_h), 2, 2)
            p.setBrush(QBrush(gui_theme.C_VEHICLE_WINDOW_BLUE))
            p.drawRect(QRectF(x_off + car_w * 0.1, y_off - car_h / 2 + 1, car_w * 0.35, car_h - 2))

        p.setPen(Qt.PenStyle.NoPen)
        for i in range(min(self.car_y, 5)):
            x_off = cx + lane_w * 0.5
            y_off = cy - road_w / 2 - car_w * 2 - i * (car_w + 4)
            p.setBrush(QBrush(gui_theme.C_VEHICLE_ORANGE))
            p.drawRoundedRect(QRectF(x_off - car_h / 2, y_off, car_h, car_w), 2, 2)
            p.setBrush(QBrush(gui_theme.C_VEHICLE_WINDOW_ORANGE))
            p.drawRect(QRectF(x_off - car_h / 2 + 1, y_off + car_w * 0.55, car_h - 2, car_w * 0.35))

        p.setPen(Qt.PenStyle.NoPen)
        for i in range(min(self.car_y, 5)):
            x_off = cx - lane_w * 0.5
            y_off = cy + road_w / 2 + car_w * 0.5 + i * (car_w + 4)
            p.setBrush(QBrush(gui_theme.C_VEHICLE_ORANGE))
            p.drawRoundedRect(QRectF(x_off - car_h / 2, y_off, car_h, car_w), 2, 2)
            p.setBrush(QBrush(gui_theme.C_VEHICLE_WINDOW_ORANGE))
            p.drawRect(QRectF(x_off - car_h / 2 + 1, y_off + car_w * 0.1, car_h - 2, car_w * 0.35))


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
            p.fillRect(self.rect(), gui_theme.C_VIDEO_BG)
            p.setPen(QPen(gui_theme.C_VIDEO_PLACEHOLDER_TEXT))
            p.setFont(QFont("Microsoft YaHei", 12))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "无视频")
        p.end()


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
                    background: {gui_theme.C_NAV_ACTIVE_BG}; color: {gui_theme.C_NAV_ACTIVE_TEXT};
                    border: none; border-radius: 8px; padding: 6px 14px;
                    text-align: left; font-size: 13px; font-weight: bold;
                }}
                QPushButton:hover {{ background: {gui_theme.C_NAV_ACTIVE_HOVER_BG}; }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: {gui_theme.C_NAV_INACTIVE_BG}; color: {gui_theme.C_NAV_INACTIVE_TEXT};
                    border: none; border-radius: 8px; padding: 6px 14px;
                    text-align: left; font-size: 13px;
                }}
                QPushButton:hover {{ background: {gui_theme.C_NAV_HOVER_BG}; }}
            """)


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
            painter.fillRect(option.rect, QColor(gui_theme.C_TABLE_SELECTED_BG))
        elif index.row() % 2 == 1:
            painter.fillRect(option.rect, QColor(gui_theme.C_TABLE_ALT_BG))

        r = option.rect.adjusted(6, 4, -6, -4)
        bar_w = int(r.width() * pct / 100)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(gui_theme.C_PROGRESS_BG)))
        painter.drawRoundedRect(r, 3, 3)

        if bar_w > 0:
            fill_rect = QRectF(r.x(), r.y(), bar_w, r.height())
            grad = QLinearGradient(fill_rect.topLeft(), fill_rect.topRight())
            grad.setColorAt(0, gui_theme.C_PRIMARY)
            grad.setColorAt(1, gui_theme.C_PRIMARY_HOVER)
            painter.setBrush(QBrush(grad))
            painter.drawRoundedRect(fill_rect, 3, 3)

        painter.setPen(QPen(gui_theme.C_TEXT_PRIMARY))
        painter.setFont(QFont("Microsoft YaHei", 9))
        painter.drawText(option.rect, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()


class StatLabel(QWidget):
    def __init__(self, value="0", label="", color_key="primary", parent=None):
        super().__init__(parent)
        self._color_key = color_key
        self.setMinimumHeight(48)
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
        c = getattr(gui_theme, f"C_{self._color_key.upper()}", gui_theme.C_PRIMARY)
        self.val_label.setStyleSheet(f"color: {c.name()}; font-size: 20px; font-weight: bold;")
        self.name_label.setStyleSheet(f"color: {gui_theme.C_TEXT_MUTED.name()}; font-size: 11px;")

    def set_value(self, v):
        self.val_label.setText(str(v))


class TrafficLightIndicator(QWidget):
    def __init__(self, label="X 方向", color_key="blue", parent=None):
        super().__init__(parent)
        self._color_key = color_key
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._lbl = QLabel(label)
        layout.addWidget(self._lbl)
        self._dots = {}
        for name in ("red", "yellow", "green"):
            dot = QLabel("●")
            dot.setStyleSheet("font-size: 18px;")
            layout.addWidget(dot)
            self._dots[name] = dot
        self._refresh_style()

    def _refresh_style(self):
        c = getattr(gui_theme, f"C_{self._color_key.upper()}", gui_theme.C_BLUE)
        self._lbl.setStyleSheet(f"color: {c.name()}; font-weight: bold; font-size: 12px;")
        dims = {"red": gui_theme.C_RED_DIM, "yellow": gui_theme.C_YELLOW_DIM, "green": gui_theme.C_GREEN_DIM}
        for name, dot in self._dots.items():
            dot.setStyleSheet(f"color: {dims[name].name()}; font-size: 18px;")

    def set_active(self, color_name):
        on_map = {"red": gui_theme.C_RED, "yellow": gui_theme.C_YELLOW, "green": gui_theme.C_GREEN}
        dim_map = {"red": gui_theme.C_RED_DIM, "yellow": gui_theme.C_YELLOW_DIM, "green": gui_theme.C_GREEN_DIM}
        for name, dot in self._dots.items():
            c = on_map[name] if name == color_name else dim_map[name]
            dot.setStyleSheet(f"color: {c.name()}; font-size: 18px;")
