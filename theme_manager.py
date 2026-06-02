#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主题管理器 —— 自动跟随 Windows 系统明暗主题切换颜色。
通过 QPalette 控制默认控件颜色，同时导出 LIGHT/DARK 调色板供自定义绘制和 QSS 使用。
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication


class _Theme:
    """单例主题。"""

    def __init__(self):
        app = QApplication.instance()
        self._dark = app.styleHints().colorScheme() == Qt.ColorScheme.Dark if app else False
        self._palette = DARK if self._dark else LIGHT

    @property
    def is_dark(self) -> bool:
        return self._dark

    @property
    def colors(self) -> dict:
        return self._palette

    def toggle(self, dark: bool):
        if self._dark == dark:
            return
        self._dark = dark
        self._palette = DARK if dark else LIGHT


# ═══════════════════════════════════════════════════════════
# 亮色主题
# ═══════════════════════════════════════════════════════════
LIGHT = {
    # 品牌 / 功能色
    "primary":        QColor(79, 70, 229),
    "primary_hover":  QColor(109, 99, 255),
    "red":            QColor(220, 38, 38),
    "yellow":         QColor(202, 138, 4),
    "green":          QColor(22, 163, 74),
    "red_dim":        QColor(220, 38, 38, 60),
    "yellow_dim":     QColor(202, 138, 4, 60),
    "green_dim":      QColor(22, 163, 74, 60),
    "blue":           QColor(37, 99, 235),
    "orange":         QColor(234, 88, 12),

    # 背景
    "bg_base":        QColor(240, 242, 245),
    "bg_surface":     QColor(255, 255, 255),
    "bg_elevated":    QColor(241, 243, 245),
    "bg_overlay":     QColor(233, 236, 239),
    "card_bg":        QColor(255, 255, 255),
    "card_border":    QColor(226, 229, 235),

    # 文字
    "text_primary":   QColor(17, 24, 39),
    "text_secondary": QColor(55, 65, 81),
    "text_muted":     QColor(107, 114, 128),

    # 边框
    "border":         QColor(220, 225, 231),
    "border_light":   QColor(235, 238, 243),

    # Canvas
    "road":                  QColor(82, 86, 89),
    "road_mark":             QColor(220, 220, 220),
    "grass":                 QColor(76, 153, 76),
    "sidewalk_bg":           QColor(160, 155, 145),
    "intersection":          QColor(88, 92, 95),
    "lane":                  QColor(200, 200, 200, 180),
    "crosswalk":             QColor(240, 240, 240, 200),
    "sidewalk":              QColor(170, 165, 155),
    "stop_line":             QColor(240, 240, 240, 220),
    "center_line":           QColor(240, 200, 60, 200),
    "pole":                  QColor(100, 100, 110),
    "curb":                  QColor(140, 135, 125),

    # 组件
    "nav_active_bg":           "#4f46e5",
    "nav_active_text":         "white",
    "nav_inactive_bg":         "#f1f3f5",
    "nav_inactive_text":       "#374151",
    "nav_hover_bg":            "#e9ecef",
    "nav_active_hover_bg":     "#6d63ff",

    "sidebar_bg":              "#ffffff",
    "sidebar_border":          "#e2e5eb",

    "input_bg":                "#f9fafb",
    "input_border":            "#d1d5db",
    "input_focus_border":      "#4f46e5",
    "input_focus_bg":          "#ffffff",

    "table_bg":                "#ffffff",
    "table_alt_bg":            "#fafbfc",
    "table_header_bg":         "#334155",
    "table_header_text":       "#ffffff",
    "table_border":            "#f3f4f6",
    "table_selected_bg":       "#eef2ff",

    "progress_bg":             "#e5e7eb",
    "progress_chunk":          "#22c55e",
    "progress_chunk_yellow":   "#ca8a04",

    "sidebar_item_bg":         "#f9fafb",
    "sidebar_item_selected_bg": "#eef2ff",
    "sidebar_item_selected_text": "#4f46e5",

    "detail_bg":               "#f9fafb",
    "detail_border":           "#e2e5eb",

    "status_bg":               "#ffffff",
    "status_border":           "#e2e5eb",

    "menu_bg":                 "#ffffff",
    "menu_border":             "#e2e5eb",
    "menu_delete_bg":          "#fee2e2",
    "menu_delete_text":        "#dc2626",

    "canvas_countdown_bg":     QColor(0, 0, 0, 80),
    "canvas_countdown_text":   QColor(255, 255, 255, 220),

    "qmessagebox_bg":          "#ffffff",
    "qmessagebox_text":        "#111827",

    "scrollbar_bg":            "#f3f4f6",
    "scrollbar_handle":        "#d1d5db",
    "scrollbar_handle_hover":  "#9ca3af",

    "tooltip_bg":              "#1f2937",
    "tooltip_text":            "white",

    "arrow_color":             QColor(220, 220, 220, 160),
    "vehicle_blue":            QColor(37, 99, 235),
    "vehicle_orange":          QColor(234, 88, 12),
    "vehicle_window_blue":     QColor(150, 200, 255, 180),
    "vehicle_window_orange":   QColor(255, 220, 150, 180),

    "light_box_outer":         QColor(30, 30, 35),
    "light_box_body":          QColor(60, 60, 68),
    "light_box_inner":         QColor(40, 40, 48),
    "video_bg":                QColor(20, 20, 26),
    "video_placeholder_text":  QColor(107, 114, 128),
}


# ═══════════════════════════════════════════════════════════
# 暗色主题
# ═══════════════════════════════════════════════════════════
DARK = {
    "primary":        QColor(129, 140, 248),
    "primary_hover":  QColor(165, 180, 252),
    "red":            QColor(248, 113, 113),
    "yellow":         QColor(250, 204, 21),
    "green":          QColor(74, 222, 128),
    "red_dim":        QColor(248, 113, 113, 60),
    "yellow_dim":     QColor(250, 204, 21, 60),
    "green_dim":      QColor(74, 222, 128, 60),
    "blue":           QColor(96, 165, 250),
    "orange":         QColor(251, 146, 60),

    "bg_base":        QColor(30, 31, 34),
    "bg_surface":     QColor(37, 38, 42),
    "bg_elevated":    QColor(44, 45, 49),
    "bg_overlay":     QColor(50, 52, 56),
    "card_bg":        QColor(44, 45, 49),
    "card_border":    QColor(55, 57, 62),

    "text_primary":   QColor(229, 231, 235),
    "text_secondary": QColor(179, 181, 187),
    "text_muted":     QColor(126, 129, 136),

    "border":         QColor(55, 57, 62),
    "border_light":   QColor(48, 49, 53),

    "road":                  QColor(55, 58, 61),
    "road_mark":             QColor(130, 130, 130),
    "grass":                 QColor(45, 90, 45),
    "sidewalk_bg":           QColor(100, 96, 88),
    "intersection":          QColor(58, 61, 64),
    "lane":                  QColor(130, 130, 130, 160),
    "crosswalk":             QColor(140, 140, 140, 160),
    "sidewalk":              QColor(108, 104, 96),
    "stop_line":             QColor(160, 160, 160, 180),
    "center_line":           QColor(200, 170, 40, 180),
    "pole":                  QColor(80, 80, 88),
    "curb":                  QColor(90, 86, 78),

    "nav_active_bg":           "#818cf8",
    "nav_active_text":         "#1e1f23",
    "nav_inactive_bg":         "#38393e",
    "nav_inactive_text":       "#b3b5bb",
    "nav_hover_bg":            "#4a4b50",
    "nav_active_hover_bg":     "#a5b4fc",

    "sidebar_bg":              "#25262a",
    "sidebar_border":          "#37393e",

    "input_bg":                "#2e2f34",
    "input_border":            "#4a4b50",
    "input_focus_border":      "#818cf8",
    "input_focus_bg":          "#37383d",

    "table_bg":                "#2c2d31",
    "table_alt_bg":            "#323338",
    "table_header_bg":         "#1e1f23",
    "table_header_text":       "#e5e7eb",
    "table_border":            "#3e3f44",
    "table_selected_bg":       "#2a2d4a",

    "progress_bg":             "#3e3f44",
    "progress_chunk":          "#4ade80",
    "progress_chunk_yellow":   "#facc15",

    "sidebar_item_bg":         "#2e2f34",
    "sidebar_item_selected_bg": "#2a2d4a",
    "sidebar_item_selected_text": "#a5b4fc",

    "detail_bg":               "#2e2f34",
    "detail_border":           "#37393e",

    "status_bg":               "#25262a",
    "status_border":           "#37393e",

    "menu_bg":                 "#35363b",
    "menu_border":             "#4a4b50",
    "menu_delete_bg":          "#4a1e1e",
    "menu_delete_text":        "#fca5a5",

    "canvas_countdown_bg":     QColor(0, 0, 0, 120),
    "canvas_countdown_text":   QColor(229, 231, 235, 200),

    "qmessagebox_bg":          "#37383d",
    "qmessagebox_text":        "#e5e7eb",

    "scrollbar_bg":            "#2e2f34",
    "scrollbar_handle":        "#55565b",
    "scrollbar_handle_hover":  "#6f7075",

    "tooltip_bg":              "#e5e7eb",
    "tooltip_text":            "#1f2937",

    "arrow_color":             QColor(140, 140, 140, 140),
    "vehicle_blue":            QColor(96, 165, 250),
    "vehicle_orange":          QColor(251, 146, 60),
    "vehicle_window_blue":     QColor(147, 197, 253, 160),
    "vehicle_window_orange":   QColor(253, 186, 116, 160),

    "light_box_outer":         QColor(20, 20, 24),
    "light_box_body":          QColor(50, 50, 56),
    "light_box_inner":         QColor(30, 30, 36),
    "video_bg":                QColor(18, 18, 22),
    "video_placeholder_text":  QColor(126, 129, 136),
}


# ── 全局单例 ──────────────────────────────────────────────
_theme = None


def _init():
    global _theme
    if _theme is None:
        _theme = _Theme()


def is_dark() -> bool:
    _init()
    return _theme.is_dark


def colors() -> dict:
    _init()
    return _theme.colors


def set_dark(dark: bool):
    _init()
    _theme.toggle(dark)


# ── QPalette ───────────────────────────────────────────────

def create_palette(dark: bool) -> QPalette:
    """根据明暗主题返回完整 QPalette，Qt 控件默认颜色由此决定。"""
    t = DARK if dark else LIGHT

    p = QPalette()

    # 通用
    p.setColor(QPalette.ColorRole.Window,          t["bg_surface"])
    p.setColor(QPalette.ColorRole.WindowText,      t["text_primary"])
    p.setColor(QPalette.ColorRole.Base,            t["card_bg"])
    p.setColor(QPalette.ColorRole.AlternateBase,   t["bg_base"])
    p.setColor(QPalette.ColorRole.Text,            t["text_primary"])
    p.setColor(QPalette.ColorRole.PlaceholderText, t["text_muted"])
    p.setColor(QPalette.ColorRole.BrightText,      t["red"])
    p.setColor(QPalette.ColorRole.Highlight,       t["primary"])
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))

    # 按钮
    p.setColor(QPalette.ColorRole.Button,       t["bg_elevated"])
    p.setColor(QPalette.ColorRole.ButtonText,   t["text_primary"])

    # 提示
    p.setColor(QPalette.ColorRole.ToolTipBase,  QColor(t["tooltip_bg"]))
    p.setColor(QPalette.ColorRole.ToolTipText,  QColor(t["tooltip_text"]))

    # 禁用态
    p.setColor(QPalette.ColorGroup.Disabled,
               QPalette.ColorRole.WindowText, t["text_muted"])
    p.setColor(QPalette.ColorGroup.Disabled,
               QPalette.ColorRole.ButtonText, t["text_muted"])
    p.setColor(QPalette.ColorGroup.Disabled,
               QPalette.ColorRole.Text,       t["text_muted"])

    return p


# ── 观察者 ─────────────────────────────────────────────────
_observers: list = []


def on_theme_changed(callback):
    """注册回调：callback(is_dark: bool)。"""
    _observers.append(callback)


def _notify(dark: bool):
    for cb in _observers:
        cb(dark)


def install(app: QApplication):
    """安装主题管理器：设置初始 palette 并连接系统主题变化信号。"""
    _init()
    app.setPalette(create_palette(is_dark()))

    def _on_scheme_change(scheme):
        dark = scheme == Qt.ColorScheme.Dark
        if dark == _theme.is_dark:
            return
        set_dark(dark)
        app.setPalette(create_palette(dark))
        _notify(dark)

    app.styleHints().colorSchemeChanged.connect(_on_scheme_change)
