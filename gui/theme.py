"""Theme helpers and dynamic style registration for the GUI."""

from __future__ import annotations

from gui import theme_manager as tm

from PyQt6.QtWidgets import QApplication

__all__ = [
    "C_PRIMARY",
    "C_PRIMARY_HOVER",
    "C_RED",
    "C_YELLOW",
    "C_GREEN",
    "C_RED_DIM",
    "C_YELLOW_DIM",
    "C_GREEN_DIM",
    "C_BLUE",
    "C_ORANGE",
    "C_BG_BASE",
    "C_BG_SURFACE",
    "C_BG_ELEVATED",
    "C_BG_OVERLAY",
    "C_CARD_BG",
    "C_CARD_BORDER",
    "C_TEXT_PRIMARY",
    "C_TEXT_SECONDARY",
    "C_TEXT_MUTED",
    "C_BORDER",
    "C_BORDER_LIGHT",
    "C_ROAD",
    "C_ROAD_MARK",
    "C_GRASS",
    "C_SIDEWALK_BG",
    "C_INTERSECTION",
    "C_LANE",
    "C_CROSSWALK",
    "C_SIDEWALK",
    "C_STOP_LINE",
    "C_CENTER_LINE",
    "C_POLE",
    "C_CURB",
    "C_NAV_ACTIVE_BG",
    "C_NAV_ACTIVE_TEXT",
    "C_NAV_INACTIVE_BG",
    "C_NAV_INACTIVE_TEXT",
    "C_NAV_HOVER_BG",
    "C_NAV_ACTIVE_HOVER_BG",
    "C_SIDEBAR_BG",
    "C_SIDEBAR_BORDER",
    "C_INPUT_BG",
    "C_INPUT_BORDER",
    "C_INPUT_FOCUS_BORDER",
    "C_INPUT_FOCUS_BG",
    "C_TABLE_BG",
    "C_TABLE_ALT_BG",
    "C_TABLE_HEADER_BG",
    "C_TABLE_HEADER_TEXT",
    "C_TABLE_BORDER",
    "C_TABLE_SELECTED_BG",
    "C_PROGRESS_BG",
    "C_PROGRESS_CHUNK",
    "C_PROGRESS_CHUNK_YELLOW",
    "C_SIDEBAR_ITEM_BG",
    "C_SIDEBAR_ITEM_SELECTED_BG",
    "C_SIDEBAR_ITEM_SELECTED_TEXT",
    "C_DETAIL_BG",
    "C_DETAIL_BORDER",
    "C_STATUS_BG",
    "C_STATUS_BORDER",
    "C_MENU_BG",
    "C_MENU_BORDER",
    "C_MENU_DELETE_BG",
    "C_MENU_DELETE_TEXT",
    "C_CANVAS_COUNTDOWN_BG",
    "C_CANVAS_COUNTDOWN_TEXT",
    "C_QMESSAGEBOX_BG",
    "C_QMESSAGEBOX_TEXT",
    "C_SCROLLBAR_BG",
    "C_SCROLLBAR_HANDLE",
    "C_SCROLLBAR_HANDLE_HOVER",
    "C_TOOLTIP_BG",
    "C_TOOLTIP_TEXT",
    "C_ARROW_COLOR",
    "C_VEHICLE_BLUE",
    "C_VEHICLE_ORANGE",
    "C_VEHICLE_WINDOW_BLUE",
    "C_VEHICLE_WINDOW_ORANGE",
    "C_LIGHT_BOX_OUTER",
    "C_LIGHT_BOX_BODY",
    "C_LIGHT_BOX_INNER",
    "C_VIDEO_BG",
    "C_VIDEO_PLACEHOLDER_TEXT",
    "C_IS_DARK",
    "init_colors",
    "make_app_stylesheet",
    "refresh_all_styles",
    "_ds",
    "_init_colors",
    "_make_app_stylesheet",
    "_refresh_all_styles",
]

_dynamic_styles: dict = {}

_theme = tm.colors()

# 显式声明，避免 IDE 将动态注入的颜色常量标为未定义
C_PRIMARY = _theme["primary"]
C_PRIMARY_HOVER = _theme["primary_hover"]
C_RED = _theme["red"]
C_YELLOW = _theme["yellow"]
C_GREEN = _theme["green"]
C_RED_DIM = _theme["red_dim"]
C_YELLOW_DIM = _theme["yellow_dim"]
C_GREEN_DIM = _theme["green_dim"]
C_BLUE = _theme["blue"]
C_ORANGE = _theme["orange"]
C_BG_BASE = _theme["bg_base"]
C_BG_SURFACE = _theme["bg_surface"]
C_BG_ELEVATED = _theme["bg_elevated"]
C_BG_OVERLAY = _theme["bg_overlay"]
C_CARD_BG = _theme["card_bg"]
C_CARD_BORDER = _theme["card_border"]
C_TEXT_PRIMARY = _theme["text_primary"]
C_TEXT_SECONDARY = _theme["text_secondary"]
C_TEXT_MUTED = _theme["text_muted"]
C_BORDER = _theme["border"]
C_BORDER_LIGHT = _theme["border_light"]
C_ROAD = _theme["road"]
C_ROAD_MARK = _theme["road_mark"]
C_GRASS = _theme["grass"]
C_SIDEWALK_BG = _theme["sidewalk_bg"]
C_INTERSECTION = _theme["intersection"]
C_LANE = _theme["lane"]
C_CROSSWALK = _theme["crosswalk"]
C_SIDEWALK = _theme["sidewalk"]
C_STOP_LINE = _theme["stop_line"]
C_CENTER_LINE = _theme["center_line"]
C_POLE = _theme["pole"]
C_CURB = _theme["curb"]
C_NAV_ACTIVE_BG = _theme["nav_active_bg"]
C_NAV_ACTIVE_TEXT = _theme["nav_active_text"]
C_NAV_INACTIVE_BG = _theme["nav_inactive_bg"]
C_NAV_INACTIVE_TEXT = _theme["nav_inactive_text"]
C_NAV_HOVER_BG = _theme["nav_hover_bg"]
C_NAV_ACTIVE_HOVER_BG = _theme["nav_active_hover_bg"]
C_SIDEBAR_BG = _theme["sidebar_bg"]
C_SIDEBAR_BORDER = _theme["sidebar_border"]
C_INPUT_BG = _theme["input_bg"]
C_INPUT_BORDER = _theme["input_border"]
C_INPUT_FOCUS_BORDER = _theme["input_focus_border"]
C_INPUT_FOCUS_BG = _theme["input_focus_bg"]
C_TABLE_BG = _theme["table_bg"]
C_TABLE_ALT_BG = _theme["table_alt_bg"]
C_TABLE_HEADER_BG = _theme["table_header_bg"]
C_TABLE_HEADER_TEXT = _theme["table_header_text"]
C_TABLE_BORDER = _theme["table_border"]
C_TABLE_SELECTED_BG = _theme["table_selected_bg"]
C_PROGRESS_BG = _theme["progress_bg"]
C_PROGRESS_CHUNK = _theme["progress_chunk"]
C_PROGRESS_CHUNK_YELLOW = _theme["progress_chunk_yellow"]
C_SIDEBAR_ITEM_BG = _theme["sidebar_item_bg"]
C_SIDEBAR_ITEM_SELECTED_BG = _theme["sidebar_item_selected_bg"]
C_SIDEBAR_ITEM_SELECTED_TEXT = _theme["sidebar_item_selected_text"]
C_DETAIL_BG = _theme["detail_bg"]
C_DETAIL_BORDER = _theme["detail_border"]
C_STATUS_BG = _theme["status_bg"]
C_STATUS_BORDER = _theme["status_border"]
C_MENU_BG = _theme["menu_bg"]
C_MENU_BORDER = _theme["menu_border"]
C_MENU_DELETE_BG = _theme["menu_delete_bg"]
C_MENU_DELETE_TEXT = _theme["menu_delete_text"]
C_CANVAS_COUNTDOWN_BG = _theme["canvas_countdown_bg"]
C_CANVAS_COUNTDOWN_TEXT = _theme["canvas_countdown_text"]
C_QMESSAGEBOX_BG = _theme["qmessagebox_bg"]
C_QMESSAGEBOX_TEXT = _theme["qmessagebox_text"]
C_SCROLLBAR_BG = _theme["scrollbar_bg"]
C_SCROLLBAR_HANDLE = _theme["scrollbar_handle"]
C_SCROLLBAR_HANDLE_HOVER = _theme["scrollbar_handle_hover"]
C_TOOLTIP_BG = _theme["tooltip_bg"]
C_TOOLTIP_TEXT = _theme["tooltip_text"]
C_ARROW_COLOR = _theme["arrow_color"]
C_VEHICLE_BLUE = _theme["vehicle_blue"]
C_VEHICLE_ORANGE = _theme["vehicle_orange"]
C_VEHICLE_WINDOW_BLUE = _theme["vehicle_window_blue"]
C_VEHICLE_WINDOW_ORANGE = _theme["vehicle_window_orange"]
C_LIGHT_BOX_OUTER = _theme["light_box_outer"]
C_LIGHT_BOX_BODY = _theme["light_box_body"]
C_LIGHT_BOX_INNER = _theme["light_box_inner"]
C_VIDEO_BG = _theme["video_bg"]
C_VIDEO_PLACEHOLDER_TEXT = _theme["video_placeholder_text"]
C_IS_DARK = tm.is_dark()


def init_colors(dark: bool):
    """根据 is_dark 重新设定所有 C_* 模块级全局变量。"""
    t = tm.DARK if dark else tm.LIGHT
    for key, val in t.items():
        globals()[f"C_{key.upper()}"] = val
    globals()["C_IS_DARK"] = dark


def _ds(widget, style_func):
    """注册控件样式表，主题切换时自动通过 style_func() 重新生成。"""
    _dynamic_styles[id(widget)] = (widget, style_func)
    widget.setStyleSheet(style_func())


def refresh_all_styles():
    """主题切换后刷新所有已注册的动态样式表。"""
    for key, (widget, style_func) in list(_dynamic_styles.items()):
        try:
            widget.setStyleSheet(style_func())
        except RuntimeError:
            del _dynamic_styles[key]


def make_app_stylesheet() -> str:
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


init_colors(tm.is_dark())

_init_colors = init_colors
_make_app_stylesheet = make_app_stylesheet
_refresh_all_styles = refresh_all_styles
