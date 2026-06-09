"""UI construction helpers for MainWindow."""

from __future__ import annotations

from typing import Any, Protocol

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget, QPushButton, QLabel,
    QLineEdit, QTextEdit, QFileDialog, QSlider, QComboBox, QListWidget,
    QProgressBar, QTableWidget, QSizePolicy, QAbstractItemView,
    QHeaderView,
)

from gui.theme import *
from gui.widgets import (
    CardWidget, IntersectionCanvas, VideoPreviewWidget, NavButton,
    StatLabel, TrafficLightIndicator,
)


class _UIBuilderHost(Protocol):
    def setCentralWidget(self, widget) -> None: ...

    nav_yolo: Any
    nav_traffic: Any
    stack: Any
    session_list: Any
    video_path_x_input: Any
    video_path_y_input: Any
    btn_detect: Any
    detect_status: Any
    btn_play: Any
    stat_frames: Any
    stat_detections: Any
    stat_vehicles: Any
    stat_x_count: Any
    stat_y_count: Any
    stat_fps: Any
    session_detail: Any
    overview_view: Any
    class_table: Any
    video_preview_x: Any
    video_preview_y: Any
    video_preview: Any
    canvas: Any
    xl_indicator: Any
    yl_indicator: Any
    metrics_view: Any
    phase_label: Any
    timer_text: Any
    progress_bar: Any
    btn_start_sim: Any
    btn_pause_sim: Any
    btn_reset_sim: Any
    speed_slider: Any
    speed_val: Any
    data_source_combo: Any
    cycle_info: Any
    history_table: Any
    status_label: Any

    def _btn_style(self, bg_color, radius=6): ...
    def _switch_page(self, idx): ...
    def _on_browse_video(self, direction): ...
    def _on_start_detect(self): ...
    def _on_play_video(self): ...
    def _on_start_sim(self): ...
    def _on_pause_sim(self): ...
    def _on_reset_sim(self): ...
    def _on_session_select(self, row): ...
    def _on_session_context_menu(self, pos): ...
    def _on_data_source_changed(self, text): ...


class UIBuilderMixin:
    def _build_ui(self: _UIBuilderHost):
        central = QWidget()
        self.setCentralWidget(central)
        _ds(central, lambda: f"background: {C_BG_BASE.name()};")

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

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

        self.stack = QStackedWidget()
        self.stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._build_yolo_page()
        self._build_traffic_page()

        body.addWidget(self.stack, 1)
        main_layout.addLayout(body, 1)

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

    def _build_yolo_page(self: _UIBuilderHost):
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
        _ds(btn_browse_x, lambda: self._btn_style(C_PRIMARY))
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
        _ds(btn_browse_y, lambda: self._btn_style(C_PRIMARY))
        btn_browse_y.clicked.connect(lambda: self._on_browse_video("y"))
        y_row.addWidget(btn_browse_y)

        self.btn_detect = QPushButton("开始检测")
        self.btn_detect.setFixedSize(90, 32)
        _ds(self.btn_detect, lambda: self._btn_style(C_GREEN))
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
        _ds(self.btn_play, lambda: self._btn_style(C_PRIMARY, 10))
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
        self.stat_vehicles = StatLabel("0", "总有效车", "green")
        self.stat_x_count = StatLabel("0", "X有效", "blue")
        self.stat_y_count = StatLabel("0", "Y有效", "orange")
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

        card_overview = CardWidget("运行概览")
        overview_layout = QVBoxLayout(card_overview)
        self.overview_view = QTextEdit()
        self.overview_view.setReadOnly(True)
        self.overview_view.setFixedHeight(112)
        self.overview_view.setPlaceholderText("等待实时数据...")
        _ds(self.overview_view, lambda: f"""
            QTextEdit {{
                border: 1px solid {C_SIDEBAR_BORDER};
                border-radius: 8px;
                padding: 8px 10px;
                background: {C_DETAIL_BG};
                color: {C_TEXT_PRIMARY.name()};
                font-size: 12px;
            }}
        """)
        overview_layout.addWidget(self.overview_view)
        left_col.addWidget(card_overview)

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
        self.class_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.class_table.setFixedHeight(216)
        class_layout.setContentsMargins(0, 0, 0, 0)
        class_layout.setSpacing(0)
        class_layout.addWidget(self.class_table)
        left_col.addWidget(card_class)

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

        metrics_lbl = QLabel("控制指标")
        _ds(metrics_lbl, lambda: f"color: {C_TEXT_MUTED.name()}; font-size: 12px; font-weight: bold;")
        traffic_right.addWidget(metrics_lbl)

        self.metrics_view = QTextEdit()
        self.metrics_view.setReadOnly(True)
        self.metrics_view.setFixedHeight(112)
        self.metrics_view.setPlaceholderText("等待检测数据...")
        _ds(self.metrics_view, lambda: f"""
            QTextEdit {{
                border: 1px solid {C_SIDEBAR_BORDER};
                border-radius: 8px;
                padding: 8px 10px;
                background: {C_DETAIL_BG};
                color: {C_TEXT_PRIMARY.name()};
                font-size: 12px;
            }}
        """)
        traffic_right.addWidget(self.metrics_view)

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
        _ds(self.btn_start_sim, lambda: self._btn_style(C_GREEN))
        self.btn_start_sim.clicked.connect(self._on_start_sim)
        btn_row.addWidget(self.btn_start_sim)

        self.btn_pause_sim = QPushButton("⏸ 暂停")
        self.btn_pause_sim.setFixedSize(90, 32)
        _ds(self.btn_pause_sim, lambda: self._btn_style(C_YELLOW))
        self.btn_pause_sim.clicked.connect(self._on_pause_sim)
        btn_row.addWidget(self.btn_pause_sim)

        self.btn_reset_sim = QPushButton("■ 重置")
        self.btn_reset_sim.setFixedSize(90, 32)
        _ds(self.btn_reset_sim, lambda: self._btn_style(C_RED))
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

    def _build_traffic_page(self: _UIBuilderHost):
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

    def _connect_signals(self: _UIBuilderHost):
        self.nav_yolo.clicked.connect(lambda: self._switch_page(0))
        self.nav_traffic.clicked.connect(lambda: self._switch_page(0))
        self.session_list.currentRowChanged.connect(self._on_session_select)
        self.data_source_combo.currentTextChanged.connect(self._on_data_source_changed)

    def _switch_page(self: _UIBuilderHost, idx):
        self.stack.setCurrentIndex(idx)
        self.nav_yolo.active = (idx == 0)
        self.nav_traffic.active = (idx == 1)

    def _on_browse_video(self: _UIBuilderHost, direction):
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
