"""Session/history related view logic for MainWindow."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QBrush
from PyQt6.QtWidgets import QMenu, QMessageBox, QTableWidgetItem

from gui.theme import *
from gui.utils import load_json

VEHICLE_CLASSES = {"car", "van", "bus", "truck"}


class SessionViewMixin:
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
                "track_count_total": summary.get("track_count_total", summary.get("line_count_total", 0)),
                "track_count_keep": summary.get("track_count_keep", summary.get("line_count_in", 0)),
                "track_count_slow": summary.get("track_count_slow", summary.get("line_count_slow", 0)),
                "track_count_filtered": summary.get("track_count_filtered", summary.get("line_count_out", 0)),
                "line_count_total": summary.get("line_count_total", 0),
                "line_count_in": summary.get("line_count_in", 0),
                "line_count_slow": summary.get("line_count_slow", 0),
                "line_count_out": summary.get("line_count_out", 0),
                "crossed_class_counts": summary.get("crossed_class_counts", {}),
                "slow_class_counts": summary.get("slow_class_counts", {}),
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
        session_dir = self.DATA_DIR / session_name
        session_dir.mkdir(parents=True, exist_ok=True)

        line_count_x = int(by_direction["X"]["track_count_total"])
        line_count_y = int(by_direction["Y"]["track_count_total"])
        summary = {
            "session_type": "direction_pair",
            "source": "same_intersection_xy_pair",
            "description": "同一路口两段垂直方向监控视频的轨迹方向过滤统计",
            "count_method": "trajectory_direction_filter_by_direction",
            "track_count_x": line_count_x,
            "track_count_y": line_count_y,
            "line_count_x": line_count_x,
            "line_count_y": line_count_y,
            "line_count_total": line_count_x + line_count_y,
            "direction_videos": by_direction,
            "preview_output": by_direction["Y"].get("output_path") or by_direction["X"].get("output_path"),
        }
        with open(session_dir / "summary.json", "w", encoding="utf-8") as f:
            import json

            json.dump(summary, f, ensure_ascii=False, indent=2)
        return str(session_dir)

    def _load_sessions(self):
        self.sessions = []
        self.session_list.clear()
        if not self.DATA_DIR.exists():
            return
        for d in sorted(self.DATA_DIR.iterdir(), reverse=True):
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
            f"有效总数: {vehicle_count}  通过:{count_in}  过滤:{count_out}  FPS: {fps_val:.1f}"
        )
        self.session_detail.setText(info)
        self.overview_view.setText(
            f"会话: 单视频轨迹过滤\n"
            f"总帧数: {frames}  总检测: {total}\n"
            f"有效车辆: {vehicle_count}  通过:{count_in}  过滤:{count_out}\n"
            f"后端: {backend_label}  分辨率: {vi.get('width', '?')}x{vi.get('height', '?')}"
        )

    def _render_direction_pair_session(self, summary):
        direction_videos = summary.get("direction_videos", {})
        x_info = direction_videos.get("X", {})
        y_info = direction_videos.get("Y", {})
        x_count = int(summary.get("track_count_x", summary.get("line_count_x", x_info.get("line_count_total", 0))))
        y_count = int(summary.get("track_count_y", summary.get("line_count_y", y_info.get("line_count_total", 0))))
        total_count = int(summary.get("track_count_total", summary.get("line_count_total", x_count + y_count)))
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
        rows = [("X方向有效", x_count), ("Y方向有效", y_count)]
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
            f"X方向有效车辆: {x_count}  Y方向有效车辆: {y_count}\n"
            f"总有效车数: {total_count}\n"
            f"可直接作为交通灯仿真的 X / Y 决策输入"
        )
        self.session_detail.setText(info)
        self.overview_view.setText(
            f"会话: 双方向聚合\n"
            f"X有效:{x_count}  Y有效:{y_count}  总有效:{total_count}\n"
            f"总帧数: {total_frames}  总检测: {total_detections}\n"
            f"后端: X={x_backend} / Y={y_backend}"
        )
