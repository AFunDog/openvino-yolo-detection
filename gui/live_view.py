"""Live video and traffic-signal simulation logic for MainWindow."""

from __future__ import annotations

import math
import os
import threading
import time
from pathlib import Path

import cv2

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QTableWidgetItem

from gui.theme import *
from gui.utils import load_json
from algorithm import load_frames


class LiveViewMixin:
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
        self.data_source_combo.blockSignals(True)
        self.data_source_combo.setCurrentIndex(0)
        self.data_source_combo.blockSignals(False)

    def _start_va_sim(self):
        """初始化离线回放仿真。"""
        sel = self.data_source_combo.currentText()
        self._sim_live_mode = False
        self.va_pair_summary = None
        if sel and sel != "(默认)" and sel != "实时检测":
            session_dir = os.path.join(str(self.DATA_DIR), sel)
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
        metrics_text = (
            f"模式: {mode_tag}\n"
            f"X: queue={feats['queue_x']}  wait={feats['wait_x']:.0f}s  gap={feats['gap_x']:.1f}s  arrival={feats['arrival_x']:.1f}/s\n"
            f"Y: queue={feats['queue_y']}  wait={feats['wait_y']:.0f}s  gap={feats['gap_y']:.1f}s  arrival={feats['arrival_y']:.1f}/s\n"
            f"有效: X={feats.get('line_count_x', 0)}  Y={feats.get('line_count_y', 0)}  总={feats.get('line_count_x', 0) + feats.get('line_count_y', 0)}\n"
            f"相位: {state['phase']}  已过绿灯:{state['phase_elapsed']:.1f}s  黄灯:{state['yellow_elapsed']:.1f}s"
        )
        self.metrics_view.setText(metrics_text)

        countdown_text = "--" if countdown is None else f"{math.ceil(countdown)}s"
        overview_text = (
            f"{mode_tag} / 周期 #{state['cycle_num'] + 1}\n"
            f"阶段: {state['phase']}  倒计时: {countdown_text}\n"
            f"X: queue={feats['queue_x']} wait={feats['wait_x']:.0f}s gap={feats['gap_x']:.1f}s arrival={feats['arrival_x']:.1f}/s\n"
            f"Y: queue={feats['queue_y']} wait={feats['wait_y']:.0f}s gap={feats['gap_y']:.1f}s arrival={feats['arrival_y']:.1f}/s\n"
            f"有效: X={feats.get('line_count_x', 0)}  Y={feats.get('line_count_y', 0)}"
        )
        self.overview_view.setText(overview_text)

        if self._sim_live_mode:
            self.cycle_info.setText(
                f"[{mode_tag}] 周期 #{state['cycle_num'] + 1}\n"
                f"X有效:{feats.get('line_count_x', 0)}  当前目标:{feats['queue_x']}  等待:{feats['wait_x']:.0f}s\n"
                f"Y有效:{feats.get('line_count_y', 0)}  当前目标:{feats['queue_y']}  等待:{feats['wait_y']:.0f}s\n"
                f"X到达:{feats['arrival_x']:.1f}/s  Y到达:{feats['arrival_y']:.1f}/s  绿灯已过:{state['phase_elapsed']:.1f}s"
            )
        else:
            self.cycle_info.setText(
                f"[{mode_tag}] 周期 #{state['cycle_num'] + 1}\n"
                f"X有效:{feats['queue_x']}  等待:{feats['wait_x']:.0f}s  "
                f"清空:{feats['gap_x']:.1f}s  到达:{feats['arrival_x']:.1f}/s\n"
                f"Y有效:{feats['queue_y']}  等待:{feats['wait_y']:.0f}s  "
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

            x_count = int(x_state.get("track_count_total", x_state.get("line_count_total", 0)))
            y_count = int(y_state.get("track_count_total", y_state.get("line_count_total", 0)))
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
                f"X有效:{x_count}  Y有效:{y_count}  总有效:{total_count}"
            )
            if self.detecting:
                self.detect_status.setText(
                    f"实时检测中... X有效:{x_count}  Y有效:{y_count}  当前目标:{total_objects}"
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
