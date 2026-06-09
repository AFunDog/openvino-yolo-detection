"""Detection orchestration logic for MainWindow."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Protocol

import cv2

from gui.theme import *


class _DetectControllerHost(Protocol):
    video_path_x_input: Any
    video_path_y_input: Any
    detect_status: Any
    detecting: bool
    detect_progress: Any
    _detect_latest: dict
    _detect_dirty: bool
    PROJECT_ROOT: Path
    TEST_OUTPUT_DIR: Path
    btn_detect: Any
    btn_play: Any
    sim_running: bool
    sim_paused: bool
    last_tick: float
    data_source_combo: Any
    cycle_info: Any
    status_label: Any
    _live_dual_lock: threading.Lock
    _live_dual_state: dict
    _new_live_direction_state: Any
    _live_dual_dirty: bool

    def _start_live_sim(self) -> None: ...
    def _reset_live_dual_state(self, video_pairs=None) -> None: ...
    def _save_direction_pair_session(self, outputs) -> str: ...
    def _load_video(self, path) -> bool: ...


class DetectControllerMixin:
    def _on_start_detect(self: _DetectControllerHost):
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

        model_onnx_path = self.PROJECT_ROOT / "public" / "yolo-v26" / "yolo26n.onnx"
        model_xml_path = self.PROJECT_ROOT / "public" / "yolo-v26" / "ir_model" / "yolo26n.xml"
        model_bin_path = self.PROJECT_ROOT / "public" / "yolo-v26" / "ir_model" / "yolo26n.bin"
        has_onnx = model_onnx_path.exists()
        has_openvino_ir = model_xml_path.exists() and model_bin_path.exists()
        if not (has_onnx or has_openvino_ir):
            self.detect_status.setText("错误: YOLOv26 模型文件不存在（需 ONNX 或 OpenVINO IR）")
            self.detect_status.setStyleSheet(f"color: {C_RED.name()}; font-size: 12px;")
            return

        os.makedirs(str(self.TEST_OUTPUT_DIR), exist_ok=True)

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

        def on_detect_frame(direction, frame_bgr, frame_idx, avg_fps, num_objects, video_fps, line_counts, class_counts, backend):
            direction = str(direction).upper()
            counts = line_counts or {}
            class_counts = class_counts or {}
            with self._live_dual_lock:
                state = self._live_dual_state.setdefault(direction, self._new_live_direction_state())
                state["frame_bgr"] = frame_bgr
                state["frame_idx"] = frame_idx
                state["fps"] = avg_fps
                state["num_objects"] = num_objects
                state["track_count_total"] = int(counts.get("track_count_total", counts.get("line_count_total", 0)))
                state["track_count_keep"] = int(counts.get("track_count_keep", counts.get("line_count_in", 0)))
                state["track_count_slow"] = int(counts.get("track_count_slow", counts.get("line_count_slow", 0)))
                state["track_count_filtered"] = int(counts.get("track_count_filtered", counts.get("line_count_out", 0)))
                state["line_count_total"] = state["track_count_total"]
                state["line_count_in"] = state["track_count_keep"]
                state["line_count_slow"] = state["track_count_slow"]
                state["line_count_out"] = state["track_count_filtered"]
                state["frame_class_counts"] = dict(class_counts.get("frame_class_counts", {}))
                state["class_counts"] = dict(class_counts.get("crossed_class_counts", class_counts.get("class_counts", {})))
                state["slow_class_counts"] = dict(class_counts.get("slow_class_counts", {}))
                state["filtered_class_counts"] = dict(class_counts.get("filtered_class_counts", {}))
                state["axis"] = counts.get("axis")
                state["axis_ready"] = bool(counts.get("axis_ready", False))
                state["video_fps"] = float(video_fps or state.get("video_fps") or 30.0)
                state["backend"] = backend
            self._live_dual_dirty = True

        def run_detect():
            outputs = []
            streams = {}
            try:
                import main as yolo
                cwd = os.getcwd()
                os.chdir(str(self.PROJECT_ROOT))
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
                        output_path = str(self.TEST_OUTPUT_DIR / f"output_{direction}_{basename}.mp4")
                        fourcc_fn = getattr(cv2, "VideoWriter_fourcc")
                        fourcc = fourcc_fn(*"mp4v")
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
                            "direction_filter": yolo.TrajectoryDirectionFilter(),
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
                            filter_result = stream["direction_filter"].update(
                                boxes, class_ids, track_ids, stream["width"], stream["height"]
                            )
                            kept_indices = filter_result["kept_indices"]
                            kept_boxes = [boxes[i] for i in kept_indices]
                            kept_confidences = [confidences[i] for i in kept_indices]
                            kept_class_ids = [class_ids[i] for i in kept_indices]
                            kept_track_ids = [track_ids[i] for i in kept_indices]
                            kept_statuses = [
                                stream["direction_filter"].track_states.get(track_id, {}).get("status", "pending")
                                for track_id in kept_track_ids
                            ]
                            filter_info = stream["direction_filter"].get_filter_info(stream["width"], stream["height"])
                            frame_class_counts = {}
                            for cls_id in kept_class_ids:
                                class_name = yolo.CLASS_NAMES[cls_id] if 0 <= cls_id < len(yolo.CLASS_NAMES) else f"class_{cls_id}"
                                frame_class_counts[class_name] = frame_class_counts.get(class_name, 0) + 1

                            if kept_boxes:
                                result_frame = yolo.draw_detections(
                                    frame.copy(), kept_boxes, kept_confidences, kept_class_ids, yolo.CLASS_NAMES,
                                    track_ids=kept_track_ids, statuses=kept_statuses
                                )
                            else:
                                result_frame = frame.copy()
                            result_frame = yolo.draw_counting_line(
                                result_frame,
                                filter_info,
                                filter_result["track_count_total"],
                                filter_result["track_count_keep"],
                                filter_result["track_count_filtered"],
                                filter_result.get("track_count_slow", 0),
                            )
                            elapsed = time.time() - step_start
                            avg_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                            stream["fps_list"].append(avg_fps)
                            if len(stream["fps_list"]) > 30:
                                stream["fps_list"].pop(0)
                            avg_fps = sum(stream["fps_list"]) / len(stream["fps_list"])
                            result_frame = yolo.draw_info_panel(
                                result_frame,
                                [
                                    f"{direction} FPS: {avg_fps:.1f}",
                                    f"Valid: {len(kept_boxes)}",
                                    f"Raw: {len(boxes)}",
                                ],
                                origin=(10, 10),
                            )

                            if stream["writer"]:
                                stream["writer"].write(result_frame)

                            stream["frame_count"] += 1
                            stream["total_detections"] += len(kept_boxes)
                            on_detect_frame(
                                direction,
                                result_frame,
                                stream["frame_count"],
                                avg_fps,
                                len(kept_boxes),
                                stream["fps"],
                                {
                                    "track_count_total": filter_result["track_count_total"],
                                    "track_count_keep": filter_result["track_count_keep"],
                                    "track_count_slow": filter_result.get("track_count_slow", 0),
                                    "track_count_filtered": filter_result["track_count_filtered"],
                                    "axis": filter_result["axis"],
                                    "axis_ready": filter_result["axis_ready"],
                                },
                                {
                                    "crossed_class_counts": filter_result["kept_class_counts"],
                                    "slow_class_counts": filter_result["slow_class_counts"],
                                    "filtered_class_counts": filter_result["filtered_class_counts"],
                                    "frame_class_counts": frame_class_counts,
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
                            "count_method": "trajectory_direction_filter",
                            "track_count_total": stream["direction_filter"].total_count,
                            "track_count_keep": stream["direction_filter"].count_in,
                            "track_count_slow": stream["direction_filter"].current_slow_count,
                            "track_count_filtered": stream["direction_filter"].count_out,
                            "line_count_total": stream["direction_filter"].total_count,
                            "line_count_in": stream["direction_filter"].count_in,
                            "line_count_slow": stream["direction_filter"].current_slow_count,
                            "line_count_out": stream["direction_filter"].count_out,
                            "crossed_class_counts": stream["direction_filter"].crossed_class_counts,
                            "slow_class_counts": stream["direction_filter"].slow_class_counts,
                            "filtered_class_counts": stream["direction_filter"].filtered_class_counts,
                            "class_counts": stream["direction_filter"].crossed_class_counts,
                            "filter_info": filter_info,
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

    def _check_detect_status(self: _DetectControllerHost):
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
            if self.cycle_info.toPlainText():
                self.cycle_info.append("\n检测结束，实时联动停止")
            self._load_sessions()
            if pair_session:
                self.status_label.setText(f"已生成双方向会话: {Path(pair_session).name}")
