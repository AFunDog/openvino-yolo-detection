"""Detection orchestration logic for MainWindow."""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
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
    PROJECT_ROOT: Path
    TEST_OUTPUT_DIR: Path
    DATA_DIR: Path
    btn_detect: Any
    btn_play: Any
    sim_running: bool
    sim_paused: bool
    last_tick: float
    data_source_combo: Any
    cycle_info: Any
    status_label: Any
    _detect_latest: dict
    _detect_dirty: bool
    _live_dual_lock: threading.Lock
    _live_dual_state: dict
    _new_live_direction_state: Any
    _live_dual_dirty: bool
    _sim_live_mode: bool

    def _start_live_sim(self) -> None: ...
    def _reset_live_dual_state(self, video_pairs=None) -> None: ...
    def _save_direction_pair_session(self, outputs) -> str: ...
    def _load_video(self, path) -> bool: ...
    def _backend_label(self, backend) -> str: ...


def _active_tracks_from_tracker(tracker):
    """Return tracker-maintained tracks that are still inside the lost-frame buffer."""
    boxes = []
    confidences = []
    class_ids = []
    track_ids = []
    max_lost = max(1, int(getattr(tracker, "max_lost", 1)))

    for track_id, track in getattr(tracker, "tracks", {}).items():
        lost = int(track.get("lost", 0))
        if lost > max_lost:
            continue
        boxes.append(track["box"])
        class_ids.append(int(track["class_id"]))
        track_ids.append(int(track_id))
        confidences.append(max(0.05, 1.0 - lost / max_lost))

    return boxes, confidences, class_ids, track_ids


def _snapshot_filter_result(direction_filter, width, height):
    """Build a filter result without advancing trajectory state on skipped frames."""
    info = direction_filter.get_filter_info(width, height)
    return {
        "events": [],
        "track_count_total": direction_filter.total_count,
        "track_count_keep": direction_filter.current_keep_count,
        "track_count_slow": direction_filter.current_slow_count,
        "track_count_filtered": direction_filter.current_filtered_count,
        "filtered_class_counts": dict(direction_filter.filtered_class_counts),
        "slow_class_counts": dict(direction_filter.slow_class_counts),
        "kept_class_counts": dict(direction_filter.crossed_class_counts),
        "axis": info.get("axis"),
        "anchor": info.get("anchor"),
        "angle_threshold_deg": info.get("angle_threshold_deg"),
        "axis_ready": info.get("axis_ready", False),
    }


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
                            "tracker": yolo.SimpleTracker(iou_threshold=0.2, max_lost=45),
                            "direction_filter": yolo.TrajectoryDirectionFilter(),
                            "done": False,
                            "next_due": 0.0,
                            "backend": backend,
                        }
                        with self._live_dual_lock:
                            state = self._live_dual_state.setdefault(direction, self._new_live_direction_state())
                            state["source"] = video_path
                            state["video_fps"] = max(1.0, fps)
                            state["backend"] = backend

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    x_name = Path(video_pairs[0][1]).stem[:16]
                    y_name = Path(video_pairs[1][1]).stem[:16]
                    snapshot_dir = self.DATA_DIR / f"detection_pair_{timestamp}_{x_name}_{y_name}"
                    snapshot_dir.mkdir(parents=True, exist_ok=True)
                    last_snapshot_time = 0.0
                    snapshot_interval = 1.0

                    start_ts = time.time()
                    while True:
                        while self.sim_paused:
                            time.sleep(0.05)
                            if all(s["done"] for s in streams.values()):
                                break
                        if all(s["done"] for s in streams.values()):
                            break

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
                                track_ids = stream["tracker"].update(boxes, class_ids)
                                filter_result = stream["direction_filter"].update(
                                    boxes, class_ids, track_ids, stream["width"], stream["height"]
                                )
                            else:
                                boxes, confidences, class_ids = [], [], []
                                stream["tracker"].update([], [])
                                filter_result = _snapshot_filter_result(
                                    stream["direction_filter"], stream["width"], stream["height"]
                                )

                            active_boxes, active_confidences, active_class_ids, active_track_ids = _active_tracks_from_tracker(
                                stream["tracker"]
                            )
                            display_boxes = []
                            display_confidences = []
                            display_class_ids = []
                            display_track_ids = []
                            display_statuses = []
                            for box, conf, cls_id, track_id in zip(
                                active_boxes, active_confidences, active_class_ids, active_track_ids
                            ):
                                status = stream["direction_filter"].track_states.get(track_id, {}).get("status", "pending")
                                if status == "reject":
                                    continue
                                display_boxes.append(box)
                                display_confidences.append(conf)
                                display_class_ids.append(cls_id)
                                display_track_ids.append(track_id)
                                display_statuses.append(status)

                            filter_info = stream["direction_filter"].get_filter_info(stream["width"], stream["height"])
                            frame_class_counts = {}
                            for cls_id in display_class_ids:
                                class_name = yolo.CLASS_NAMES[cls_id] if 0 <= cls_id < len(yolo.CLASS_NAMES) else f"class_{cls_id}"
                                frame_class_counts[class_name] = frame_class_counts.get(class_name, 0) + 1

                            if display_boxes:
                                result_frame = yolo.draw_detections(
                                    frame.copy(), display_boxes, display_confidences, display_class_ids, yolo.CLASS_NAMES,
                                    track_ids=display_track_ids, statuses=display_statuses
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
                                    f"Valid: {len(display_boxes)}",
                                    f"Raw: {len(boxes)}",
                                ],
                                origin=(10, 10),
                            )

                            if stream["writer"]:
                                stream["writer"].write(result_frame)

                            stream["frame_count"] += 1
                            stream["total_detections"] += len(display_boxes)
                            on_detect_frame(
                                direction,
                                result_frame,
                                stream["frame_count"],
                                avg_fps,
                                len(display_boxes),
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

                        now = time.time()
                        if now - last_snapshot_time >= snapshot_interval:
                            last_snapshot_time = now
                            try:
                                by_direction = {}
                                for d, st in streams.items():
                                    df = st["direction_filter"]
                                    by_direction[d] = {
                                        "source": st["source_path"],
                                        "output_path": st["output_path"],
                                        "track_count_total": df.total_count,
                                        "track_count_keep": df.count_in,
                                        "track_count_slow": df.current_slow_count,
                                        "track_count_filtered": df.count_out,
                                        "line_count_total": df.total_count,
                                        "line_count_in": df.count_in,
                                        "line_count_slow": df.current_slow_count,
                                        "line_count_out": df.count_out,
                                        "crossed_class_counts": dict(df.crossed_class_counts),
                                        "slow_class_counts": dict(df.slow_class_counts),
                                        "total_frames": st["frame_count"],
                                        "total_detections": st["total_detections"],
                                        "avg_fps": round(
                                            sum(st["fps_list"]) / len(st["fps_list"]) if st["fps_list"] else 0.0, 1
                                        ),
                                        "video_info": {
                                            "width": st["width"],
                                            "height": st["height"],
                                            "fps": st["fps"],
                                        },
                                        "backend": st["backend"],
                                    }
                                lx = by_direction.get("X", {}).get("track_count_total", 0)
                                ly = by_direction.get("Y", {}).get("track_count_total", 0)
                                snapshot = {
                                    "session_type": "direction_pair",
                                    "source": "same_intersection_xy_pair",
                                    "description": "同一路口两段垂直方向监控视频的轨迹方向过滤统计（增量快照）",
                                    "count_method": "trajectory_direction_filter_by_direction",
                                    "track_count_x": lx,
                                    "track_count_y": ly,
                                    "line_count_x": lx,
                                    "line_count_y": ly,
                                    "line_count_total": lx + ly,
                                    "direction_videos": by_direction,
                                    "preview_output": by_direction.get("Y", {}).get("output_path")
                                                      or by_direction.get("X", {}).get("output_path"),
                                }
                                snapshot_path = snapshot_dir / "summary.json"
                                tmp_path = snapshot_dir / "summary.json.tmp"
                                with open(tmp_path, "w", encoding="utf-8") as f:
                                    json.dump(snapshot, f, ensure_ascii=False, indent=2)
                                os.replace(tmp_path, snapshot_path)
                            except Exception:
                                pass

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
                            "session_dir": str(snapshot_dir),
                            "summary": summary,
                        })

                    by_direction_final = {}
                    for item in outputs:
                        d = str(item.get("direction", "")).upper()
                        s = item.get("summary", {}) or {}
                        by_direction_final[d] = {
                            "source": item.get("source_path"),
                            "session_dir": item.get("session_dir"),
                            "output_path": item.get("output_path"),
                            "track_count_total": s.get("track_count_total", s.get("line_count_total", 0)),
                            "track_count_keep": s.get("track_count_keep", s.get("line_count_in", 0)),
                            "track_count_slow": s.get("track_count_slow", s.get("line_count_slow", 0)),
                            "track_count_filtered": s.get("track_count_filtered", s.get("line_count_out", 0)),
                            "line_count_total": s.get("line_count_total", 0),
                            "line_count_in": s.get("line_count_in", 0),
                            "line_count_slow": s.get("line_count_slow", 0),
                            "line_count_out": s.get("line_count_out", 0),
                            "crossed_class_counts": s.get("crossed_class_counts", {}),
                            "slow_class_counts": s.get("slow_class_counts", {}),
                            "total_frames": s.get("total_frames", 0),
                            "total_detections": s.get("total_detections", 0),
                            "avg_fps": s.get("avg_fps", 0),
                            "video_info": s.get("video_info", {}),
                            "backend": s.get("backend", ""),
                        }
                    lx_final = int(by_direction_final.get("X", {}).get("track_count_total", 0))
                    ly_final = int(by_direction_final.get("Y", {}).get("track_count_total", 0))
                    final_summary = {
                        "session_type": "direction_pair",
                        "source": "same_intersection_xy_pair",
                        "description": "同一路口两段垂直方向监控视频的轨迹方向过滤统计",
                        "count_method": "trajectory_direction_filter_by_direction",
                        "track_count_x": lx_final,
                        "track_count_y": ly_final,
                        "line_count_x": lx_final,
                        "line_count_y": ly_final,
                        "line_count_total": lx_final + ly_final,
                        "direction_videos": by_direction_final,
                        "preview_output": by_direction_final.get("Y", {}).get("output_path")
                                          or by_direction_final.get("X", {}).get("output_path"),
                    }
                    snapshot_path = snapshot_dir / "summary.json"
                    tmp_path = snapshot_dir / "summary.json.tmp"
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        json.dump(final_summary, f, ensure_ascii=False, indent=2)
                    os.replace(tmp_path, snapshot_path)
                finally:
                    os.chdir(cwd)
                pair_session = str(snapshot_dir)
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