"""Detection orchestration logic for MainWindow."""

from __future__ import annotations
import json, os, threading, time
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
    _detect_latest: dict
    _detect_dirty: bool
    _live_dual_lock: threading.Lock
    _live_dual_state: dict
    _new_live_direction_state: Any
    _live_dual_dirty: bool
    _sim_live_mode: bool

    def _start_live_sim(self) -> None: ...
    def _reset_live_dual_state(self, video_pairs=None) -> None: ...
    def _load_video(self, path) -> bool: ...
    def _backend_label(self, backend) -> str: ...


def _snapshot(df, w, h):
    info = df.get_filter_info(w, h)
    return {
        "events": [],
        "track_count_total": df.total_count,
        "track_count_keep": df.current_keep_count,
        "track_count_slow": df.current_slow_count,
        "track_count_filtered": df.current_filtered_count,
        "filtered_class_counts": dict(df.filtered_class_counts),
        "slow_class_counts": dict(df.slow_class_counts),
        "kept_class_counts": dict(df.crossed_class_counts),
        "axis": info.get("axis"),
        "anchor": info.get("anchor"),
        "angle_threshold_deg": info.get("angle_threshold_deg"),
        "axis_ready": info.get("axis_ready", False),
    }


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    u = aa + bb - inter
    return inter / u if u > 0 else 0


def _dedupe(boxes, confs, clss, tids, stats, th=0.65):
    keep = []
    for i in sorted(
        range(len(boxes)), key=lambda k: confs[k] if k < len(confs) else 0, reverse=True
    ):
        if any(
            (clss[i] == clss[j] or tids[i] == tids[j]) and _iou(boxes[i], boxes[j]) >= th
            for j in keep
        ):
            continue
        keep.append(i)
    keep.sort()
    return (
        [boxes[i] for i in keep],
        [confs[i] for i in keep],
        [clss[i] for i in keep],
        [tids[i] for i in keep],
        [stats[i] for i in keep],
    )


def _cache_tuple(cache):
    if not cache:
        return [], [], [], [], []
    return (
        cache["boxes"][:],
        cache["confs"][:],
        cache["clss"][:],
        cache["tids"][:],
        cache["stats"][:],
    )


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
        model_onnx = self.PROJECT_ROOT / "public" / "yolo-v26" / "yolo26n.onnx"
        model_xml = self.PROJECT_ROOT / "public" / "yolo-v26" / "ir_model" / "yolo26n.xml"
        model_bin = self.PROJECT_ROOT / "public" / "yolo-v26" / "ir_model" / "yolo26n.bin"
        if not (model_onnx.exists() or (model_xml.exists() and model_bin.exists())):
            self.detect_status.setText("错误: YOLOv26 模型文件不存在（需 ONNX 或 OpenVINO IR）")
            self.detect_status.setStyleSheet(f"color: {C_RED.name()}; font-size: 12px;")
            return
        os.makedirs(str(self.TEST_OUTPUT_DIR), exist_ok=True)
        self.detecting = True
        self.detect_progress = None
        self.btn_detect.setEnabled(False)
        self.detect_status.setText("检测中... 0/2")
        self.detect_status.setStyleSheet(f"color: {C_PRIMARY.name()}; font-size: 12px;")
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

        def push(direction, frame, idx, avg_fps, n, video_fps, counts, classes, backend):
            with self._live_dual_lock:
                st = self._live_dual_state.setdefault(direction, self._new_live_direction_state())
                st["frame_bgr"] = frame
                st["frame_idx"] = idx
                st["fps"] = avg_fps
                st["num_objects"] = n
                st["track_count_total"] = int(counts.get("track_count_total", 0))
                st["track_count_keep"] = int(counts.get("track_count_keep", 0))
                st["track_count_slow"] = int(counts.get("track_count_slow", 0))
                st["track_count_filtered"] = int(counts.get("track_count_filtered", 0))
                st["line_count_total"] = st["track_count_total"]
                st["line_count_in"] = st["track_count_keep"]
                st["line_count_slow"] = st["track_count_slow"]
                st["line_count_out"] = st["track_count_filtered"]
                st["frame_class_counts"] = dict(classes.get("frame_class_counts", {}))
                st["class_counts"] = dict(classes.get("crossed_class_counts", {}))
                st["slow_class_counts"] = dict(classes.get("slow_class_counts", {}))
                st["filtered_class_counts"] = dict(classes.get("filtered_class_counts", {}))
                st["axis"] = counts.get("axis")
                st["axis_ready"] = bool(counts.get("axis_ready", False))
                st["video_fps"] = float(video_fps or 30.0)
                st["backend"] = backend
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
                    skip = max(1, int(getattr(yolo, "SKIP_FRAMES", 1)))
                    for direction, path in video_pairs:
                        cap = cv2.VideoCapture(path)
                        if not cap.isOpened():
                            raise FileNotFoundError(f"无法打开视频源: {path}")
                        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                        out = str(
                            self.TEST_OUTPUT_DIR / f"output_{direction}_{Path(path).stem}.mp4"
                        )
                        writer = cv2.VideoWriter(
                            out, cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps), (w, h)
                        )
                        streams[direction] = {
                            "source_path": path,
                            "output_path": out,
                            "cap": cap,
                            "writer": writer,
                            "fps": max(1.0, fps),
                            "width": w,
                            "height": h,
                            "frame_count": 0,
                            "fps_list": [],
                            "total_detections": 0,
                            "tracker": yolo.SimpleTracker(iou_threshold=0.2, max_lost=45),
                            "direction_filter": yolo.TrajectoryDirectionFilter(),
                            "cache": None,
                            "cache_age": 999,
                            "hold": skip * 3,
                            "done": False,
                            "next_due": 0.0,
                            "backend": backend,
                        }
                        with self._live_dual_lock:
                            st = self._live_dual_state.setdefault(
                                direction, self._new_live_direction_state()
                            )
                            st["source"] = path
                            st["video_fps"] = max(1.0, fps)
                            st["backend"] = backend
                    snap_dir = (
                        self.DATA_DIR
                        / f"detection_pair_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{Path(video_pairs[0][1]).stem[:16]}_{Path(video_pairs[1][1]).stem[:16]}"
                    )
                    snap_dir.mkdir(parents=True, exist_ok=True)
                    start = time.time()
                    filter_info = None
                    while True:
                        while self.sim_paused and not all(s["done"] for s in streams.values()):
                            time.sleep(0.05)
                        if all(s["done"] for s in streams.values()):
                            break
                        pending = False
                        clock = time.time() - start
                        for direction, s in streams.items():
                            if s["done"]:
                                continue
                            pending = True
                            if clock + 1e-6 < s["next_due"]:
                                continue
                            ret, frame = s["cap"].read()
                            if not ret:
                                s["done"] = True
                                with self._live_dual_lock:
                                    self._live_dual_state[direction]["done"] = True
                                self._live_dual_dirty = True
                                continue
                            s["next_due"] += 1.0 / max(s["fps"], 1.0)
                            fno = s["frame_count"]
                            t0 = time.time()
                            boxes, confs, clss = [], [], []
                            draw_boxes, draw_confs, draw_clss, draw_tids, draw_statuses = [], [], [], [], []
                            if fno % skip == 0:
                                boxes, confs, clss = yolo.process_frame(frame, detector)
                                tids = s["tracker"].update(boxes, clss)
                                if boxes:
                                    fr = s["direction_filter"].update(
                                        boxes, clss, tids, s["width"], s["height"]
                                    )
                                    for i, box in enumerate(boxes):
                                        if i >= len(tids):
                                            continue
                                        tid = int(tids[i])
                                        status = (
                                            s["direction_filter"]
                                            .track_states.get(tid, {})
                                            .get("status", "pending")
                                        )
                                        draw_boxes.append(box)
                                        draw_confs.append(confs[i] if i < len(confs) else 0.0)
                                        draw_clss.append(clss[i])
                                        draw_tids.append(tid)
                                        draw_statuses.append(status)
                                    if draw_boxes:
                                        draw_boxes, draw_confs, draw_clss, draw_tids, draw_statuses = _dedupe(
                                            draw_boxes, draw_confs, draw_clss, draw_tids, draw_statuses
                                        )
                                        s["cache"] = {
                                            "boxes": draw_boxes[:],
                                            "confs": draw_confs[:],
                                            "clss": draw_clss[:],
                                            "tids": draw_tids[:],
                                            "stats": draw_statuses[:],
                                        }
                                        s["cache_age"] = 0
                                    else:
                                        s["cache"] = None
                                        s["cache_age"] = s["hold"] + 1
                                else:
                                    fr = _snapshot(s["direction_filter"], s["width"], s["height"])
                                    s["cache_age"] += 1
                            else:
                                fr = _snapshot(s["direction_filter"], s["width"], s["height"])
                                s["cache_age"] += 1
                            if not draw_boxes and s.get("cache") and s["cache_age"] <= s["hold"]:
                                draw_boxes, draw_confs, draw_clss, draw_tids, draw_statuses = _cache_tuple(s["cache"])
                            filter_info = s["direction_filter"].get_filter_info(
                                s["width"], s["height"]
                            )
                            live_valid_count = sum(
                                1 for status in draw_statuses if str(status).lower() != "reject"
                            )
                            frame_classes = {}
                            for c, status in zip(draw_clss, draw_statuses):
                                if str(status).lower() == "reject":
                                    continue
                                name = (
                                    yolo.CLASS_NAMES[c]
                                    if 0 <= c < len(yolo.CLASS_NAMES)
                                    else f"class_{c}"
                                )
                                frame_classes[name] = frame_classes.get(name, 0) + 1
                            result = (
                                yolo.draw_detections(
                                    frame.copy(),
                                    draw_boxes,
                                    draw_confs,
                                    draw_clss,
                                    yolo.CLASS_NAMES,
                                    track_ids=draw_tids,
                                    statuses=draw_statuses,
                                )
                                if draw_boxes
                                else frame.copy()
                            )
                            result = yolo.draw_counting_line(
                                result,
                                filter_info,
                                fr["track_count_total"],
                                fr["track_count_keep"],
                                fr["track_count_filtered"],
                                fr.get("track_count_slow", 0),
                            )
                            fps_now = 1.0 / max(time.time() - t0, 1e-6)
                            s["fps_list"].append(fps_now)
                            s["fps_list"] = s["fps_list"][-30:]
                            avg = sum(s["fps_list"]) / len(s["fps_list"])
                            result = yolo.draw_info_panel(
                                result,
                                [
                                    f"{direction} FPS: {avg:.1f}",
                                    f"Valid: {live_valid_count}",
                                    f"Raw: {len(boxes)}",
                                ],
                                origin=(10, 10),
                            )
                            if s["writer"]:
                                s["writer"].write(result)
                            s["frame_count"] += 1
                            s["total_detections"] += live_valid_count
                            push(
                                direction,
                                result,
                                s["frame_count"],
                                avg,
                                live_valid_count,
                                s["fps"],
                                {
                                    "track_count_total": fr["track_count_total"],
                                    "track_count_keep": fr["track_count_keep"],
                                    "track_count_slow": fr.get("track_count_slow", 0),
                                    "track_count_filtered": fr["track_count_filtered"],
                                    "axis": fr["axis"],
                                    "axis_ready": fr["axis_ready"],
                                },
                                {
                                    "crossed_class_counts": fr["kept_class_counts"],
                                    "slow_class_counts": fr["slow_class_counts"],
                                    "filtered_class_counts": fr["filtered_class_counts"],
                                    "frame_class_counts": frame_classes,
                                },
                                backend,
                            )
                        if not pending:
                            break
                        time.sleep(0.001)
                    for direction, s in streams.items():
                        df = s["direction_filter"]
                        avg = sum(s["fps_list"]) / len(s["fps_list"]) if s["fps_list"] else 0.0
                        summary = {
                            "source": s["source_path"],
                            "total_frames": s["frame_count"],
                            "avg_fps": round(avg, 1),
                            "total_detections": s["total_detections"],
                            "count_method": "trajectory_direction_filter",
                            "track_count_total": df.total_count,
                            "track_count_keep": df.count_in,
                            "track_count_slow": df.current_slow_count,
                            "track_count_filtered": df.count_out,
                            "line_count_total": df.total_count,
                            "line_count_in": df.count_in,
                            "line_count_slow": df.current_slow_count,
                            "line_count_out": df.count_out,
                            "crossed_class_counts": df.crossed_class_counts,
                            "slow_class_counts": df.slow_class_counts,
                            "filtered_class_counts": df.filtered_class_counts,
                            "class_counts": df.crossed_class_counts,
                            "filter_info": filter_info
                            or df.get_filter_info(s["width"], s["height"]),
                            "video_info": {
                                "width": s["width"],
                                "height": s["height"],
                                "fps": s["fps"],
                            },
                            "model": detector.get("model_path"),
                            "backend": s["backend"],
                        }
                        outputs.append(
                            {
                                "direction": direction,
                                "source_path": s["source_path"],
                                "output_path": s["output_path"],
                                "session_dir": str(snap_dir),
                                "summary": summary,
                            }
                        )
                    final = {
                        "session_type": "direction_pair",
                        "source": "same_intersection_xy_pair",
                        "description": "同一路口两段垂直方向监控视频的轨迹方向过滤统计",
                        "count_method": "trajectory_direction_filter_by_direction",
                        "direction_videos": {o["direction"]: o["summary"] for o in outputs},
                        "preview_output": outputs[-1]["output_path"] if outputs else None,
                    }
                    with open(snap_dir / "summary.json", "w", encoding="utf-8") as f:
                        json.dump(final, f, ensure_ascii=False, indent=2)
                finally:
                    os.chdir(cwd)
                self.detect_progress = {
                    "status": "done",
                    "outputs": outputs,
                    "pair_session": str(snap_dir),
                }
            except Exception as e:
                import traceback

                self.detect_progress = {
                    "status": "fail",
                    "message": f"{str(e)[:200]}\n{traceback.format_exc()[:300]}",
                }
            finally:
                for s in streams.values():
                    try:
                        if s.get("cap"):
                            s["cap"].release()
                    except Exception:
                        pass
                    try:
                        if s.get("writer"):
                            s["writer"].release()
                    except Exception:
                        pass
                self.detecting = False

        threading.Thread(target=run_detect, daemon=True).start()

    def _check_detect_status(self: _DetectControllerHost):
        prog = self.detect_progress
        if self.detecting and isinstance(prog, dict) and prog.get("status") == "running":
            self.detect_status.setText(
                f"检测中... {prog.get('direction','?')}方向 {prog.get('current',0)}/{prog.get('total',0)}"
            )
            self.detect_status.setStyleSheet(f"color: {C_PRIMARY.name()}; font-size: 12px;")
            self.detect_progress = None
            return
        if not self.detecting and prog:
            self.detect_progress = None
            self.btn_detect.setEnabled(True)
            self._detect_latest = {"bgr": None, "fps": 0.0, "count": 0, "idx": 0, "video_fps": None}
            self._detect_dirty = False
            if isinstance(prog, dict) and prog.get("status") == "fail":
                self.detect_status.setText(prog.get("message", "检测失败"))
                self.detect_status.setStyleSheet(f"color: {C_RED.name()}; font-size: 12px;")
                self.sim_running = False
                self._sim_live_mode = False
            else:
                outputs = prog.get("outputs", []) if isinstance(prog, dict) else []
                names = [
                    f"{str(o.get('direction','?')).upper()}:{self._backend_label((o.get('summary') or {}).get('backend'))}"
                    for o in outputs
                ]
                backend_text = f"  后端: {' / '.join(names)}" if names else ""
                last = outputs[-1]["output_path"] if outputs else None
                if last and self._load_video(last):
                    self.btn_play.setText("⏸ 暂停")
                    self.detect_status.setText(f"播放: {Path(last).name}{backend_text}")
                    self.detect_status.setStyleSheet(f"color: {C_GREEN.name()}; font-size: 12px;")
                else:
                    self.detect_status.setText(f"检测完成，但结果视频无法播放{backend_text}")
                    self.detect_status.setStyleSheet(f"color: {C_ORANGE.name()}; font-size: 12px;")
                self.sim_running = False
                self._sim_live_mode = False
            if self.cycle_info.toPlainText():
                self.cycle_info.append("\n检测结束，实时联动停止")
