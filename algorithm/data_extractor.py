"""
YOLO 检测数据 → 交通特征提取层

从 frames.json 的原始检测框中提取控制算法需要的特征:
  - 排队/通行分类: 基于 track 帧间位移

双视频模式下，X/Y 路由视频来源决定，本模块不再负责方向标定。
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

VEHICLE_CLASSES = {"car", "van", "bus", "truck"}


@dataclass
class TrackState:
    track_id: int
    class_name: str
    frames: List[Tuple[int, float, float, float]] = field(default_factory=list)

    queued: bool = False
    queued_seconds: float = 0.0
    _consecutive_queued_frames: int = 0
    _last_cx: float = 0.0
    _last_cy: float = 0.0
    _speed_ema: float = 0.0


class FrameFeatures:
    """从检测数据实时提取的每帧特征"""

    def __init__(
        self,
        fps: float = 30.0,
        queue_speed_px_per_frame: float = 2.0,
        arrival_ema_alpha: float = 0.1,
    ):
        self.fps = fps
        self.queue_speed_threshold = queue_speed_px_per_frame
        self.arrival_alpha = arrival_ema_alpha

        self._tracks: Dict[int, TrackState] = {}
        self._processed_frames: int = 0
        self._sim_time: float = 0.0

        self._arrival_count: float = 0.0

        self._last_queue: int = 0
        self._last_arrival: float = 0.0
        self._last_frame_idx: int = -1

    # ── 主接口 ──────────────────────────────────────────

    def process_frame(self, frame_data: dict) -> dict:
        frame_idx = frame_data["frame"]
        self._processed_frames += 1
        self._sim_time = frame_idx / max(self.fps, 0.1)

        active_tracks = self._update_tracks(frame_data)

        queue = 0

        for ts in self._tracks.values():
            if ts.track_id not in active_tracks:
                continue
            if ts.class_name.lower() not in VEHICLE_CLASSES:
                continue
            if ts.queued:
                queue += 1

        new_count = sum(1 for t in self._tracks.values()
                        if t.track_id in active_tracks
                        and t.class_name.lower() in VEHICLE_CLASSES
                        and len(t.frames) == 1)

        self._arrival_count = (self.arrival_alpha * new_count +
                               (1 - self.arrival_alpha) * self._arrival_count)
        arrival = self._arrival_count * self.fps

        self._last_queue = queue
        self._last_arrival = arrival
        self._last_frame_idx = frame_idx

        return {
            "queue": queue,
            "arrival": arrival,
            "total_vehicles": len(active_tracks),
        }

    def get_features(self) -> dict:
        return {
            "queue": self._last_queue,
            "arrival": self._last_arrival,
        }

    def reset(self):
        self._tracks.clear()
        self._processed_frames = 0
        self._sim_time = 0.0
        self._arrival_count = 0.0

    # ── track 更新 ────────────────────────────────────

    def _update_tracks(self, frame_data: dict) -> set:
        active = set()

        for det in frame_data.get("detections", []):
            tid = det["track_id"]
            cx = (det["x1"] + det["x2"]) / 2.0
            cy = (det["y1"] + det["y2"]) / 2.0
            bw = abs(det["x2"] - det["x1"])
            active.add(tid)

            if tid not in self._tracks:
                self._tracks[tid] = TrackState(
                    track_id=tid,
                    class_name=det["class"],
                )
                self._tracks[tid]._last_cx = cx
                self._tracks[tid]._last_cy = cy

            ts = self._tracks[tid]
            ts.frames.append((frame_data["frame"], cx, cy, bw))

            disp = math.sqrt((cx - ts._last_cx) ** 2 + (cy - ts._last_cy) ** 2)

            ts._speed_ema = 0.3 * disp + 0.7 * ts._speed_ema

            is_stopped = ts._speed_ema < self.queue_speed_threshold
            if is_stopped:
                ts._consecutive_queued_frames += 1
                if ts._consecutive_queued_frames >= 2:
                    ts.queued = True
            else:
                ts._consecutive_queued_frames = 0
                ts.queued = False

            if ts.queued:
                ts.queued_seconds += 1.0 / self.fps

            ts._last_cx = cx
            ts._last_cy = cy

        return active


def load_frames(session_dir: str) -> Tuple[List[dict], float]:
    session_path = Path(session_dir)
    json_path = session_path / "frames.json"
    if not json_path.exists():
        raise FileNotFoundError(f"找不到帧数据: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        frames = json.load(f)

    summary_path = session_path / "summary.json"
    fps = 30.0
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        fps = summary.get("video_info", {}).get("fps", 30.0)

    return frames, fps
