"""
YOLO 检测数据 → 交通特征提取层

从 frames.json 的原始检测框中提取控制算法需要的特征:
  - X/Y 路归属: 自动从 track 位移向量中发现两条主方向（PCA/角度聚类）
  - 排队/通行分类: 基于 track 帧间位移
  - 等待时间: track 连续处于"排队"状态的时长
  - 到达率: 新 track 出现速率 (EMA 平滑)

自动方向发现:
  十字路口车辆主要在两条互相垂直的方向上运动。
  收集所有长 track 的位移向量，通过角度聚类找到两个峰值方向，
  无需人工标定，适用于任意摄像机角度。

  两阶段:
    预热阶段: 积累 track，收集位移向量
    激活阶段: 完成方向标定后，所有新 track 按最近方向归入 X(0°) 或 Y(90°)
"""

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike", "bicycle"}

# 预热帧数: 积累到此数量后执行方向标定
WARMUP_FRAMES = 150
# 最小 track 长度 (帧数), 不足则方向不可靠, 不参与标定
MIN_TRACK_LEN_FOR_CALIB = 15


@dataclass
class TrackState:
    track_id: int
    class_name: str
    frames: List[Tuple[int, float, float, float]] = field(default_factory=list)

    # 0=未判定, 1=X路, 2=Y路
    direction: int = 0
    direction_confidence: float = 0.0

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
        # ── 自动标定 ──
        warmup_frames: int = WARMUP_FRAMES,
        min_track_len_for_calib: int = MIN_TRACK_LEN_FOR_CALIB,
        min_total_displacement: float = 30.0,  # 最小位移 (px), 过滤静止 track
        # ── 排队判定 ──
        queue_speed_px_per_frame: float = 2.0,
        gap_seconds: float = 3.0,
        arrival_ema_alpha: float = 0.1,
    ):
        self.fps = fps
        self.warmup_frames = warmup_frames
        self.min_track_len_for_calib = min_track_len_for_calib
        self.min_total_displacement = min_total_displacement

        self.queue_speed_threshold = queue_speed_px_per_frame
        self.gap_seconds = gap_seconds
        self.arrival_alpha = arrival_ema_alpha

        # ── 标定状态 ──
        self._calibrated: bool = False
        # 两条主方向的单位向量: direction_x (归一化), direction_y (归一化)
        # X路方向归一化向量, Y路方向归一化向量 (在图像坐标系中)
        self._dir_x_vec: Optional[np.ndarray] = None
        self._dir_y_vec: Optional[np.ndarray] = None
        # 标定用的位移向量缓存: [(dx, dy), ...]
        self._displacement_buffer: List[Tuple[float, float]] = []

        # 已完成方向判定的 track 及其方向 (用于重分类)
        self._track_directions: Dict[int, int] = {}

        self._tracks: Dict[int, TrackState] = {}
        self._processed_frames: int = 0
        self._sim_time: float = 0.0

        self._arrival_count_x: float = 0.0
        self._arrival_count_y: float = 0.0

        self._last_queue_x: int = 0
        self._last_queue_y: int = 0
        self._last_wait_x: float = 0.0
        self._last_wait_y: float = 0.0
        self._last_arrival_x: float = 0.0
        self._last_arrival_y: float = 0.0
        self._last_frame_idx: int = -1

        self._gap_x_frames: int = 0
        self._gap_y_frames: int = 0

    # ── 主接口 ──────────────────────────────────────────

    def process_frame(self, frame_data: dict) -> dict:
        frame_idx = frame_data["frame"]
        self._processed_frames += 1
        self._sim_time = frame_idx / max(self.fps, 0.1)

        active_tracks = self._update_tracks(frame_data)

        # ── 标定检查 ──
        if not self._calibrated and self._processed_frames >= self.warmup_frames:
            self._run_calibration()

        # ── 重分类已有 track ──
        if self._calibrated:
            self._reclassify_all_tracks()

        # ── 统计 ──
        queue_x, queue_y = 0, 0
        wait_x, wait_y = 0.0, 0.0
        x_active, y_active = False, False

        for ts in self._tracks.values():
            if ts.track_id not in active_tracks:
                continue
            if ts.class_name.lower() not in VEHICLE_CLASSES:
                continue
            if not ts.queued:
                continue
            if ts.direction == 0:
                continue

            if ts.direction == 1:
                queue_x += 1
                wait_x += ts.queued_seconds
                x_active = True
            elif ts.direction == 2:
                queue_y += 1
                wait_y += ts.queued_seconds
                y_active = True

        self._gap_x_frames = 0 if x_active else self._gap_x_frames + 1
        self._gap_y_frames = 0 if y_active else self._gap_y_frames + 1
        gap_x = self._gap_x_frames / max(self.fps, 0.1)
        gap_y = self._gap_y_frames / max(self.fps, 0.1)

        new_x = sum(1 for t in self._tracks.values()
                    if t.track_id in active_tracks
                    and t.class_name.lower() in VEHICLE_CLASSES
                    and len(t.frames) == 1
                    and t.direction == 1)
        new_y = sum(1 for t in self._tracks.values()
                    if t.track_id in active_tracks
                    and t.class_name.lower() in VEHICLE_CLASSES
                    and len(t.frames) == 1
                    and t.direction == 2)

        self._arrival_count_x = (self.arrival_alpha * new_x +
                                 (1 - self.arrival_alpha) * self._arrival_count_x)
        self._arrival_count_y = (self.arrival_alpha * new_y +
                                 (1 - self.arrival_alpha) * self._arrival_count_y)
        arrival_x = self._arrival_count_x * self.fps
        arrival_y = self._arrival_count_y * self.fps

        self._last_queue_x = queue_x
        self._last_queue_y = queue_y
        self._last_wait_x = wait_x
        self._last_wait_y = wait_y
        self._last_arrival_x = arrival_x
        self._last_arrival_y = arrival_y
        self._last_frame_idx = frame_idx

        return {
            "queue_x": queue_x,
            "queue_y": queue_y,
            "wait_x": wait_x,
            "wait_y": wait_y,
            "arrival_x": arrival_x,
            "arrival_y": arrival_y,
            "gap_x": gap_x,
            "gap_y": gap_y,
            "total_vehicles": len(active_tracks),
            "calibrated": self._calibrated,
        }

    def get_features(self) -> dict:
        return {
            "queue_x": self._last_queue_x,
            "queue_y": self._last_queue_y,
            "wait_x": self._last_wait_x,
            "wait_y": self._last_wait_y,
            "arrival_x": self._last_arrival_x,
            "arrival_y": self._last_arrival_y,
            "gap_x": self._gap_x_frames / max(self.fps, 0.1),
            "gap_y": self._gap_y_frames / max(self.fps, 0.1),
            "calibrated": self._calibrated,
        }

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def reset(self):
        self._tracks.clear()
        self._displacement_buffer.clear()
        self._track_directions.clear()
        self._calibrated = False
        self._dir_x_vec = None
        self._dir_y_vec = None
        self._processed_frames = 0
        self._sim_time = 0.0
        self._arrival_count_x = 0.0
        self._arrival_count_y = 0.0
        self._gap_x_frames = 0
        self._gap_y_frames = 0

    # ── 自动标定 ────────────────────────────────────────

    def _run_calibration(self):
        """从采集的位移向量中自动发现两条主方向"""
        if len(self._displacement_buffer) < 5:
            return  # 数据不足

        # 1. 计算每个位移向量的角度
        vecs = np.array(self._displacement_buffer, dtype=np.float64)
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])  # [-π, π]
        angles_deg = np.degrees(angles) % 180  # 方向模 180°（正反向视为同一路线）

        # 2. 角度直方图 + 找两个峰值
        hist, bins = np.histogram(angles_deg, bins=36, range=(0, 180))
        # 平滑
        hist_smooth = np.convolve(hist, [0.3, 0.4, 0.3], mode='same')

        # 找两个最高峰（不能太近，至少间隔 30°）
        peak_indices = np.argsort(hist_smooth)[::-1]
        peaks = []
        for idx in peak_indices:
            angle = (bins[idx] + bins[idx + 1]) / 2
            if all(abs(angle - p) > 30 and abs(angle - p) < 150 for p in peaks):
                peaks.append(angle)
            if len(peaks) >= 2:
                break

        if len(peaks) < 2:
            # 只有一个明显的主方向 → 假设正交
            p0 = peaks[0] if peaks else 45.0
            peaks = [p0, (p0 + 90) % 180]

        # 3. 确定 X/Y: 哪个角度更接近 0° (水平) 就是 X 路
        peaks.sort()

        def dist_to_horizontal(a):
            return min(abs(a - 0), abs(a - 180), abs(a - 180))

        if dist_to_horizontal(peaks[0]) < dist_to_horizontal(peaks[1]):
            x_angle_deg = peaks[0]
            y_angle_deg = peaks[1]
        else:
            x_angle_deg = peaks[1]
            y_angle_deg = peaks[0]

        # 4. 转为归一化方向向量
        x_rad = math.radians(x_angle_deg)
        y_rad = math.radians(y_angle_deg)

        self._dir_x_vec = np.array([math.cos(x_rad), math.sin(x_rad)], dtype=np.float64)
        self._dir_y_vec = np.array([math.cos(y_rad), math.sin(y_rad)], dtype=np.float64)

        # 5. 确保 X 和 Y 近似正交（修正）
        dot = abs(np.dot(self._dir_x_vec, self._dir_y_vec))
        if dot > 0.5:
            # 不正交 → 用 X + 90° 生成 Y
            self._dir_y_vec = np.array([-self._dir_x_vec[1], self._dir_x_vec[0]])

        self._calibrated = True

        print(f"[标定完成] X路方向: {x_angle_deg:.0f}°, Y路方向: {y_angle_deg:.0f}°")
        print(f"            基于 {len(self._displacement_buffer)} 个位移向量")

    def _classify_by_angle(self, dx: float, dy: float) -> Tuple[int, float]:
        """根据标定好的方向向量，判断位移向量属于 X 还是 Y"""
        if self._dir_x_vec is None or self._dir_y_vec is None:
            return 0, 0.0

        v = np.array([dx, dy], dtype=np.float64)
        norm = np.linalg.norm(v)
        if norm < 0.01:
            return 0, 0.0

        v_norm = v / norm

        # 与 X 方向的相似度（取绝对值，因为正反方向算同一路线）
        sim_x = abs(np.dot(v_norm, self._dir_x_vec))
        sim_y = abs(np.dot(v_norm, self._dir_y_vec))

        # 需要明显偏向某一边
        if sim_x > sim_y and sim_x > 0.5:
            return 1, float(sim_x)
        elif sim_y > sim_x and sim_y > 0.5:
            return 2, float(sim_y)
        return 0, 0.0

    def _reclassify_all_tracks(self):
        """标定完成后，用新的方向向量重新判定所有已有 track"""
        for ts in self._tracks.values():
            if ts.direction != 0:
                continue  # 已分类，跳过
            if len(ts.frames) < 3:
                continue

            first_cx = ts.frames[0][1]
            first_cy = ts.frames[0][2]
            last = ts.frames[-1]
            dx = last[1] - first_cx
            dy = last[2] - first_cy

            ts.direction, ts.direction_confidence = self._classify_by_angle(dx, dy)

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

            # 帧间位移
            disp = math.sqrt((cx - ts._last_cx) ** 2 + (cy - ts._last_cy) ** 2)

            # EMA 平滑速度
            ts._speed_ema = 0.3 * disp + 0.7 * ts._speed_ema

            # ── 标定前: 收集位移向量（仅已完成的 track） ──
            if not self._calibrated and len(ts.frames) >= self.min_track_len_for_calib:
                # 每个 track 只贡献一次（结束时）
                first_cx = ts.frames[0][1]
                first_cy = ts.frames[0][2]
                total_disp = math.sqrt((cx - first_cx)**2 + (cy - first_cy)**2)
                if total_disp > self.min_total_displacement:
                    self._displacement_buffer.append((cx - first_cx, cy - first_cy))

            # ── 标定后或标定前: 个体分类 ──
            if ts.direction == 0 and len(ts.frames) >= 3:
                first_cx = ts.frames[0][1]
                first_cy = ts.frames[0][2]

                if self._calibrated:
                    ts.direction, ts.direction_confidence = self._classify_by_angle(
                        cx - first_cx, cy - first_cy
                    )
                else:
                    # 标定前: 用临时角度，后期会在 _reclassify 中更新
                    dx = cx - first_cx
                    dy = cy - first_cy
                    total = abs(dx) + abs(dy)
                    if total > self.min_total_displacement:
                        if abs(dx) > 2.0 * abs(dy):
                            ts.direction, ts.direction_confidence = 1, min(1.0, abs(dx)/max(total, 0.1))
                        elif abs(dy) > 2.0 * abs(dx):
                            ts.direction, ts.direction_confidence = 2, min(1.0, abs(dy)/max(total, 0.1))

            # 排队判定
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
