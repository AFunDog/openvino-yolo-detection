"""
YOLOv26 实现代码
默认使用 ONNX Runtime 加载导出的 best.onnx，
同时保留 OpenVINO IR 作为备用后端。
"""

from pathlib import Path
import math
import re

import openvino as ov
import onnxruntime as ort
import cv2
import numpy as np
import os
import time
import json
import csv
from datetime import datetime
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent

# ==================== 模型参数配置 ====================
# 模型文件路径
MODEL_ONNX_PATH = str(BASE_DIR / "public" / "yolo-v26" / "yolo26n.onnx")
MODEL_XML_PATH = str(BASE_DIR / "public" / "yolo-v26" / "ir_model" / "yolo26n.xml")

# 默认优先使用 ONNX 模型
MODEL_BACKEND = "onnxruntime"

# 设备选择: "CPU", "GPU", "NPU" 等
# 如果有独立显卡，改为 "GPU" 可显著提升速度
DEVICE = "CPU"  # 改为 "GPU" 如果有Intel集成显卡或独显

# 输入参数
INPUT_SIZE = 640  # 从640降低到320，速度提升约4倍，精度略降
INPUT_CHANNELS = 3

# 检测参数
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7
MAX_DETECTIONS = 300

# 性能优化参数
SKIP_FRAMES = 2  # 跳帧检测：每N帧检测一次（1=每帧检测，2=每隔1帧检测）

# 显示参数
DISPLAY_WINDOW_NAME = "YOLOv26 Detection"
DISPLAY_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
DISPLAY_LABEL_SCALE = 0.5
DISPLAY_LABEL_THICKNESS = 2
DISPLAY_BOX_THICKNESS = 2
DISPLAY_BOX_COLOR = (0, 255, 0)
DISPLAY_BOX_COLOR_KEEP = (0, 200, 0)
DISPLAY_BOX_COLOR_REJECT = (0, 0, 255)
DISPLAY_BOX_COLOR_SLOW = (0, 165, 255)
DISPLAY_BOX_COLOR_PENDING = (0, 165, 255)
DISPLAY_TEXT_COLOR = (0, 0, 0)

# COCO 类别标签 (80类)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# UA-DETRAC 类别标签
CLASS_NAMES = ['others', 'car', 'van', 'bus']


# 数据记录目录
DATA_DIR = "data"


def _safe_name(text):
    """将任意文本压缩为适合文件名的短标识。"""
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", str(text))
    cleaned = cleaned.strip("._-")
    return cleaned[:40] or "source"


@lru_cache(maxsize=4)
def _get_font(font_size):
    """加载中文字体，优先使用 Microsoft YaHei。"""
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for font_path in candidates:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, font_size)
            except Exception:
                continue
    return ImageFont.load_default()


def _draw_chinese_labels(img, labels):
    """使用 PIL 一次性绘制一组中文标签。"""
    if not labels:
        return img
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    font_size = max(12, int(18 * DISPLAY_LABEL_SCALE))
    font = _get_font(font_size)
    pad_x = 6
    pad_y = 4
    for x, y, label, box_color in labels:
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        bg_left = x
        bg_top = max(0, y - text_h - pad_y * 2)
        bg_right = min(pil_img.width, x + text_w + pad_x * 2)
        bg_bottom = min(pil_img.height, y)
        draw.rectangle([bg_left, bg_top, bg_right, bg_bottom], fill=tuple(int(v) for v in box_color))
        text_x = bg_left + pad_x
        text_y = bg_top + pad_y - bbox[1]
        draw.text((text_x, text_y), label, font=font, fill=DISPLAY_TEXT_COLOR)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_info_panel(img, lines, origin=(10, 10), text_color=(255, 0, 0), bg_color=(255, 255, 255), alpha=0.65):
    """在图像左上角绘制带底色的信息面板。"""
    if not lines:
        return img
    font = DISPLAY_LABEL_FONT
    font_scale = 0.8
    thickness = 2
    padding_x = 8
    padding_y = 6
    line_gap = 4
    widths = []
    heights = []
    for line in lines:
        (w, h), _ = cv2.getTextSize(str(line), font, font_scale, thickness)
        widths.append(w)
        heights.append(h)
    box_w = max(widths) + padding_x * 2
    box_h = sum(heights) + line_gap * max(0, len(lines) - 1) + padding_y * 2

    overlay = img.copy()
    x, y = origin
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    text_y = y + padding_y + heights[0]
    for idx, line in enumerate(lines):
        cv2.putText(img, str(line), (x + padding_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        if idx + 1 < len(lines):
            text_y += heights[idx] + line_gap

    return img


# ─── IoU 追踪器（跨帧去重） ─────────────────────────────

def _iou(box_a, box_b):
    """计算两个框的 IoU"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


class SimpleTracker:
    """基于 IoU 的简易目标追踪器，用于跨帧去重统计"""

    def __init__(self, iou_threshold=0.3, max_lost=5):
        self.next_id = 1
        self.tracks = {}        # track_id -> {"box": [...], "class_id": int, "lost": int}
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.unique_class_counts = {}  # 去重后的类别计数

    def update(self, boxes, class_ids):
        """更新追踪器，返回每帧每个检测的 track_id"""
        matched_old = set()  # 已匹配的旧轨迹 id
        matched_new = set()  # 已匹配的新检测索引

        # 构建匹配关系：旧轨迹 ↔ 新检测
        matches = []  # (track_id, new_idx)
        for tid, trk in self.tracks.items():
            best_iou = 0
            best_idx = -1
            for i, (box, cls) in enumerate(zip(boxes, class_ids)):
                if i in matched_new or cls != trk["class_id"]:
                    continue
                iou_val = _iou(box, trk["box"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = i
            if best_idx >= 0 and best_iou >= self.iou_threshold:
                matches.append((tid, best_idx))
                matched_old.add(tid)
                matched_new.add(best_idx)

        # 构建新轨迹字典
        new_tracks = {}

        # 更新已匹配的轨迹
        for tid, new_idx in matches:
            new_tracks[tid] = {"box": boxes[new_idx], "class_id": class_ids[new_idx], "lost": 0}

        # 保留短暂丢失的轨迹
        for tid, trk in self.tracks.items():
            if tid not in matched_old:
                trk["lost"] += 1
                if trk["lost"] <= self.max_lost:
                    new_tracks[tid] = trk

        # 未匹配的新检测 → 分配新 ID
        track_ids = [0] * len(boxes)
        for tid, new_idx in matches:
            track_ids[new_idx] = tid

        for i, (box, cls) in enumerate(zip(boxes, class_ids)):
            if i not in matched_new:
                tid = self.next_id
                self.next_id += 1
                new_tracks[tid] = {"box": box, "class_id": cls, "lost": 0}
                track_ids[i] = tid
                # 新目标计入去重统计
                class_name = CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else f"class_{cls}"
                self.unique_class_counts[class_name] = self.unique_class_counts.get(class_name, 0) + 1

        self.tracks = new_tracks
        return track_ids


class TrajectoryDirectionFilter:
    """基于 track_id 的轨迹方向过滤器。"""

    def __init__(self, min_displacement=4.0, angle_threshold_deg=75.0):
        self.min_displacement = min_displacement
        self.angle_threshold_deg = angle_threshold_deg
        self.min_similarity = math.cos(math.radians(angle_threshold_deg))
        self.track_states = {}   # track_id -> state
        self.axis_vec = np.array([0.0, 1.0], dtype=np.float32)
        self.anchor_point = None
        self.axis_ready = False

        # 兼容旧统计字段，但语义已改为“过滤后车辆”
        self.total_count = 0
        self.count_in = 0
        self.count_out = 0
        self.crossed_class_counts = {}
        self.slow_class_counts = {}
        self.filtered_class_counts = {}
        self.current_keep_count = 0
        self.current_slow_count = 0
        self.current_filtered_count = 0

    def _update_axis(self, frame_width, frame_height):
        vectors = []
        for state in self.track_states.values():
            start = state.get("start")
            last = state.get("last")
            if start is None or last is None:
                continue
            dx = last[0] - start[0]
            dy = last[1] - start[1]
            if math.hypot(dx, dy) >= self.min_displacement:
                vectors.append([dx, dy])

        if vectors:
            vecs = np.asarray(vectors, dtype=np.float32)
            cov = vecs.T @ vecs
            eigvals, eigvecs = np.linalg.eigh(cov)
            axis = eigvecs[:, int(np.argmax(eigvals))]
            norm = np.linalg.norm(axis)
            if norm > 1e-6:
                axis = axis / norm
                if axis[1] < 0:
                    axis = -axis
                self.axis_vec = axis.astype(np.float32)
                self.axis_ready = True

        self.anchor_point = np.array(
            [frame_width * 0.5, frame_height * 0.5],
            dtype=np.float32,
        )

    def _classify_track(self, state):
        start = state.get("start")
        last = state.get("last")
        prev = state.get("prev")
        hits = int(state.get("hits", 0))
        if start is None or last is None:
            return "pending", 0.0, 0.0

        if hits < 2:
            return "pending", 0.0, 0.0

        ref = prev if prev is not None else start
        dx = last[0] - ref[0]
        dy = last[1] - ref[1]
        moved = math.hypot(dx, dy)
        if moved < self.min_displacement:
            return "slow", 0.0, moved

        v = np.array([dx, dy], dtype=np.float32) / max(moved, 1e-6)
        similarity = float(np.dot(v, self.axis_vec))
        if similarity >= self.min_similarity:
            return "keep", similarity, moved
        if similarity <= -self.min_similarity:
            return "reject", similarity, moved
        return "reject", similarity, moved

    def update(self, boxes, class_ids, track_ids, frame_width, frame_height):
        self._update_axis(frame_width, frame_height)

        kept_indices = []
        kept_boxes = []
        kept_confidences = []
        kept_class_ids = []
        kept_track_ids = []
        events = []
        current_keep_count = 0
        current_slow_count = 0
        current_filtered_count = 0

        for idx, (box, cls_id, track_id) in enumerate(zip(boxes, class_ids, track_ids)):
            cx = float((box[0] + box[2]) / 2.0)
            cy = float((box[1] + box[3]) / 2.0)
            center = np.array([cx, cy], dtype=np.float32)

            state = self.track_states.setdefault(track_id, {
                "start": center.copy(),
                "prev": None,
                "last": center.copy(),
                "class_id": int(cls_id),
                "status": "pending",
                "similarity": 0.0,
                "moved": 0.0,
                "hits": 0,
                "counted_keep": False,
                "counted_slow": False,
                "counted_reject": False,
            })
            state["prev"] = state.get("last", center).copy()
            state["last"] = center.copy()
            state["class_id"] = int(cls_id)
            state["hits"] = int(state.get("hits", 0)) + 1

            status, similarity, moved = self._classify_track(state)
            state["status"] = status
            state["similarity"] = similarity
            state["moved"] = moved

            class_name = get_class_name(cls_id)
            if status == "keep":
                current_keep_count += 1
                kept_indices.append(idx)
                kept_boxes.append(box)
                kept_confidences.append(0.0)
                kept_class_ids.append(int(cls_id))
                kept_track_ids.append(int(track_id))
                if not state["counted_keep"]:
                    state["counted_keep"] = True
                    self.total_count += 1
                    self.count_in = self.total_count
                    self.crossed_class_counts[class_name] = (
                        self.crossed_class_counts.get(class_name, 0) + 1
                    )
                    events.append({
                        "track_id": int(track_id),
                        "class": class_name,
                        "class_id": int(cls_id),
                        "status": "keep",
                        "similarity": round(float(similarity), 4),
                        "moved": round(float(moved), 2),
                        "count_total": self.total_count,
                    })
            elif status == "slow":
                current_slow_count += 1
                kept_indices.append(idx)
                kept_boxes.append(box)
                kept_confidences.append(0.0)
                kept_class_ids.append(int(cls_id))
                kept_track_ids.append(int(track_id))
                if not state["counted_slow"]:
                    state["counted_slow"] = True
                    self.slow_class_counts[class_name] = (
                        self.slow_class_counts.get(class_name, 0) + 1
                    )
                    events.append({
                        "track_id": int(track_id),
                        "class": class_name,
                        "class_id": int(cls_id),
                        "status": "slow",
                        "similarity": round(float(similarity), 4),
                        "moved": round(float(moved), 2),
                    })
            elif status == "reject":
                current_filtered_count += 1
                if not state["counted_reject"]:
                    state["counted_reject"] = True
                    self.count_out += 1
                    self.filtered_class_counts[class_name] = (
                        self.filtered_class_counts.get(class_name, 0) + 1
                    )
                    events.append({
                        "track_id": int(track_id),
                        "class": class_name,
                        "class_id": int(cls_id),
                        "status": "reject",
                        "similarity": round(float(similarity), 4),
                        "moved": round(float(moved), 2),
                        "count_total": self.total_count,
                    })
            else:
                # pending 阶段：保留显示，但不参与主方向统计
                kept_indices.append(idx)
                kept_boxes.append(box)
                kept_confidences.append(0.0)
                kept_class_ids.append(int(cls_id))
                kept_track_ids.append(int(track_id))

        self.current_keep_count = current_keep_count
        self.current_slow_count = current_slow_count
        self.current_filtered_count = current_filtered_count

        return {
            "kept_indices": kept_indices,
            "kept_boxes": kept_boxes,
            "kept_confidences": kept_confidences,
            "kept_class_ids": kept_class_ids,
            "kept_track_ids": kept_track_ids,
            "events": events,
            "track_count_total": self.total_count,
            "track_count_keep": self.current_keep_count,
            "track_count_slow": self.current_slow_count,
            "track_count_filtered": self.current_filtered_count,
            "filtered_class_counts": dict(self.filtered_class_counts),
            "slow_class_counts": dict(self.slow_class_counts),
            "kept_class_counts": dict(self.crossed_class_counts),
            "axis": [float(self.axis_vec[0]), float(self.axis_vec[1])],
            "anchor": [float(self.anchor_point[0]), float(self.anchor_point[1])] if self.anchor_point is not None else None,
            "angle_threshold_deg": float(self.angle_threshold_deg),
            "axis_ready": self.axis_ready,
        }

    def get_line_info(self, frame_width, frame_height):
        if self.anchor_point is None:
            self._update_axis(frame_width, frame_height)
        point = self.anchor_point if self.anchor_point is not None else np.array(
            [frame_width * 0.5, frame_height * 0.5], dtype=np.float32
        )
        direction = self.axis_vec
        return {
            "axis": [float(direction[0]), float(direction[1])],
            "anchor": [float(point[0]), float(point[1])],
            "angle_threshold_deg": float(self.angle_threshold_deg),
            "axis_ready": self.axis_ready,
        }

    def get_filter_info(self, frame_width, frame_height):
        return self.get_line_info(frame_width, frame_height)

    def reset(self):
        self.track_states.clear()
        self.axis_vec = np.array([0.0, 1.0], dtype=np.float32)
        self.anchor_point = None
        self.axis_ready = False
        self.total_count = 0
        self.count_in = 0
        self.count_out = 0
        self.crossed_class_counts.clear()
        self.slow_class_counts.clear()
        self.filtered_class_counts.clear()
        self.current_keep_count = 0
        self.current_slow_count = 0
        self.current_filtered_count = 0

# ─── 数据保存 ──────────────────────────────────────────

def save_detection_data(source, frame_data, summary):
    """保存检测数据到 data 目录

    Args:
        source: 视频源标识
        frame_data: 帧级检测数据列表
        summary: 统计汇总信息
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    source_name = _safe_name(Path(str(source)).stem if str(source) else "source")
    session_dir = os.path.join(DATA_DIR, f"detection_{timestamp}_{source_name}")
    os.makedirs(session_dir, exist_ok=True)

    # 保存帧级 JSON 数据
    json_path = os.path.join(session_dir, "frames.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(frame_data, f, ensure_ascii=False, indent=2)

    # 保存帧级 CSV 数据
    csv_path = os.path.join(session_dir, "frames.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "timestamp", "class", "class_id", "confidence", "x1", "y1", "x2", "y2"])
        for frame in frame_data:
            for det in frame["detections"]:
                writer.writerow([
                    frame["frame"], frame["timestamp"],
                    det["class"], det["class_id"], det["confidence"],
                    det["x1"], det["y1"], det["x2"], det["y2"]
                ])

    # 保存统计汇总
    summary_path = os.path.join(session_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"数据已保存到: {session_dir}/")
    print(f"  - frames.json  (帧级详细数据)")
    print(f"  - frames.csv   (帧级表格数据)")
    print(f"  - summary.json (统计汇总)")
    return session_dir


def convert_boxes(x):
    """将中心点坐标 (cx, cy, w, h) 转换为 (x1, y1, x2, y2)"""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1 = cx - w/2
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1 = cy - h/2
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2 = cx + w/2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2 = cy + h/2
    return y


def nms(boxes, scores, iou_threshold):
    """非极大值抑制"""
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        scores.tolist(),
        CONF_THRESHOLD,
        iou_threshold
    )
    return indices.flatten() if len(indices) > 0 else []


def sigmoid(x):
    """Sigmoid 激活函数"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def dist2bbox(distance_points, anchor_points):
    """
    将距离预测转换为边界框坐标
    distance_points: (N, 4) - (l, t, r, b)
    anchor_points: (N, 2) - (cx, cy)
    """
    lt, rb = np.split(distance_points, 2, axis=-1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    return np.concatenate([x1y1, x2y2], axis=-1)


def get_class_name(class_id):
    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]
    return f"class_{class_id}"


def decode_yolov26_output(output, img_width, img_height):
    """
    解码 YOLOv26 输出

    YOLOv26 输出格式: (batch, 300, 6)
    每个候选框: [x1, y1, x2, y2, conf, class_id]
    """
    output = np.asarray(output)
    if output.ndim == 2:
        output = np.expand_dims(output, axis=0)
    print(f"输出shape: {output.shape}")

    batch, num_boxes, num_values = output.shape
    print(f"Batch: {batch}, Boxes: {num_boxes}, Values per box: {num_values}")

    boxes = []
    confidences = []
    class_ids = []

    # 遍历所有候选框
    for i in range(num_boxes):
        pred = output[0, i]  # (6,)

        # 提取边界框坐标 (x1, y1, x2, y2)
        x1, y1, x2, y2 = pred[:4]

        # 提取置信度和类别ID
        conf = pred[4]
        class_id = int(pred[5])

        # 只保留置信度超过阈值的框
        if conf > CONF_THRESHOLD:
            # 裁剪到图像范围内
            x1 = np.clip(x1, 0, img_width)
            y1 = np.clip(y1, 0, img_height)
            x2 = np.clip(x2, 0, img_width)
            y2 = np.clip(y2, 0, img_height)

            # 确保坐标顺序正确
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            boxes.append([x1, y1, x2, y2])
            confidences.append(float(conf))
            class_ids.append(class_id)

    return boxes, confidences, class_ids


def load_detector():
    """
    优先加载 ONNX Runtime 模型；如果不可用则回退 OpenVINO IR。
    返回一个统一的 detector 字典供后续推理调用。
    """
    if MODEL_BACKEND in ("auto", "onnxruntime") and os.path.exists(MODEL_ONNX_PATH):
        try:
            session = ort.InferenceSession(MODEL_ONNX_PATH, providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name
            output_names = [item.name for item in session.get_outputs()]
            return {
                "backend": "onnxruntime",
                "model_path": MODEL_ONNX_PATH,
                "session": session,
                "input_name": input_name,
                "output_names": output_names,
            }
        except Exception as exc:
            print(f"ONNX Runtime 加载失败，回退 OpenVINO: {exc}")

    core = ov.Core()
    model = core.read_model(MODEL_XML_PATH)
    model.reshape({model.input().any_name: (1, 3, INPUT_SIZE, INPUT_SIZE)})

    config = {}
    if DEVICE == "CPU":
        config = {
            "PERFORMANCE_HINT": "LATENCY",
            "NUM_STREAMS": "1",
            "AFFINITY": "CORE",
        }

    compiled_model = core.compile_model(model, DEVICE, config)
    return {
        "backend": "openvino",
        "model_path": MODEL_XML_PATH,
        "compiled_model": compiled_model,
        "outputs": compiled_model.outputs,
        "device": DEVICE,
    }


def preprocess_frame(frame, backend):
    """根据后端准备输入张量。"""
    img_resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    if backend == "onnxruntime":
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    img_input = img_resized.astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0)
    return img_input


def run_detector(detector, img_input):
    """执行一次推理并统一返回输出张量列表。"""
    backend = detector["backend"]
    if backend == "onnxruntime":
        return detector["session"].run(detector["output_names"], {detector["input_name"]: img_input})

    results = detector["compiled_model"]([img_input])
    return [results[output] for output in detector["outputs"]]


def process_frame(frame, detector):
    """
    处理单帧图像进行检测
    """
    orig_h, orig_w = frame.shape[:2]

    img_input = preprocess_frame(frame, detector["backend"])
    results = run_detector(detector, img_input)

    # 解码输出
    all_boxes = []
    all_confidences = []
    all_class_ids = []

    for out_data in results:
        boxes, confidences, class_ids = decode_yolov26_output(
            out_data, INPUT_SIZE, INPUT_SIZE
        )
        all_boxes.extend(boxes)
        all_confidences.extend(confidences)
        all_class_ids.extend(class_ids)

    # NMS 非极大值抑制
    if len(all_boxes) > 0:
        indices = nms(np.array(all_boxes), np.array(all_confidences), IOU_THRESHOLD)
        final_boxes = [all_boxes[i] for i in indices]
        final_confidences = [all_confidences[i] for i in indices]
        final_class_ids = [all_class_ids[i] for i in indices]

        # 缩放检测框到原图尺寸
        scale_x = orig_w / INPUT_SIZE
        scale_y = orig_h / INPUT_SIZE
        for box in final_boxes:
            box[0] *= scale_x
            box[1] *= scale_y
            box[2] *= scale_x
            box[3] *= scale_y

        return final_boxes, final_confidences, final_class_ids

    return [], [], []


def draw_detections(img, boxes, confidences, class_ids, classes, track_ids=None, statuses=None):
    """在图像上绘制检测结果"""
    label_items = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        status = None
        if statuses is not None and i < len(statuses):
            status = str(statuses[i]).lower()

        if status == "keep":
            box_color = DISPLAY_BOX_COLOR_KEEP
            status_text = "有效"
        elif status == "reject":
            box_color = DISPLAY_BOX_COLOR_REJECT
            status_text = "无效"
        elif status == "slow" or status == "pending":
            box_color = DISPLAY_BOX_COLOR_SLOW
            status_text = "低速/待定"
        else:
            box_color = DISPLAY_BOX_COLOR_PENDING
            status_text = "待定"

        # 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, DISPLAY_BOX_THICKNESS)

        # 绘制标签
        class_name = classes[class_ids[i]] if 0 <= class_ids[i] < len(classes) else f"class_{class_ids[i]}"
        if track_ids is not None and i < len(track_ids):
            label = f"{status_text} | ID {track_ids[i]} | {class_name} | {confidences[i]:.2f}"
        else:
            label = f"{status_text} | {class_name} | {confidences[i]:.2f}"
        label_items.append((x1, y1, label, box_color))

    return _draw_chinese_labels(img, label_items)


def draw_counting_line(img, line_info, total_count, count_in, count_out, count_slow=0):
    """绘制轨迹方向过滤信息。"""
    axis = line_info.get("axis", [0.0, 1.0])
    anchor = line_info.get("anchor", [img.shape[1] * 0.5, img.shape[0] * 0.5])
    anchor_pt = (int(anchor[0]), int(anchor[1]))
    axis_vec = np.array(axis, dtype=np.float32)
    norm = float(np.linalg.norm(axis_vec))
    if norm > 1e-6:
        axis_vec = axis_vec / norm
        arrow_end = (
            int(anchor_pt[0] + axis_vec[0] * 120),
            int(anchor_pt[1] + axis_vec[1] * 120),
        )
        cv2.arrowedLine(img, anchor_pt, arrow_end, (0, 200, 255), 2, tipLength=0.2)
    cv2.circle(img, anchor_pt, 4, (0, 200, 255), -1)
    cv2.putText(
        img,
        f"Tracks: {total_count}  Keep:{count_in}  Slow:{count_slow}  Reject:{count_out}",
        (10, 90),
        DISPLAY_LABEL_FONT,
        0.8,
        (0, 200, 255),
        2,
    )
    return img


# ─── 视频检测 ──────────────────────────────────────────

def detect_video(source=0, output_path=None, frame_callback=None):
    """
    视频检测函数

    Args:
        source: 视频源，可以是:
            - 0, 1, 2... (摄像头编号)
            - "video.mp4" (视频文件路径)
            - "rtsp://..." (RTSP流地址)
        output_path: 输出视频保存路径，None则不保存
        frame_callback: 每帧回调函数，签名为 callback(frame_bgr, frame_index, avg_fps, num_objects)
                         frame_bgr 为 BGR 格式 numpy 数组
    """
    print("正在加载 YOLOv26 模型...")
    detector = load_detector()
    if detector["backend"] == "onnxruntime":
        print(f"模型加载完成! 后端: ONNX Runtime, 输出节点数: {len(detector['output_names'])}")
    else:
        print(f"模型加载完成! 后端: OpenVINO, 设备: {detector['device']}, 输出节点数: {len(detector['outputs'])}")
    print(f"输入尺寸: {INPUT_SIZE}x{INPUT_SIZE}, 跳帧: {SKIP_FRAMES}")

    # 打开视频源
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频源: {source}")

    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频信息: {width}x{height}, {fps} FPS")

    # 视频写入器
    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    fps_list = []
    frame_data = []
    class_counts = {}
    last_boxes = []
    last_confidences = []
    last_class_ids = []
    tracker = SimpleTracker()
    direction_filter = TrajectoryDirectionFilter()
    filter_info = direction_filter.get_filter_info(max(width, 1), max(height, 1))
    print("开始检测，按 'q' 退出...")

    while True:
        # 记录开始时间
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # 跳帧检测：每隔SKIP_FRAMES-1帧检测一次
        is_real_detection = (frame_count % SKIP_FRAMES == 0)
        if is_real_detection:
            boxes, confidences, class_ids = process_frame(frame, detector)
            last_boxes, last_confidences, last_class_ids = boxes, confidences, class_ids
        else:
            boxes, confidences, class_ids = last_boxes, last_confidences, last_class_ids

        # 追踪器更新（去重统计）
        track_ids = tracker.update(boxes, class_ids)
        filter_result = direction_filter.update(boxes, class_ids, track_ids, width, height)
        kept_indices = filter_result["kept_indices"]
        kept_boxes = [boxes[i] for i in kept_indices]
        kept_confidences = [confidences[i] for i in kept_indices]
        kept_class_ids = [class_ids[i] for i in kept_indices]
        kept_track_ids = [track_ids[i] for i in kept_indices]
        track_statuses = [direction_filter.track_states.get(tid, {}).get("status", "pending") for tid in track_ids]
        filter_info = direction_filter.get_filter_info(width, height)

        # 记录检测数据
        detections = []
        for i, (box, conf, cls_id, track_id) in enumerate(
            zip(kept_boxes, kept_confidences, kept_class_ids, kept_track_ids)
        ):
            det = {
                "class": get_class_name(cls_id),
                "class_id": int(cls_id),
                "confidence": round(float(conf), 4),
                "x1": round(float(box[0]), 2),
                "y1": round(float(box[1]), 2),
                "x2": round(float(box[2]), 2),
                "y2": round(float(box[3]), 2),
                "track_id": track_id,
            }
            detections.append(det)
            class_name = get_class_name(cls_id)
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        frame_data.append({
            "frame": frame_count,
            "timestamp": round(time.time(), 3),
            "raw_num_objects": len(boxes),
            "detections": detections,
            "num_objects": len(detections),
            "track_filter_events": filter_result["events"],
            "track_count_total": filter_result["track_count_total"],
            "track_count_keep": filter_result["track_count_keep"],
            "track_count_filtered": filter_result["track_count_filtered"],
        })

        # 绘制结果：有效/无效车辆分色显示
        if boxes:
            result_frame = draw_detections(
                frame.copy(), boxes, confidences, class_ids, CLASS_NAMES,
                track_ids=track_ids, statuses=track_statuses
            )
        else:
            result_frame = frame.copy()
        result_frame = draw_counting_line(
            result_frame,
            filter_info,
            filter_result["track_count_total"],
            filter_result["track_count_keep"],
            filter_result["track_count_filtered"],
        )

        # 计算FPS
        end_time = time.time()
        elapsed = end_time - start_time
        fps_val = 1.0 / elapsed if elapsed > 0 else 0.0
        fps_list.append(fps_val)
        if len(fps_list) > 30:  # 保留最近30帧的FPS
            fps_list.pop(0)
        avg_fps = sum(fps_list) / len(fps_list)

        # 在画面左上角显示统计信息面板
        result_frame = draw_info_panel(
            result_frame,
            [
                f"FPS: {avg_fps:.1f}",
                f"Valid Objects: {len(kept_boxes)}",
                f"Raw Objects: {len(boxes)}",
            ],
            origin=(10, 10),
        )

        # 显示
        cv2.imshow(DISPLAY_WINDOW_NAME, result_frame)

        # 保存
        if writer:
            writer.write(result_frame)

        # 实时回调 — skipped frames 不传 detections 避免假排队
        if frame_callback:
            cb_dets = detections if is_real_detection else None
        frame_callback(
            result_frame,
            frame_count,
            avg_fps,
            len(kept_boxes),
                cb_dets,
                fps,
                {
                    "track_count_total": filter_result["track_count_total"],
                    "track_count_keep": filter_result["track_count_keep"],
                    "track_count_slow": filter_result["track_count_slow"],
                    "track_count_filtered": filter_result["track_count_filtered"],
                    "axis": filter_result["axis"],
                    "axis_ready": filter_result["axis_ready"],
                },
            )

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"已处理 {frame_count} 帧, FPS: {avg_fps:.1f}...")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # 输出统计信息
    final_avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    print(f"检测完成! 共处理 {frame_count} 帧")
    print(f"平均 FPS: {final_avg_fps:.1f}")

    # 保存检测数据
    summary = {
        "source": str(source),
        "total_frames": frame_count,
        "avg_fps": round(final_avg_fps, 1),
        "total_detections": sum(f["num_objects"] for f in frame_data),
        "raw_total_detections": sum(f.get("raw_num_objects", 0) for f in frame_data),
        "class_counts": class_counts,
        "unique_class_counts": filter_result["kept_class_counts"],
        "unique_vehicle_count": sum(
            v for k, v in filter_result["kept_class_counts"].items()
            if k.lower() in {"car", "van", "truck", "bus", "motorcycle", "bicycle"}
        ),
        "count_method": "trajectory_direction_filter",
        "track_count_total": filter_result["track_count_total"],
        "track_count_keep": filter_result["track_count_keep"],
        "track_count_slow": filter_result["track_count_slow"],
        "track_count_filtered": filter_result["track_count_filtered"],
        "line_count_total": filter_result["track_count_total"],
        "line_count_in": filter_result["track_count_keep"],
        "line_count_slow": filter_result["track_count_slow"],
        "line_count_out": filter_result["track_count_filtered"],
        "crossed_class_counts": filter_result["kept_class_counts"],
        "slow_class_counts": filter_result["slow_class_counts"],
        "filtered_class_counts": filter_result["filtered_class_counts"],
        "filter_info": filter_info,
        "video_info": {"width": width, "height": height, "fps": fps},
        "model": detector["model_path"],
        "backend": detector["backend"],
        "confidence_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
    }
    session_dir = save_detection_data(source, frame_data, summary)
    return {"session_dir": session_dir, "summary": summary, "output_path": output_path}


def run_cli():
    """命令行入口函数"""
    import sys

    # 支持命令行参数
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "video"  # 默认视频模式

    # ===== 选择检测模式 =====

    if mode == "camera":
        # 模式1: 检测摄像头
        camera_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        output_path = sys.argv[3] if len(sys.argv) > 3 else "test/output/output_camera_yolov26.mp4"
        detect_video(camera_id, output_path)

    elif mode == "video":
        # 模式2: 检测视频文件 (默认)
        video_path = sys.argv[2] if len(sys.argv) > 2 else "test/input/input_video.mp4"
        output_path = sys.argv[3] if len(sys.argv) > 3 else "test/output/output_yolov26.mp4"
        detect_video(video_path, output_path)

    elif mode == "rtsp":
        # 模式3: RTSP流
        rtsp_url = sys.argv[2] if len(sys.argv) > 2 else "rtsp://your_stream_url"
        output_path = sys.argv[3] if len(sys.argv) > 3 else "test/output/output_rtsp_yolov26.mp4"
        detect_video(rtsp_url, output_path)

    else:
        print("使用方法:")
        print("  yolo-v26 camera [摄像头ID] [输出路径]")
        print("  yolo-v26 video [视频路径] [输出路径]")
        print("  yolo-v26 rtsp [RTSP地址] [输出路径]")
        print("\n示例:")
        print("  yolo-v26 camera 0 output.mp4")
        print("  yolo-v26 video test_video.mp4 output.mp4")


if __name__ == "__main__":
    run_cli()
