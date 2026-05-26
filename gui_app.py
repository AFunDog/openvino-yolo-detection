#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 智能交通灯控制系统 - Dear PyGui 桌面应用
"""

import json
import csv
import os
import time
import math
import threading
import sys
from pathlib import Path

import dearpygui.dearpygui as dpg
import numpy as np

# ─── HiDPI & Encoding ───────────────────────────────
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ─── 路径 ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
TEST_OUTPUT_DIR = PROJECT_ROOT / "test" / "output"

VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike", "bicycle"}

# ─── 基准尺寸 ───────────────────────────────────────
BASE_W, BASE_H = 1200, 750

# ─── 颜色 ────────────────────────────────────────────
C_PRIMARY        = (79, 70, 229, 255)
C_PRIMARY_HOVER   = (109, 99, 255, 255)
C_PRIMARY_LIGHT   = (238, 242, 255, 255)
C_RED             = (220, 38, 38, 255)
C_YELLOW          = (202, 138, 4, 255)
C_GREEN           = (22, 163, 74, 255)
C_RED_DIM         = (220, 38, 38, 60)
C_YELLOW_DIM      = (202, 138, 4, 60)
C_GREEN_DIM       = (22, 163, 74, 60)
C_BLUE            = (37, 99, 235, 255)
C_ORANGE          = (234, 88, 12, 255)
C_PURPLE          = (147, 51, 234, 255)

C_BG_BASE         = (240, 242, 245, 255)
C_BG_SURFACE      = (255, 255, 255, 255)
C_BG_ELEVATED     = (241, 243, 245, 255)
C_BG_OVERLAY      = (233, 236, 239, 255)
C_CARD_BG         = (255, 255, 255, 255)
C_CARD_BORDER     = (226, 229, 235, 255)

C_TEXT_PRIMARY     = (17, 24, 39, 255)
C_TEXT_SECONDARY   = (55, 65, 81, 255)
C_TEXT_MUTED       = (107, 114, 128, 255)

C_BORDER          = (220, 225, 231, 255)
C_BORDER_LIGHT    = (235, 238, 243, 255)

# Canvas
C_ROAD = (209, 213, 219, 255)
C_GRASS = (187, 247, 208, 255)
C_INTERSECTION = (199, 203, 209, 255)
C_LANE = (156, 163, 175, 255)
C_CROSSWALK = (107, 114, 128, 40)
C_SIDEWALK = (156, 163, 175, 120)
C_STOP_LINE = (107, 114, 128, 80)
C_CENTER_LINE = (202, 138, 4, 100)
C_VEHICLE_BG = (255, 255, 255, 180)


# ─── 数据加载 ────────────────────────────────────────

def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_frames(session_dir):
    json_path = os.path.join(session_dir, "frames.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    csv_path = os.path.join(session_dir, "frames.csv")
    if not os.path.exists(csv_path):
        return []
    frames = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = int(row["frame"])
            if fn not in frames:
                frames[fn] = {"frame": fn, "timestamp": float(row["timestamp"]), "detections": []}
            frames[fn]["detections"].append({
                "class": row["class"], "class_id": int(row["class_id"]),
                "confidence": float(row["confidence"]),
                "x1": float(row["x1"]), "y1": float(row["y1"]),
                "x2": float(row["x2"]), "y2": float(row["y2"]),
            })
    return [frames[k] for k in sorted(frames.keys())]


def build_timeline(frames, video_width=1280, fps=30.0):
    YELLOW = 3.0; STD = 10.0; MIN_G = 10.0; MAX_G = 30.0; RATIO = 0.2
    mid_x = video_width / 2
    total_dur = len(frames) / fps
    timeline = []; phase = "X_GREEN"; elapsed = 0.0; idx = 0
    while elapsed < total_dur:
        cx_list, cy_list = [], []
        dur = STD
        if timeline:
            last = timeline[-1]
            dur = last["x_green"] if phase == "X_GREEN" else last["y_green"]
        end_t = elapsed + dur
        while idx < len(frames):
            ft = frames[idx]["frame"] / fps
            if ft > end_t:
                break
            left = right = 0
            for d in frames[idx].get("detections", []):
                if d.get("class", "").lower() not in VEHICLE_CLASSES:
                    continue
                cx = (d["x1"] + d["x2"]) / 2
                if cx < mid_x:
                    left += 1
                else:
                    right += 1
            cx_list.append(left); cy_list.append(right); idx += 1
        ax = sum(cx_list) / len(cx_list) if cx_list else 0
        ay = sum(cy_list) / len(cy_list) if cy_list else 0
        xg, yg = STD, STD
        if ax > ay:
            xg = min(STD + STD * RATIO, MAX_G); yg = max(STD - STD * RATIO, MIN_G)
        elif ay > ax:
            yg = min(STD + STD * RATIO, MAX_G); xg = max(STD - STD * RATIO, MIN_G)
        timeline.append({
            "phase": phase, "start_time": round(elapsed, 1),
            "car_x_avg": round(ax, 1), "car_y_avg": round(ay, 1),
            "x_green": round(xg, 1), "x_red": round(yg + YELLOW, 1),
            "y_green": round(yg, 1), "y_red": round(xg + YELLOW, 1),
            "yellow_duration": YELLOW,
        })
        gt = xg if phase == "X_GREEN" else yg
        elapsed += gt + YELLOW
        phase = "Y_GREEN" if phase == "X_GREEN" else "X_GREEN"
    return timeline, round(total_dur, 1)


# ─── 应用状态 ────────────────────────────────────────

class AppState:
    def __init__(self):
        self.sessions = []
        self.selected_session = None
        self.selected_summary = {}
        self.timeline = []
        self.total_duration = 0
        self.current_cycle = -1
        self.x_light = "off"
        self.y_light = "off"
        self.sim_running = False
        self.sim_paused = False
        self.sim_speed = 5.0
        self.cycle_elapsed = 0.0
        self.yellow_elapsed = 0.0
        self.in_yellow = False
        self.last_tick = 0
        self.active_page = "yolo"
        self.detecting = False
        self.detect_progress = ""
        self.scale = 1.0

state = AppState()


# ─── 视频播放器 ───────────────────────────────────────

class VideoPlayer:
    TEX_W, TEX_H = 640, 360

    def __init__(self):
        self.cap = None
        self.playing = False
        self.video_path = ""
        self.fps = 30
        self.last_frame_time = 0
        self.frame_count = 0
        self.current_frame = 0

    def load(self, path):
        self.stop()
        import cv2
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            return False
        self.video_path = path
        self.fps = max(1, self.cap.get(cv2.CAP_PROP_FPS) or 30)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.playing = True
        self.last_frame_time = time.time()
        # Show first frame
        self._read_and_update()
        return True

    def _read_and_update(self):
        if not self.cap:
            return False
        ret, frame = self.cap.read()
        if not ret:
            self.playing = False
            return False
        self.current_frame += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        frame = cv2.resize(frame, (self.TEX_W, self.TEX_H))
        data = (frame.astype(np.float32) / 255.0).flatten().tolist()
        dpg.set_value("video_texture", data)
        return True

    def tick(self):
        if not self.playing or not self.cap:
            return
        now = time.time()
        interval = 1.0 / self.fps
        if now - self.last_frame_time < interval:
            return
        self.last_frame_time = now
        if not self._read_and_update():
            dpg.set_value("detect_status", "播放结束")
            self.playing = False

    def toggle_pause(self):
        if self.cap:
            self.playing = not self.playing
            if self.playing:
                self.last_frame_time = time.time()

    def stop(self):
        self.playing = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.current_frame = 0
        self.frame_count = 0
        # Clear texture
        data = [0.08, 0.08, 0.10, 1.0] * (self.TEX_W * self.TEX_H)
        try:
            dpg.set_value("video_texture", data)
        except Exception:
            pass

    def is_loaded(self):
        return self.cap is not None


video_player = VideoPlayer()


# ─── 十字路口 Canvas ────────────────────────────────

def draw_intersection(x_color="off", y_color="off", car_x=None, car_y=None,
                      countdown=None, width=None, height=None):
    tag = "intersection_draw"
    dpg.delete_item(tag, children_only=True)
    parent = tag
    W = width or 580
    H = height or 580
    road_w = int(W * 0.22)
    cx, cy = W / 2, H / 2

    # 四角草地
    corners = [
        (0, 0, cx - road_w/2, cy - road_w/2),
        (cx + road_w/2, 0, W, cy - road_w/2),
        (0, cy + road_w/2, cx - road_w/2, H),
        (cx + road_w/2, cy + road_w/2, W, H),
    ]
    for x1, y1, x2, y2 in corners:
        dpg.draw_rectangle((x1, y1), (x2, y2), color=C_GRASS, fill=C_GRASS, parent=parent)
        sw = max(4, int(W * 0.014))
        if x1 == 0:
            dpg.draw_rectangle((x2 - sw, y1), (x2, y2), color=C_SIDEWALK, fill=C_SIDEWALK, parent=parent)
        if x2 == W:
            dpg.draw_rectangle((x1, y1), (x1 + sw, y2), color=C_SIDEWALK, fill=C_SIDEWALK, parent=parent)
        if y1 == 0:
            dpg.draw_rectangle((x1, y2 - sw), (x2, y2), color=C_SIDEWALK, fill=C_SIDEWALK, parent=parent)
        if y2 == H:
            dpg.draw_rectangle((x1, y1), (x2, y1 + sw), color=C_SIDEWALK, fill=C_SIDEWALK, parent=parent)

    # 道路
    dpg.draw_rectangle((0, cy - road_w/2), (W, cy + road_w/2), color=C_ROAD, fill=C_ROAD, parent=parent)
    dpg.draw_rectangle((cx - road_w/2, 0), (cx + road_w/2, H), color=C_ROAD, fill=C_ROAD, parent=parent)
    dpg.draw_rectangle((cx - road_w/2, cy - road_w/2), (cx + road_w/2, cy + road_w/2),
                        color=C_INTERSECTION, fill=C_INTERSECTION, parent=parent)

    # 中心线
    for offset in [-2, 2]:
        for x_s in range(0, int(cx - road_w/2), 30):
            dpg.draw_line((x_s, cy + offset), (x_s + 20, cy + offset),
                          color=C_CENTER_LINE, thickness=1, parent=parent)
        for x_s in range(int(cx + road_w/2), W, 30):
            dpg.draw_line((x_s, cy + offset), (x_s + 20, cy + offset),
                          color=C_CENTER_LINE, thickness=1, parent=parent)
        for y_s in range(0, int(cy - road_w/2), 30):
            dpg.draw_line((cx + offset, y_s), (cx + offset, y_s + 20),
                          color=C_CENTER_LINE, thickness=1, parent=parent)
        for y_s in range(int(cy + road_w/2), H, 30):
            dpg.draw_line((cx + offset, y_s), (cx + offset, y_s + 20),
                          color=C_CENTER_LINE, thickness=1, parent=parent)

    # 车道虚线
    hw = road_w / 4
    for lane_offset in [-hw, hw]:
        for x_s in range(0, int(cx - road_w/2), 24):
            dpg.draw_line((x_s, cy + lane_offset), (x_s + 14, cy + lane_offset),
                          color=C_LANE, thickness=1, parent=parent)
        for x_s in range(int(cx + road_w/2), W, 24):
            dpg.draw_line((x_s, cy + lane_offset), (x_s + 14, cy + lane_offset),
                          color=C_LANE, thickness=1, parent=parent)
        for y_s in range(0, int(cy - road_w/2), 24):
            dpg.draw_line((cx + lane_offset, y_s), (cx + lane_offset, y_s + 14),
                          color=C_LANE, thickness=1, parent=parent)
        for y_s in range(int(cy + road_w/2), H, 24):
            dpg.draw_line((cx + lane_offset, y_s), (cx + lane_offset, y_s + 14),
                          color=C_LANE, thickness=1, parent=parent)

    # 人行横道
    for i in range(8):
        ox = cx - road_w/2 + i * (road_w/8) + 2; ow = road_w/8 - 4
        dpg.draw_rectangle((ox, cy - road_w/2 - 14), (ox + ow, cy - road_w/2 - 2),
                            color=C_CROSSWALK, fill=C_CROSSWALK, parent=parent)
        dpg.draw_rectangle((ox, cy + road_w/2 + 2), (ox + ow, cy + road_w/2 + 14),
                            color=C_CROSSWALK, fill=C_CROSSWALK, parent=parent)
        oy = cy - road_w/2 + i * (road_w/8) + 2; oh = road_w/8 - 4
        dpg.draw_rectangle((cx - road_w/2 - 14, oy), (cx - road_w/2 - 2, oy + oh),
                            color=C_CROSSWALK, fill=C_CROSSWALK, parent=parent)
        dpg.draw_rectangle((cx + road_w/2 + 2, oy), (cx + road_w/2 + 14, oy + oh),
                            color=C_CROSSWALK, fill=C_CROSSWALK, parent=parent)

    # 停车线
    dpg.draw_line((cx - road_w/2, cy - road_w/2 - 2), (cx, cy - road_w/2 - 2),
                  color=C_STOP_LINE, thickness=2, parent=parent)
    dpg.draw_line((cx, cy + road_w/2 + 2), (cx + road_w/2, cy + road_w/2 + 2),
                  color=C_STOP_LINE, thickness=2, parent=parent)
    dpg.draw_line((cx - road_w/2 - 2, cy), (cx - road_w/2 - 2, cy + road_w/2),
                  color=C_STOP_LINE, thickness=2, parent=parent)
    dpg.draw_line((cx + road_w/2 + 2, cy - road_w/2), (cx + road_w/2 + 2, cy),
                  color=C_STOP_LINE, thickness=2, parent=parent)

    # 道路标签
    fs = max(10, int(W * 0.022))
    dpg.draw_text((cx - 18, cy - road_w/2 - 36), "X", color=C_TEXT_SECONDARY, size=fs, parent=parent)
    dpg.draw_text((cx - 18, cy + road_w/2 + 22), "X", color=C_TEXT_SECONDARY, size=fs, parent=parent)
    dpg.draw_text((cx - road_w/2 - 30, cy - 8), "Y", color=C_TEXT_SECONDARY, size=fs, parent=parent)
    dpg.draw_text((cx + road_w/2 + 16, cy - 8), "Y", color=C_TEXT_SECONDARY, size=fs, parent=parent)

    # 交通灯
    def draw_light(lx, ly, ac):
        bw = max(20, int(W * 0.048)); bh = max(56, int(H * 0.14))
        r = max(5, int(W * 0.013))
        dpg.draw_rectangle((lx - bw/2 - 2, ly - bh/2 - 2), (lx + bw/2 + 2, ly + bh/2 + 2),
                            color=(180, 180, 190, 255), fill=(80, 80, 90, 255), parent=parent)
        dpg.draw_rectangle((lx - bw/2, ly - bh/2), (lx + bw/2, ly + bh/2),
                            color=(100, 100, 110, 255), fill=(50, 50, 60, 255), parent=parent)
        for i, cn in enumerate(["red", "yellow", "green"]):
            by = ly - bh/3 + i * (bh/3)
            is_on = cn == ac
            if cn == "red":    fill = C_RED if is_on else C_RED_DIM
            elif cn == "yellow": fill = C_YELLOW if is_on else C_YELLOW_DIM
            else:              fill = C_GREEN if is_on else C_GREEN_DIM
            if is_on:
                glow = fill[:3] + (30,)
                dpg.draw_circle((lx, by), r + 6, color=glow, fill=glow, parent=parent)
            dpg.draw_circle((lx, by), r, color=fill, fill=fill, parent=parent)

    off_x = max(20, int(W * 0.04))
    off_y = max(30, int(H * 0.08))
    draw_light(cx - road_w/2 - off_x, cy - road_w/2 - off_y, y_color)
    draw_light(cx + road_w/2 + off_x, cy + road_w/2 + off_y, y_color)
    draw_light(cx - road_w/2 - off_y, cy + road_w/2 + off_x, x_color)
    draw_light(cx + road_w/2 + off_y, cy - road_w/2 - off_x, x_color)

    # 车辆数
    if car_x is not None:
        dpg.draw_text((20, cy - 14), f"X: {car_x}", color=C_BLUE, size=fs, parent=parent)
        dpg.draw_text((W - 80, cy - 14), f"X: {car_x}", color=C_BLUE, size=fs, parent=parent)
    if car_y is not None:
        dpg.draw_text((cx - 14, 14), f"Y: {car_y}", color=C_ORANGE, size=fs, parent=parent)
        dpg.draw_text((cx - 14, H - 30), f"Y: {car_y}", color=C_ORANGE, size=fs, parent=parent)

    # 倒计时
    if countdown is not None:
        cfs = max(14, int(W * 0.03))
        dpg.draw_text((cx - cfs/2, cy - cfs/2), str(math.ceil(countdown)),
                      color=C_TEXT_PRIMARY, size=cfs, parent=parent)


# ─── YOLO 检测 ────────────────────────────────────────

def on_select_video(sender, app_data):
    if not app_data or not app_data.get("file_path_name"):
        return
    dpg.set_value("video_path_input", app_data["file_path_name"])


def on_browse_video():
    dpg.show_item("file_dialog_video")


def on_start_detect():
    video_path = dpg.get_value("video_path_input").strip()
    if not video_path or not os.path.exists(video_path):
        dpg.set_value("detect_status", "错误: 视频路径不存在")
        return
    if state.detecting:
        return

    model_path = PROJECT_ROOT / "public" / "yolo-v26" / "ir_model" / "yolo26n.xml"
    if not model_path.exists():
        dpg.set_value("detect_status", "错误: YOLOv26 模型文件不存在")
        return

    os.makedirs(str(TEST_OUTPUT_DIR), exist_ok=True)
    basename = Path(video_path).stem
    output_path = str(TEST_OUTPUT_DIR / f"output_{basename}.mp4")

    state.detecting = True
    video_player.stop()
    dpg.set_value("detect_status", "检测中... (YOLOv26)")
    dpg.configure_item("btn_start_detect", enabled=False)

    def run_detect():
        import cv2
        orig = (cv2.imshow, cv2.waitKey, cv2.destroyAllWindows)
        try:
            cv2.imshow = lambda *_a, **_k: None
            cv2.waitKey = lambda *_a, **_k: -1
            cv2.destroyAllWindows = lambda: None

            import yolov26 as yolo
            cwd = os.getcwd()
            os.chdir(str(PROJECT_ROOT))
            yolo.detect_video(video_path, output_path)
            os.chdir(cwd)
            state.detect_progress = output_path
        except Exception as e:
            import traceback
            state.detect_progress = f"FAIL:{str(e)[:200]}\n{traceback.format_exc()[:300]}"
        finally:
            cv2.imshow, cv2.waitKey, cv2.destroyAllWindows = orig
            state.detecting = False

    threading.Thread(target=run_detect, daemon=True).start()


def check_detect_status():
    if not state.detecting and state.detect_progress:
        prog = state.detect_progress
        state.detect_progress = ""
        dpg.configure_item("btn_start_detect", enabled=True)
        if prog.startswith("FAIL:"):
            dpg.set_value("detect_status", prog[5:])
        else:
            dpg.set_value("detect_status", "检测完成，正在加载视频...")
            if video_player.load(prog):
                dpg.set_value("detect_status", f"播放: {Path(prog).name}")
            else:
                dpg.set_value("detect_status", "检测完成，但视频无法播放")
        load_sessions()


def on_play_video():
    if video_player.is_loaded():
        video_player.toggle_pause()
        label = "⏸ 暂停" if video_player.playing else "▶ 播放"
        dpg.configure_item("btn_play_video", label=label)


# ─── YOLO 页面逻辑 ───────────────────────────────────

def load_sessions():
    state.sessions = []
    if not DATA_DIR.exists():
        return
    for d in sorted(DATA_DIR.iterdir(), reverse=True):
        if d.is_dir() and d.name.startswith("detection_"):
            summary = load_json(d / "summary.json")
            state.sessions.append({"name": d.name, "path": str(d), "summary": summary})
    dpg.configure_item("session_list", items=[s["name"] for s in state.sessions])
    items = ["(默认周期)"] + [s["name"] for s in state.sessions]
    dpg.configure_item("data_source_combo", items=items)


def on_session_select(sender, app_data):
    idx = app_data
    if idx < 0 or idx >= len(state.sessions):
        return
    s = state.sessions[idx]
    state.selected_session = s
    summary = s.get("summary", {})
    state.selected_summary = summary

    classes = summary.get("class_counts", {})
    total = summary.get("total_detections", 0)
    frames = summary.get("total_frames", 0)
    fps_val = summary.get("avg_fps", 0)
    vi = summary.get("video_info", {})
    vehicle_count = sum(v for k, v in classes.items() if k.lower() in VEHICLE_CLASSES)

    dpg.set_value("stat_frames", str(frames))
    dpg.set_value("stat_detections", str(total))
    dpg.set_value("stat_vehicles", str(vehicle_count))
    dpg.set_value("stat_fps", str(round(fps_val, 1)))

    # 类别表格
    dpg.delete_item("class_table_container", children_only=True)
    with dpg.table(header_row=True, policy=dpg.mvTable_SizingStretchProp,
                   borders_innerH=True, borders_outerH=True,
                   borders_innerV=True, borders_outerV=True,
                   width=-1, parent="class_table_container"):
        dpg.add_table_column(label="类别", width_stretch=True)
        dpg.add_table_column(label="数量", width_stretch=True)
        dpg.add_table_column(label="占比", width_stretch=True)
        for cls, count in sorted(classes.items(), key=lambda x: -x[1]):
            pct = f"{(count / total * 100):.1f}%" if total else "0%"
            with dpg.table_row():
                dpg.add_text(cls, color=C_TEXT_PRIMARY)
                dpg.add_text(str(count), color=C_PRIMARY)
                dpg.add_text(pct, color=C_TEXT_SECONDARY)

    info = (
        f"视频源: {summary.get('source', 'N/A')}\n"
        f"分辨率: {vi.get('width', '?')}x{vi.get('height', '?')}\n"
        f"总帧数: {frames}  总检测: {total}\n"
        f"车辆数: {vehicle_count}  FPS: {fps_val:.1f}"
    )
    dpg.set_value("session_detail", info)


# ─── 十字路口模拟 ────────────────────────────────────

def _default_timeline():
    state.timeline = []
    for i in range(20):
        is_x = i % 2 == 0
        state.timeline.append({
            "phase": "X_GREEN" if is_x else "Y_GREEN",
            "start_time": i * 13,
            "car_x_avg": round(3 + (i % 3) * 1.5, 1),
            "car_y_avg": round(2 + (i % 4) * 1.2, 1),
            "x_green": 10, "x_red": 13, "y_green": 10, "y_red": 13,
            "yellow_duration": 3,
        })
    state.total_duration = 260


def on_load_timeline():
    sel = dpg.get_value("data_source_combo")
    if sel and sel != "(默认周期)":
        session_dir = os.path.join(DATA_DIR, sel)
        frames = load_frames(session_dir)
        if frames:
            summary = load_json(os.path.join(session_dir, "summary.json"))
            vi = summary.get("video_info", {})
            vw = vi.get("width", 1280)
            fps_val = vi.get("fps", 30)
            state.timeline, state.total_duration = build_timeline(frames, vw, fps_val)
        else:
            _default_timeline()
    else:
        _default_timeline()

    dpg.delete_item("timeline_table_container", children_only=True)
    with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit,
                   borders_innerH=True, borders_outerH=True,
                   borders_innerV=True, borders_outerV=True,
                   width=-1, height=200, parent="timeline_table_container"):
        dpg.add_table_column(label="相位", width_fixed=True, init_width_or_weight=50)
        dpg.add_table_column(label="时间", width_fixed=True, init_width_or_weight=55)
        dpg.add_table_column(label="车辆", width_stretch=True)
        dpg.add_table_column(label="绿灯", width_fixed=True, init_width_or_weight=50)
        for t in state.timeline:
            phase_text = "X绿" if t["phase"] == "X_GREEN" else "Y绿"
            phase_color = C_BLUE if t["phase"] == "X_GREEN" else C_ORANGE
            with dpg.table_row():
                dpg.add_text(phase_text, color=phase_color)
                dpg.add_text(f"{t['start_time']}s", color=C_TEXT_SECONDARY)
                dpg.add_text(f"X:{t['car_x_avg']} Y:{t['car_y_avg']}", color=C_TEXT_SECONDARY)
                dpg.add_text(f"{t['x_green'] if t['phase']=='X_GREEN' else t['y_green']}s", color=C_GREEN)

    state.current_cycle = -1
    state.cycle_elapsed = 0
    state.in_yellow = False


def on_start_sim():
    if state.sim_running and not state.sim_paused:
        return
    if not state.sim_running:
        on_load_timeline()
        state.sim_running = True
        state.sim_paused = False
        state.current_cycle = 0
        state.cycle_elapsed = 0
        state.in_yellow = False
        state.yellow_elapsed = 0
        state.last_tick = time.time()
    else:
        state.sim_paused = False
        state.last_tick = time.time()


def on_pause_sim():
    state.sim_paused = True


def on_reset_sim():
    state.sim_running = False
    state.sim_paused = False
    state.current_cycle = -1
    state.x_light = "off"
    state.y_light = "off"
    draw_intersection()
    dpg.set_value("timer_text", "--")
    dpg.set_value("progress_bar", 0)
    dpg.set_value("cycle_info_text", "点击 ▶ 开始模拟")


def update_light_indicators(x_color, y_color):
    on_map = {"red": C_RED, "yellow": C_YELLOW, "green": C_GREEN, "off": C_BORDER}
    dim_map = {"red": C_RED_DIM, "yellow": C_YELLOW_DIM, "green": C_GREEN_DIM, "off": C_BORDER}
    for c_name in ["red", "yellow", "green"]:
        dpg.configure_item(f"xl_{c_name}", color=on_map[c_name] if x_color == c_name else dim_map[c_name])
        dpg.configure_item(f"yl_{c_name}", color=on_map[c_name] if y_color == c_name else dim_map[c_name])


def sim_tick():
    if not state.sim_running or state.sim_paused:
        return
    now = time.time()
    dt = (now - state.last_tick) * state.sim_speed
    state.last_tick = now

    tl = state.timeline
    if not tl or state.current_cycle >= len(tl):
        on_reset_sim()
        dpg.set_value("cycle_info_text", "模拟结束")
        return

    cycle = tl[state.current_cycle]
    is_x = cycle["phase"] == "X_GREEN"
    green_dur = cycle["x_green"] if is_x else cycle["y_green"]
    yellow_dur = cycle.get("yellow_duration", 3)

    if not state.in_yellow:
        state.cycle_elapsed += dt
        remaining = max(0, green_dur - state.cycle_elapsed)
        dpg.set_value("timer_text", f"{math.ceil(remaining)}s")
        dpg.set_value("progress_bar", state.cycle_elapsed / green_dur if green_dur > 0 else 0)

        if is_x:
            state.x_light = "green"; state.y_light = "red"
            draw_intersection("green", "red", round(cycle["car_x_avg"]), round(cycle["car_y_avg"]), remaining)
        else:
            state.x_light = "red"; state.y_light = "green"
            draw_intersection("red", "green", round(cycle["car_x_avg"]), round(cycle["car_y_avg"]), remaining)

        update_light_indicators(state.x_light, state.y_light)
        phase_label = "X路绿灯 / Y路红灯" if is_x else "Y路绿灯 / X路红灯"
        dpg.configure_item("phase_label", color=C_BLUE if is_x else C_ORANGE)
        dpg.set_value("phase_label", phase_label)
        dpg.set_value("cycle_info_text",
            f"周期 {state.current_cycle+1}/{len(tl)}\n"
            f"X路: {cycle['car_x_avg']}辆(均) | Y路: {cycle['car_y_avg']}辆(均)\n"
            f"绿灯: {green_dur}s | 红灯: {cycle['x_red'] if is_x else cycle['y_red']}s")

        if state.cycle_elapsed >= green_dur:
            state.in_yellow = True; state.yellow_elapsed = 0
    else:
        state.yellow_elapsed += dt
        remaining = max(0, yellow_dur - state.yellow_elapsed)
        dpg.set_value("timer_text", f"黄灯 {math.ceil(remaining)}s")
        dpg.set_value("progress_bar", state.yellow_elapsed / yellow_dur if yellow_dur > 0 else 0)

        flash = int(state.yellow_elapsed * 3) % 2 == 0
        xc = "yellow" if flash else "off"
        yc = "yellow" if flash else "off"
        state.x_light = xc; state.y_light = yc
        draw_intersection(xc, yc, round(cycle["car_x_avg"]), round(cycle["car_y_avg"]), remaining)
        update_light_indicators(xc, yc)

        dpg.configure_item("phase_label", color=C_YELLOW)
        dpg.set_value("phase_label", "黄灯过渡")
        dpg.set_value("cycle_info_text",
            f"周期 {state.current_cycle+1} -> {state.current_cycle+2}\n"
            f"双向黄灯 {yellow_dur}s")

        if state.yellow_elapsed >= yellow_dur:
            state.current_cycle += 1; state.cycle_elapsed = 0; state.in_yellow = False


def on_speed_change(sender, app_data):
    state.sim_speed = app_data


# ─── 导航 ─────────────────────────────────────────────

_nav_active_theme = None
_nav_inactive_theme = None

def _init_nav_themes():
    global _nav_active_theme, _nav_inactive_theme
    _nav_active_theme = dpg.generate_uuid()
    _nav_inactive_theme = dpg.generate_uuid()
    with dpg.theme(tag=_nav_active_theme):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, C_PRIMARY)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, C_PRIMARY_HOVER)
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)
    with dpg.theme(tag=_nav_inactive_theme):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, C_BG_ELEVATED)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, C_BG_OVERLAY)
            dpg.add_theme_color(dpg.mvThemeCol_Text, C_TEXT_SECONDARY)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)


def switch_page(page):
    state.active_page = page
    dpg.configure_item("page_yolo", show=page == "yolo")
    dpg.configure_item("page_traffic", show=page == "traffic")
    dpg.bind_item_theme("nav_yolo", _nav_active_theme if page == "yolo" else _nav_inactive_theme)
    dpg.bind_item_theme("nav_traffic", _nav_active_theme if page == "traffic" else _nav_inactive_theme)


# ─── 卡片主题 ─────────────────────────────────────────

_card_theme = None

def _init_card_theme():
    global _card_theme
    _card_theme = dpg.generate_uuid()
    with dpg.theme(tag=_card_theme):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, C_CARD_BG)
            dpg.add_theme_color(dpg.mvThemeCol_Border, C_CARD_BORDER)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 10)
            dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 1)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 14, 12)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 6, 5)


# ─── UI 构建 ─────────────────────────────────────────

def build_ui():
    # 全局主题
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, C_BG_BASE)
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, C_BG_SURFACE)
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, C_BG_SURFACE)
            dpg.add_theme_color(dpg.mvThemeCol_Border, C_BORDER)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, C_BG_SURFACE)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, C_BG_OVERLAY)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, C_BG_SURFACE)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, C_BG_SURFACE)
            dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, C_BG_SURFACE)
            dpg.add_theme_color(dpg.mvThemeCol_Tab, C_BG_ELEVATED)
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, C_PRIMARY)
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, C_PRIMARY_HOVER)
            dpg.add_theme_color(dpg.mvThemeCol_Button, C_PRIMARY)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, C_PRIMARY_HOVER)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (67, 56, 202, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text, C_TEXT_PRIMARY)
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, C_TEXT_MUTED)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, C_BG_BASE)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, C_BORDER)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, C_BORDER_LIGHT)
            dpg.add_theme_color(dpg.mvThemeCol_Separator, C_BORDER)
            dpg.add_theme_color(dpg.mvThemeCol_TableRowBg, C_BG_SURFACE)
            dpg.add_theme_color(dpg.mvThemeCol_TableRowBgAlt, C_BG_BASE)
            dpg.add_theme_color(dpg.mvThemeCol_TableBorderStrong, C_BORDER)
            dpg.add_theme_color(dpg.mvThemeCol_TableBorderLight, C_BORDER_LIGHT)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, C_PRIMARY)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, C_PRIMARY_HOVER)
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, C_PRIMARY)
            dpg.add_theme_color(dpg.mvThemeCol_Header, C_PRIMARY_LIGHT)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, C_PRIMARY)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, C_PRIMARY)
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, C_BG_SURFACE)

            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 10)
            dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 12, 10)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 4)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 6)
            dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 0)
            dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 1)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)

    dpg.bind_theme(global_theme)

    # 视频纹理
    texture_data = [0.08, 0.08, 0.10, 1.0] * (VideoPlayer.TEX_W * VideoPlayer.TEX_H)
    dpg.add_dynamic_texture(VideoPlayer.TEX_W, VideoPlayer.TEX_H, texture_data, tag="video_texture")

    # 按钮主题
    btn_start_theme = dpg.generate_uuid()
    with dpg.theme(tag=btn_start_theme):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, C_GREEN)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (16, 185, 129, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255))
    btn_pause_theme = dpg.generate_uuid()
    with dpg.theme(tag=btn_pause_theme):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, C_YELLOW)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (245, 158, 11, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255))
    btn_reset_theme = dpg.generate_uuid()
    with dpg.theme(tag=btn_reset_theme):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, C_RED)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (248, 113, 113, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255))

    # 文件对话框
    with dpg.file_dialog(directory_selector=False, show=False, tag="file_dialog_video",
                          callback=on_select_video, width=600, height=400):
        dpg.add_file_extension(".mp4")
        dpg.add_file_extension(".avi")
        dpg.add_file_extension(".*")

    _init_nav_themes()
    _init_card_theme()

    # ─── 主窗口（无标题栏） ───
    with dpg.window(tag="primary_window", no_move=True, no_collapse=True,
                    no_title_bar=True, no_bring_to_front_on_focus=True):

        # 主体：侧栏 + 内容
        with dpg.group(horizontal=True, horizontal_spacing=0):

            # ── 左侧导航栏 ──
            with dpg.child_window(width=200, border=False, tag="sidebar"):
                dpg.add_spacer(height=12)
                with dpg.group(width=-1):
                    dpg.add_button(label="  YOLO Video Analysis", tag="nav_yolo",
                                   callback=lambda: switch_page("yolo"), width=-1)
                    dpg.add_spacer(height=4)
                    dpg.add_button(label="  Traffic Simulation", tag="nav_traffic",
                                   callback=lambda: switch_page("traffic"), width=-1)

                dpg.add_spacer(height=16)
                dpg.add_separator()
                dpg.add_spacer(height=16)

                dpg.add_text("Sessions", color=C_TEXT_SECONDARY)
                dpg.add_spacer(height=4)
                dpg.add_listbox(tag="session_list", items=[], num_items=8,
                                callback=on_session_select, width=-1)

            dpg.bind_item_theme("nav_yolo", _nav_active_theme)
            dpg.bind_item_theme("nav_traffic", _nav_inactive_theme)

            dpg.add_spacer(width=4)

            # ── 右侧内容区 ──
            with dpg.child_window(tag="page_yolo", border=False, show=True, width=-1):
                dpg.add_spacer(height=8)

                # 上传区卡片
                with dpg.child_window(tag="card_upload", border=True, width=-1):
                    dpg.bind_item_theme("card_upload", _card_theme)
                    dpg.add_text("YOLOv26 Video Detection", color=C_PRIMARY)
                    dpg.add_spacer(height=6)
                    with dpg.group(horizontal=True):
                        dpg.add_input_text(tag="video_path_input", default_value="", width=-1,
                                           hint="Video path...")
                        dpg.add_button(label="Browse", callback=on_browse_video, width=70)
                        dpg.add_button(label="Detect", tag="btn_start_detect",
                                       callback=on_start_detect, width=80)
                    dpg.add_spacer(height=4)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Ready", tag="detect_status", color=C_TEXT_SECONDARY)
                        dpg.add_spacer(width=-1)
                        dpg.add_button(label="Play", tag="btn_play_video",
                                       callback=on_play_video, width=70, height=28)

                dpg.add_spacer(height=8)

                # 下方两栏
                with dpg.group(horizontal=True):

                    # 左列：统计 + 详情
                    with dpg.group(width=320):
                        # 统计卡片
                        with dpg.child_window(tag="card_stats", border=True, width=-1):
                            dpg.bind_item_theme("card_stats", _card_theme)
                            with dpg.group(horizontal=True):
                                with dpg.group():
                                    dpg.add_text("0", tag="stat_frames", color=C_BLUE)
                                    dpg.add_text("Frames", color=C_TEXT_MUTED)
                                dpg.add_spacer(width=16)
                                with dpg.group():
                                    dpg.add_text("0", tag="stat_detections", color=C_PRIMARY)
                                    dpg.add_text("Detections", color=C_TEXT_MUTED)
                                dpg.add_spacer(width=16)
                                with dpg.group():
                                    dpg.add_text("0", tag="stat_vehicles", color=C_GREEN)
                                    dpg.add_text("Vehicles", color=C_TEXT_MUTED)
                                dpg.add_spacer(width=16)
                                with dpg.group():
                                    dpg.add_text("0", tag="stat_fps", color=C_ORANGE)
                                    dpg.add_text("FPS", color=C_TEXT_MUTED)

                        dpg.add_spacer(height=8)

                        # 详情卡片
                        with dpg.child_window(tag="card_detail", border=True, width=-1):
                            dpg.bind_item_theme("card_detail", _card_theme)
                            dpg.add_text("Session Detail", color=C_TEXT_SECONDARY)
                            dpg.add_spacer(height=4)
                            dpg.add_input_text(tag="session_detail",
                                               default_value="Select a session from the sidebar",
                                               multiline=True, height=70, width=-1, readonly=True)

                        dpg.add_spacer(height=8)

                        # 类别统计卡片
                        with dpg.child_window(tag="card_class", border=True, width=-1):
                            dpg.bind_item_theme("card_class", _card_theme)
                            dpg.add_text("Class Statistics", color=C_TEXT_SECONDARY)
                            dpg.add_spacer(height=4)
                            dpg.add_group(tag="class_table_container")

                    dpg.add_spacer(width=8)

                    # 右列：视频预览
                    with dpg.group(width=-1):
                        with dpg.child_window(tag="card_video", border=True, width=-1):
                            dpg.bind_item_theme("card_video", _card_theme)
                            dpg.add_text("Video Preview", color=C_TEXT_SECONDARY)
                            dpg.add_spacer(height=4)
                            with dpg.drawlist(width=VideoPlayer.TEX_W, height=VideoPlayer.TEX_H,
                                              tag="video_drawlist"):
                                dpg.draw_image("video_texture", (0, 0),
                                               (VideoPlayer.TEX_W, VideoPlayer.TEX_H),
                                               tag="video_draw_img")

            # ── 交通灯页 ──
            with dpg.child_window(tag="page_traffic", border=False, show=False, width=-1):
                dpg.add_spacer(height=8)
                with dpg.group(horizontal=True):
                    # 画布
                    with dpg.group(width=-1):
                        with dpg.child_window(tag="card_intersection", border=True, width=-1):
                            dpg.bind_item_theme("card_intersection", _card_theme)
                            dpg.add_text("Intersection Status", color=C_TEXT_SECONDARY)
                            dpg.add_spacer(height=4)
                            with dpg.drawlist(width=580, height=580, tag="intersection_draw"):
                                pass

                    dpg.add_spacer(width=8)

                    # 控制面板
                    with dpg.group(width=300):
                        with dpg.child_window(tag="card_control", border=True, width=-1):
                            dpg.bind_item_theme("card_control", _card_theme)
                            dpg.add_text("Traffic Light Status", color=C_TEXT_SECONDARY)
                            dpg.add_spacer(height=4)

                            with dpg.group(horizontal=True):
                                with dpg.group():
                                    dpg.add_text("X Road", color=C_BLUE)
                                    with dpg.group(horizontal=True):
                                        dpg.add_text("●", tag="xl_red", color=C_RED_DIM)
                                        dpg.add_text("●", tag="xl_yellow", color=C_YELLOW_DIM)
                                        dpg.add_text("●", tag="xl_green", color=C_GREEN_DIM)
                                dpg.add_spacer(width=20)
                                with dpg.group():
                                    dpg.add_text("Y Road", color=C_ORANGE)
                                    with dpg.group(horizontal=True):
                                        dpg.add_text("●", tag="yl_red", color=C_RED_DIM)
                                        dpg.add_text("●", tag="yl_yellow", color=C_YELLOW_DIM)
                                        dpg.add_text("●", tag="yl_green", color=C_GREEN_DIM)

                            dpg.add_spacer(height=8)

                            dpg.add_text("Phase:", color=C_TEXT_MUTED)
                            dpg.add_text("--", tag="phase_label", color=C_TEXT_PRIMARY)

                            with dpg.group(horizontal=True):
                                dpg.add_text("Countdown", color=C_TEXT_MUTED)
                                dpg.add_text("--", tag="timer_text", color=C_GREEN)

                            dpg.add_progress_bar(tag="progress_bar", default_value=0, width=-1, height=6, overlay="")

                            dpg.add_spacer(height=12)

                            with dpg.group(horizontal=True):
                                btn_s = dpg.add_button(label="▶ Start", callback=on_start_sim, width=90, height=32)
                                btn_p = dpg.add_button(label="⏸ Pause", callback=on_pause_sim, width=90, height=32)
                                btn_r = dpg.add_button(label="■ Reset", callback=on_reset_sim, width=90, height=32)
                            dpg.bind_item_theme(btn_s, btn_start_theme)
                            dpg.bind_item_theme(btn_p, btn_pause_theme)
                            dpg.bind_item_theme(btn_r, btn_reset_theme)

                            dpg.add_spacer(height=12)

                            dpg.add_text("Speed", color=C_TEXT_MUTED)
                            dpg.add_slider_float(tag="speed_slider", default_value=5.0,
                                                 min_value=1.0, max_value=20.0,
                                                 callback=on_speed_change, width=-1,
                                                 format="%.0fx", height=20)

                            dpg.add_spacer(height=12)
                            dpg.add_text("Data Source", color=C_TEXT_MUTED)
                            dpg.add_combo(tag="data_source_combo", items=["(Default)"],
                                          default_value="(Default)", width=-1)

                            dpg.add_spacer(height=8)
                            dpg.add_separator()
                            dpg.add_spacer(height=8)

                            dpg.add_text("Cycle Info", color=C_TEXT_MUTED)
                            dpg.add_input_text(tag="cycle_info_text",
                                               default_value="Click ▶ Start",
                                               multiline=True, height=60, width=-1, readonly=True)

                            dpg.add_spacer(height=8)
                            dpg.add_text("Timeline", color=C_TEXT_MUTED)
                            dpg.add_group(tag="timeline_table_container")

        # 底部状态栏
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_text("Ready", color=C_TEXT_MUTED, tag="status_text")
            dpg.add_spacer(width=-1)
            dpg.add_text("OpenVINO + YOLOv26", color=C_TEXT_MUTED)


# ─── 主入口 ──────────────────────────────────────────

def main():
    dpg.create_context()

    # 中文字体
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    with dpg.font_registry():
        for fp in font_paths:
            if os.path.exists(fp):
                default_font = dpg.add_font(fp, 20)
                dpg.bind_font(default_font)
                break

    # 用 ASCII 标题避免乱码
    dpg.create_viewport(title="YOLO Traffic Light System", width=BASE_W, height=BASE_H)

    build_ui()
    draw_intersection()
    load_sessions()

    dpg.set_primary_window("primary_window", True)
    dpg.setup_dearpygui()
    dpg.set_viewport_pos([100, 100])
    dpg.show_viewport()

    _last_vp_w = BASE_W
    _last_vp_h = BASE_H

    while dpg.is_dearpygui_running():
        sim_tick()
        check_detect_status()
        video_player.tick()

        # 响应式布局
        vp_w = dpg.get_viewport_width()
        vp_h = dpg.get_viewport_height()
        if vp_w != _last_vp_w or vp_h != _last_vp_h:
            _last_vp_w = vp_w
            _last_vp_h = vp_h

            scale = min(vp_w / BASE_W, vp_h / BASE_H)
            state.scale = scale

            # 内容区高度
            content_h = max(400, vp_h - 30)
            dpg.configure_item("sidebar", height=content_h)
            dpg.configure_item("page_yolo", height=content_h)
            dpg.configure_item("page_traffic", height=content_h)

            # YOLO 页：视频 drawlist 自适应
            video_w = max(320, int(vp_w - 200 - 340 - 40))
            video_h = max(180, int(video_w * 9 / 16))
            dpg.configure_item("video_drawlist", width=video_w, height=video_h)
            dpg.configure_item("video_draw_img", pmax=(video_w, video_h))

            # 交通灯页：drawlist 自适应
            iw = max(300, int(vp_w - 200 - 320 - 40))
            ih = max(300, int(vp_h - 80))
            dpg.configure_item("intersection_draw", width=iw, height=ih)
            draw_intersection(x_color=state.x_light, y_color=state.y_light,
                              width=iw, height=ih)

        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == "__main__":
    main()
