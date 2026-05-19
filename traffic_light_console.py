#!/usr/bin/env python3
"""
交通灯控制台模拟器
读取 data/ 目录下的检测数据，按画面左右区域划分为 X路/Y路，
根据车辆数量算法决定交通灯时长，按真实时间线模拟完整交通灯周期。
"""

import json
import csv
import os
import time
import sys
from enum import Enum
from datetime import datetime


class LightColor(Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    OFF = "off"


class TrafficLightConsole:
    """交通灯控制台模拟器 - 基于 YOLO 检测数据"""

    ANSI = {
        "red": "\033[41m",
        "yellow": "\033[43m",
        "green": "\033[42m",
        "off": "\033[40m",
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
    }

    # 车辆类别
    VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike", "bicycle"}

    # 交通灯参数
    YELLOW_DURATION = 3.0        # 黄灯时长(秒)
    YELLOW_FLASH_INTERVAL = 0.6  # 黄灯闪烁间隔(秒)
    MIN_GREEN = 10.0             # 最短绿灯(秒)
    MAX_GREEN = 30.0            # 最长绿灯(秒)
    STD_GREEN = 10.0            # 标准绿灯(秒)
    ADJUST_RATIO = 0.2          # 车辆差异调整比例

    def __init__(self):
        self.data_dir = "data"
        self.video_width = 1280

        # 当前灯时长
        self.x_green = self.STD_GREEN
        self.x_red = self.STD_GREEN
        self.y_green = self.STD_GREEN
        self.y_red = self.STD_GREEN

    # ─── 数据加载 ──────────────────────────────────

    def list_sessions(self):
        if not os.path.exists(self.data_dir):
            return []
        sessions = [
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d)) and d.startswith("detection_")
        ]
        sessions.sort(reverse=True)
        return sessions

    def find_latest_session(self):
        sessions = self.list_sessions()
        return os.path.join(self.data_dir, sessions[0]) if sessions else None

    def load_summary(self, session_dir):
        path = os.path.join(session_dir, "summary.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def load_frames(self, session_dir):
        """加载帧数据，优先JSON回退CSV"""
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
                    "class": row["class"],
                    "class_id": int(row["class_id"]),
                    "confidence": float(row["confidence"]),
                    "x1": float(row["x1"]), "y1": float(row["y1"]),
                    "x2": float(row["x2"]), "y2": float(row["y2"]),
                })
        return [frames[k] for k in sorted(frames.keys())]

    # ─── 核心算法 ──────────────────────────────────

    def count_vehicles(self, detections, mid_x):
        """按画面中线统计左右两侧车辆数"""
        left = right = 0
        for d in detections:
            if d.get("class", "").lower() not in self.VEHICLE_CLASSES:
                continue
            cx = (d["x1"] + d["x2"]) / 2
            if cx < mid_x:
                left += 1
            else:
                right += 1
        return left, right

    def compute_light_timing(self, car_x, car_y):
        """
        根据两路车辆数计算绿灯时长

        算法:
        - 标准绿灯 = 10秒
        - 车多的路绿灯 +20%（最多30秒）
        - 车少的路绿灯 -20%（最少10秒）
        - 对面路红灯 = 本路绿灯（反之亦然）
        """
        x_green = self.STD_GREEN
        y_green = self.STD_GREEN

        if car_x > car_y:
            x_green = min(self.STD_GREEN + self.STD_GREEN * self.ADJUST_RATIO, self.MAX_GREEN)
            y_green = max(self.STD_GREEN - self.STD_GREEN * self.ADJUST_RATIO, self.MIN_GREEN)
        elif car_y > car_x:
            y_green = min(self.STD_GREEN + self.STD_GREEN * self.ADJUST_RATIO, self.MAX_GREEN)
            x_green = max(self.STD_GREEN - self.STD_GREEN * self.ADJUST_RATIO, self.MIN_GREEN)

        return {
            "x_green": x_green,
            "x_red": y_green + self.YELLOW_DURATION,  # X路红灯 = Y路绿灯 + 黄灯
            "y_green": y_green,
            "y_red": x_green + self.YELLOW_DURATION,   # Y路红灯 = X路绿灯 + 黄灯
        }

    def build_timeline(self, frames, fps):
        """
        构建完整交通灯时间线

        按视频时间线，每隔一个灯周期取一个统计窗口，
        窗口内所有帧的车辆数取均值，作为下一周期的决策依据。
        """
        mid_x = self.video_width / 2
        total_duration = len(frames) / fps

        timeline = []

        # 初始周期: X路绿灯
        phase = "X_GREEN"
        elapsed = 0.0
        frame_idx = 0

        while elapsed < total_duration:
            # 收集当前周期内所有帧的车辆数
            cycle_car_x = []
            cycle_car_y = []

            # 用上一个周期的时长来决定本周期统计多久的帧
            if timeline:
                last = timeline[-1]
                cycle_duration = last["x_green"] if phase == "X_GREEN" else last["y_green"]
            else:
                cycle_duration = self.STD_GREEN

            # 收集本周期时间窗口内的帧
            cycle_end = elapsed + cycle_duration
            while frame_idx < len(frames):
                frame_time = frames[frame_idx]["frame"] / fps
                if frame_time > cycle_end:
                    break

                dets = frames[frame_idx].get("detections", [])
                cx, cy = self.count_vehicles(dets, mid_x)
                cycle_car_x.append(cx)
                cycle_car_y.append(cy)
                frame_idx += 1

            # 计算本周期平均车辆数
            avg_x = sum(cycle_car_x) / len(cycle_car_x) if cycle_car_x else 0
            avg_y = sum(cycle_car_y) / len(cycle_car_y) if cycle_car_y else 0

            # 根据车辆数计算下一周期的灯时长
            timing = self.compute_light_timing(avg_x, avg_y)

            entry = {
                "phase": phase,
                "start_time": elapsed,
                "car_x_avg": round(avg_x, 1),
                "car_y_avg": round(avg_y, 1),
                "car_x_total": sum(cycle_car_x),
                "car_y_total": sum(cycle_car_y),
                "frames_in_cycle": len(cycle_car_x),
                **timing,
            }
            timeline.append(entry)

            # 推进时间: 绿灯 + 黄灯
            green_time = timing["x_green"] if phase == "X_GREEN" else timing["y_green"]
            elapsed += green_time + self.YELLOW_DURATION

            # 切换相位
            phase = "Y_GREEN" if phase == "X_GREEN" else "X_GREEN"

        return timeline, total_duration

    # ─── 显示 ──────────────────────────────────────

    def _light_str(self, color, seconds=None):
        c = self.ANSI.get(color.value, self.ANSI["off"])
        r = self.ANSI["reset"]
        sym = "●" if color != LightColor.OFF else "○"
        names = {LightColor.RED: "红灯", LightColor.YELLOW: "黄灯",
                 LightColor.GREEN: "绿灯", LightColor.OFF: "熄灭"}
        s = f"{c} {sym} {names[color]} {r}"
        if seconds is not None:
            s += f" [{seconds:.1f}s]"
        return s

    def print_phase(self, entry, remaining=None):
        """打印当前相位状态"""
        phase = entry["phase"]
        if phase == "X_GREEN":
            x_color, x_sec = LightColor.GREEN, entry["x_green"]
            y_color, y_sec = LightColor.RED, entry["y_red"]
        else:
            x_color, x_sec = LightColor.RED, entry["x_red"]
            y_color, y_sec = LightColor.GREEN, entry["y_green"]

        if remaining is not None:
            x_sec = remaining if phase == "X_GREEN" else entry["x_red"] - (entry["x_green"] + self.YELLOW_DURATION - remaining)
            y_sec = entry["y_red"] - (entry["x_green"] + self.YELLOW_DURATION - remaining) if phase == "X_GREEN" else remaining
            # 简化：用倒计时
            if phase == "X_GREEN":
                x_sec = remaining
                y_sec = entry["y_red"]  # Y路一直红灯
            else:
                y_sec = remaining
                x_sec = entry["x_red"]

        print(f"\n  X路(左): {self._light_str(x_color, x_sec)}")
        print(f"  Y路(右): {self._light_str(y_color, y_sec)}")
        print(f"  车辆 — X路: {entry['car_x_avg']:.1f}辆(均)  Y路: {entry['car_y_avg']:.1f}辆(均)")

    def print_yellow(self, road_name):
        """打印黄灯过渡"""
        print(f"  {road_name}: {self._light_str(LightColor.YELLOW, self.YELLOW_DURATION)}")

    def print_summary(self, timeline, total_duration):
        """打印最终结果汇总"""
        print(f"\n{'=' * 60}")
        print(f"  交通灯控制结果汇总")
        print(f"{'=' * 60}")
        print(f"  视频总时长: {total_duration:.1f}s")
        print(f"  总周期数:   {len(timeline)}")
        print()

        for i, entry in enumerate(timeline, 1):
            phase = entry["phase"]
            phase_cn = "X路绿灯/Y路红灯" if phase == "X_GREEN" else "Y路绿灯/X路红灯"
            print(f"  周期 {i}: {phase_cn}")
            print(f"    X路均车: {entry['car_x_avg']:.1f}  Y路均车: {entry['car_y_avg']:.1f}")
            print(f"    X路绿灯: {entry['x_green']:.1f}s  X路红灯: {entry['x_red']:.1f}s")
            print(f"    Y路绿灯: {entry['y_green']:.1f}s  Y路红灯: {entry['y_red']:.1f}s")
            print()

        # 最终建议
        if timeline:
            last = timeline[-1]
            print(f"{'─' * 60}")
            print(f"  最终建议配时:")
            print(f"    X路绿灯: {last['x_green']:.1f}s  红灯: {last['x_red']:.1f}s")
            print(f"    Y路绿灯: {last['y_green']:.1f}s  红灯: {last['y_red']:.1f}s")
            print(f"    黄灯: {self.YELLOW_DURATION:.1f}s")

            # 整体统计
            all_x = [e["car_x_avg"] for e in timeline]
            all_y = [e["car_y_avg"] for e in timeline]
            print(f"\n  全程车辆统计:")
            print(f"    X路平均: {sum(all_x)/len(all_x):.1f}辆/周期  Y路平均: {sum(all_y)/len(all_y):.1f}辆/周期")

            if sum(all_x) > sum(all_y):
                print(f"    → X路车辆较多，建议 X路绿灯时长适当延长")
            elif sum(all_y) > sum(all_x):
                print(f"    → Y路车辆较多，建议 Y路绿灯时长适当延长")
            else:
                print(f"    → 两路车辆相当，建议均衡配时")

        print(f"{'=' * 60}")

    # ─── 运行模式 ──────────────────────────────────

    def simulate(self, session_dir=None, speed=1.0):
        """
        从 data 数据按真实时间线模拟交通灯

        Args:
            session_dir: 会话目录，None则自动选最新
            speed: 模拟速度倍率 (1.0=实时)
        """
        if session_dir is None:
            session_dir = self.find_latest_session()

        if not session_dir or not os.path.isdir(session_dir):
            print("错误: 未找到检测数据，请先运行检测程序生成数据")
            return

        # 加载数据
        summary = self.load_summary(session_dir)
        frames = self.load_frames(session_dir)

        if not frames:
            print("错误: 未找到帧数据")
            return

        video_info = summary.get("video_info", {})
        self.video_width = video_info.get("width", 1280)
        fps = video_info.get("fps", 30)
        total_duration = len(frames) / fps

        print(f"\n{'=' * 60}")
        print(f"  交通灯智能模拟系统")
        print(f"{'=' * 60}")
        print(f"  数据源: {session_dir}")
        print(f"  视频源: {summary.get('source', 'N/A')}")
        print(f"  分辨率: {self.video_width}x{video_info.get('height', 720)}")
        print(f"  总帧数: {len(frames)} ({total_duration:.1f}s @ {fps:.1f}fps)")
        print(f"  画面中线: x={self.video_width // 2} (左=X路, 右=Y路)")
        print(f"  模拟速度: {speed}x")
        print(f"{'=' * 60}")

        # 构建完整时间线
        print("\n正在分析检测数据并构建交通灯时间线...")
        timeline, _ = self.build_timeline(frames, fps)

        if not timeline:
            print("错误: 无法构建时间线")
            return

        print(f"已生成 {len(timeline)} 个灯周期，开始模拟...\n")

        # 按时间线实时模拟
        try:
            for i, entry in enumerate(timeline):
                phase = entry["phase"]
                green_seconds = entry["x_green"] if phase == "X_GREEN" else entry["y_green"]
                phase_cn = "X路绿灯 / Y路红灯" if phase == "X_GREEN" else "Y路绿灯 / X路红灯"

                print(f"┌─ 周期 {i+1}/{len(timeline)} ─ {phase_cn} ─ 视频时间 {entry['start_time']:.1f}s")
                print(f"│  统计帧数: {entry['frames_in_cycle']}")

                # 绿灯阶段 - 按秒倒计时
                remaining = green_seconds
                while remaining > 0:
                    step = min(1.0, remaining)
                    # 显示当前状态
                    if phase == "X_GREEN":
                        print(f"│  X路: {self._light_str(LightColor.GREEN, remaining)}  "
                              f"Y路: {self._light_str(LightColor.RED)}")
                    else:
                        print(f"│  X路: {self._light_str(LightColor.RED)}  "
                              f"Y路: {self._light_str(LightColor.GREEN, remaining)}")
                    sys.stdout.flush()
                    time.sleep(step / speed)
                    remaining -= step

                # 黄灯过渡
                print(f"│  ── 黄灯过渡 {self.YELLOW_DURATION:.0f}s ──")
                yellow_remaining = self.YELLOW_DURATION
                while yellow_remaining > 0:
                    step = min(self.YELLOW_FLASH_INTERVAL, yellow_remaining)
                    flash_on = int(yellow_remaining / self.YELLOW_FLASH_INTERVAL) % 2 == 0
                    if flash_on:
                        print(f"│  X路: {self._light_str(LightColor.YELLOW, yellow_remaining)}  "
                              f"Y路: {self._light_str(LightColor.YELLOW)}")
                    else:
                        print(f"│  X路: {self._light_str(LightColor.OFF, yellow_remaining)}  "
                              f"Y路: {self._light_str(LightColor.OFF)}")
                    sys.stdout.flush()
                    time.sleep(step / speed)
                    yellow_remaining -= step

                print(f"└─ 周期 {i+1} 结束")

        except KeyboardInterrupt:
            print("\n\n>>> 模拟已中断")

        # 打印最终结果
        self.print_summary(timeline, total_duration)

    def analyze_only(self, session_dir=None):
        """仅分析数据，输出时间线和结果，不实时模拟"""
        if session_dir is None:
            session_dir = self.find_latest_session()

        if not session_dir or not os.path.isdir(session_dir):
            print("错误: 未找到检测数据")
            return

        summary = self.load_summary(session_dir)
        frames = self.load_frames(session_dir)

        if not frames:
            print("错误: 未找到帧数据")
            return

        video_info = summary.get("video_info", {})
        self.video_width = video_info.get("width", 1280)
        fps = video_info.get("fps", 30)
        total_duration = len(frames) / fps

        print(f"\n{'=' * 60}")
        print(f"  交通灯数据分析 (仅输出结果)")
        print(f"{'=' * 60}")
        print(f"  数据源: {session_dir}")
        print(f"  视频时长: {total_duration:.1f}s ({len(frames)}帧)")

        timeline, _ = self.build_timeline(frames, fps)
        self.print_summary(timeline, total_duration)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="交通灯控制台模拟器 - 基于YOLO检测数据")
    parser.add_argument("--simulate", action="store_true", help="按真实时间线模拟交通灯 (默认)")
    parser.add_argument("--analyze", action="store_true", help="仅分析数据，直接输出结果")
    parser.add_argument("--session", type=str, default=None, help="指定会话目录名")
    parser.add_argument("--speed", type=float, default=10.0, help="模拟速度倍率 (默认: 10)")
    parser.add_argument("--list", action="store_true", help="列出所有检测会话")

    args = parser.parse_args()

    light = TrafficLightConsole()

    if args.list:
        sessions = light.list_sessions()
        if sessions:
            print("检测会话:")
            for s in sessions:
                sd = os.path.join(light.data_dir, s)
                sm = light.load_summary(sd)
                vi = sm.get("video_info", {})
                cc = sm.get("class_counts", {})
                vehicles = {k: v for k, v in cc.items() if k.lower() in light.VEHICLE_CLASSES}
                total_vehicles = sum(vehicles.values())
                print(f"  {s}")
                print(f"    视频: {sm.get('source', '?')}  "
                      f"分辨率: {vi.get('width', '?')}x{vi.get('height', '?')}  "
                      f"时长: {sm.get('total_frames', 0)/vi.get('fps', 30):.1f}s")
                print(f"    车辆: {vehicles} (合计: {total_vehicles})")
        else:
            print("未找到检测数据")
        return

    session_dir = None
    if args.session:
        session_dir = os.path.join(light.data_dir, args.session)
        if not os.path.isdir(session_dir):
            print(f"错误: 会话目录不存在 - {session_dir}")
            return

    if args.analyze:
        light.analyze_only(session_dir)
    else:
        # 默认: 模拟
        light.simulate(session_dir, speed=args.speed)


if __name__ == "__main__":
    main()
