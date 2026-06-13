"""
VA vs 固定配时 对比评估脚本

在 SUMO 微观仿真环境下对比：
- VA 自适应控制（VAController）
- 固定配时控制（FixedTimeController）

输出对比指标：
- 平均延误
- 总通过量
- 最大排队
- 切换次数
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from algorithm import VAController

SUMO_DIR = Path(__file__).parent
NET_FILE = SUMO_DIR / "network" / "intersection.net.xml"
SUMO_HOME = os.environ.get("SUMO_HOME", "")
SUMO_BIN = Path(SUMO_HOME) / "bin" / "sumo.exe" if SUMO_HOME else "sumo"
SUMO_GUI_BIN = Path(SUMO_HOME) / "bin" / "sumo-gui.exe" if SUMO_HOME else "sumo-gui"


@dataclass
class CompareResult:
    """对比结果"""
    scenario: str
    strategy: str
    avg_delay: float
    total_delay: float
    passed_total: float
    arrived_total: float
    max_queue_total: float
    switch_count: int
    sim_time: float


def run_fixed_time(
    scenario: str,
    duration: float = 1800.0,
    green_x: float = 20.0,
    green_y: float = 20.0,
    gui: bool = False,
    delay: float = 0.0,
    verbose: bool = True,
) -> CompareResult:
    """运行固定配时仿真"""
    try:
        import traci
    except ImportError:
        raise ImportError("traci 未安装。请运行: pip install traci")

    route_file = SUMO_DIR / "routes" / f"{scenario}.xml"
    if not route_file.exists():
        raise FileNotFoundError(f"路由文件不存在: {route_file}")
    if not NET_FILE.exists():
        raise FileNotFoundError(f"路网文件不存在: {NET_FILE}")

    sumo_cmd = [
        str(SUMO_GUI_BIN if gui else SUMO_BIN),
        "-n", str(NET_FILE),
        "-r", str(route_file),
        "--step-length", "1.0",
        "--delay", str(delay),
        "--no-warnings", "true",
        "--no-step-log", "true",
        "--quit-on-end",
        "--start",
    ]

    traci.start(sumo_cmd)

    # 固定配时逻辑
    phase = 0  # 0=X, 1=Y
    phase_elapsed = 0.0
    in_yellow = False
    yellow_elapsed = 0.0
    yellow_duration = 3.0
    switch_count = 0

    total_delay = 0.0
    arrived = 0
    max_queue = 0.0
    step = 0

    while traci.simulation.getMinExpectedNumber() > 0 and step < duration:
        traci.simulationStep()

        queue_x, queue_y = 0, 0
        for veh in traci.vehicle.getIDList():
            try:
                edge = traci.vehicle.getRoadID(veh)
                speed = traci.vehicle.getSpeed(veh)
                if speed < 0.5:
                    if edge in ("left0A0", "right0A0"):
                        queue_x += 1
                    elif edge in ("bottom0A0", "top0A0"):
                        queue_y += 1
            except Exception:
                pass

        total_delay += (queue_x + queue_y) * 1.0
        max_queue = max(max_queue, queue_x + queue_y)

        if in_yellow:
            yellow_elapsed += 1.0
            if yellow_elapsed >= yellow_duration:
                in_yellow = False
                yellow_elapsed = 0.0
                phase = 1 - phase
                phase_elapsed = 0.0
                switch_count += 1
            x_state = "yellow"
            y_state = "yellow"
        else:
            phase_elapsed += 1.0
            target = green_x if phase == 0 else green_y
            if phase_elapsed >= target:
                in_yellow = True
                yellow_elapsed = 0.0
                x_state = "yellow"
                y_state = "yellow"
            else:
                x_state = "green" if phase == 0 else "red"
                y_state = "red" if phase == 0 else "green"

        # 12 位信号: Y(top)X(right)Y(bottom)X(left) 各 3 位
        def to_sumo(s):
            return {"green": "G", "red": "r", "yellow": "y"}.get(s, "r")
        x_sig = to_sumo(x_state)
        y_sig = to_sumo(y_state)
        tl_state = f"{y_sig}{y_sig}{y_sig}{x_sig}{x_sig}{x_sig}{y_sig}{y_sig}{y_sig}{x_sig}{x_sig}{x_sig}"
        try:
            traci.trafficlight.setRedYellowGreenState("A0", tl_state)
        except Exception:
            pass

        arrived += len(traci.simulation.getArrivedIDList())
        step += 1

    traci.close()

    return CompareResult(
        scenario=scenario,
        strategy="fixed_time",
        avg_delay=total_delay / arrived if arrived > 0 else 0,
        total_delay=total_delay,
        passed_total=arrived,
        arrived_total=arrived,
        max_queue_total=max_queue,
        switch_count=switch_count,
        sim_time=step,
    )


def run_va_control(
    scenario: str,
    duration: float = 1800.0,
    controller_params: Optional[dict] = None,
    gui: bool = False,
    delay: float = 0.0,
    verbose: bool = True,
) -> CompareResult:
    """运行 VA 控制仿真"""
    try:
        import traci
    except ImportError:
        raise ImportError("traci 未安装。请运行: pip install traci")

    route_file = SUMO_DIR / "routes" / f"{scenario}.xml"
    if not route_file.exists():
        raise FileNotFoundError(f"路由文件不存在: {route_file}")
    if not NET_FILE.exists():
        raise FileNotFoundError(f"路网文件不存在: {NET_FILE}")

    sumo_cmd = [
        str(SUMO_GUI_BIN if gui else SUMO_BIN),
        "-n", str(NET_FILE),
        "-r", str(route_file),
        "--step-length", "1.0",
        "--delay", str(delay),
        "--no-warnings", "true",
        "--no-step-log", "true",
        "--quit-on-end",
        "--start",
    ]

    if controller_params is None:
        controller_params = {}
    controller = VAController(**controller_params)

    traci.start(sumo_cmd)

    total_delay = 0.0
    arrived = 0
    max_queue = 0.0
    step = 0

    while traci.simulation.getMinExpectedNumber() > 0 and step < duration:
        traci.simulationStep()

        queue_x, queue_y = 0, 0
        for veh in traci.vehicle.getIDList():
            try:
                edge = traci.vehicle.getRoadID(veh)
                speed = traci.vehicle.getSpeed(veh)
                if speed < 0.5:
                    if edge in ("left0A0", "right0A0"):
                        queue_x += 1
                    elif edge in ("bottom0A0", "top0A0"):
                        queue_y += 1
            except Exception:
                pass

        total_delay += (queue_x + queue_y) * 1.0
        max_queue = max(max_queue, queue_x + queue_y)

        x_light, y_light, _ = controller.step(queue_x, queue_y, 1.0)

        # 12 位信号: Y(top)X(right)Y(bottom)X(left) 各 3 位
        def to_sumo(s):
            return {"green": "G", "red": "r", "yellow": "y"}.get(s, "r")

        x_sig = to_sumo(x_light)
        y_sig = to_sumo(y_light)
        tl_state = f"{y_sig}{y_sig}{y_sig}{x_sig}{x_sig}{x_sig}{y_sig}{y_sig}{y_sig}{x_sig}{x_sig}{x_sig}"
        try:
            traci.trafficlight.setRedYellowGreenState("A0", tl_state)
        except Exception:
            pass

        arrived += len(traci.simulation.getArrivedIDList())
        step += 1

    traci.close()

    return CompareResult(
        scenario=scenario,
        strategy="adaptive",
        avg_delay=total_delay / arrived if arrived > 0 else 0,
        total_delay=total_delay,
        passed_total=arrived,
        arrived_total=arrived,
        max_queue_total=max_queue,
        switch_count=len(controller.phase_history),
        sim_time=step,
    )


def compare(
    scenarios: list = None,
    duration: float = 1800.0,
    fixed_green_x: float = 20.0,
    fixed_green_y: float = 20.0,
    va_params: Optional[dict] = None,
    gui: bool = False,
    delay: float = 0.0,
    pause: bool = False,
) -> dict:
    """
    运行完整对比

    Args:
        pause: GUI模式下，每个仿真结束后暂停，按Enter继续

    Returns:
        对比结果字典
    """
    if scenarios is None:
        scenarios = ["balanced", "imbalanced", "tidal", "burst"]

    results = {}

    print("=" * 70)
    print("VA vs 固定配时 对比评估")
    print("=" * 70)
    print(f"场景: {scenarios}")
    print(f"仿真时长: {duration}s")
    print(f"固定配时: X={fixed_green_x}s, Y={fixed_green_y}s")
    print(f"VA 参数: {va_params or '默认'}")
    if gui:
        print(f"GUI延迟: {delay}ms  |  对比模式: 依次展示")
    print("=" * 70)

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n\033[1;36m{'=' * 60}\033[0m")
        print(f"\033[1;36m  [{i}/{len(scenarios)}] 场景: {scenario}\033[0m")
        print(f"\033[1;36m{'=' * 60}\033[0m")

        # 固定配时
        print(f"\n  >>> 固定配时 (Fixed-Time) 仿真运行中 ...")
        fixed = run_fixed_time(scenario, duration, fixed_green_x, fixed_green_y, gui=gui, delay=delay)
        print(f"  固定配时: avg_delay={fixed.avg_delay:.2f}s, "
              f"passed={fixed.passed_total:.0f}, "
              f"max_queue={fixed.max_queue_total:.0f}, "
              f"switches={fixed.switch_count}")
        if gui and pause:
            input("  [Enter] 继续 VA 自适应仿真...")

        # VA 控制
        print(f"\n  >>> VA 自适应 仿真运行中 ...")
        va = run_va_control(scenario, duration, va_params, gui=gui, delay=delay)
        print(f"  VA 控制:  avg_delay={va.avg_delay:.2f}s, "
              f"passed={va.passed_total:.0f}, "
              f"max_queue={va.max_queue_total:.0f}, "
              f"switches={va.switch_count}")
        if gui and pause and i < len(scenarios):
            input("  [Enter] 继续下一个场景...")

        # 计算提升
        delay_reduction = ((fixed.avg_delay - va.avg_delay) / fixed.avg_delay * 100
                          if fixed.avg_delay > 0 else 0)
        queue_reduction = ((fixed.max_queue_total - va.max_queue_total) / fixed.max_queue_total * 100
                          if fixed.max_queue_total > 0 else 0)

        print(f"  \033[32m延误下降: {delay_reduction:.1f}%\033[0m")
        print(f"  \033[32m排队下降: {queue_reduction:.1f}%\033[0m")

        results[scenario] = {
            "fixed": {
                "avg_delay": fixed.avg_delay,
                "total_delay": fixed.total_delay,
                "passed_total": fixed.passed_total,
                "max_queue_total": fixed.max_queue_total,
                "switch_count": fixed.switch_count,
            },
            "adaptive": {
                "avg_delay": va.avg_delay,
                "total_delay": va.total_delay,
                "passed_total": va.passed_total,
                "max_queue_total": va.max_queue_total,
                "switch_count": va.switch_count,
            },
            "improvement": {
                "delay_reduction_pct": delay_reduction,
                "queue_reduction_pct": queue_reduction,
            }
        }

    # 汇总
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)

    fixed_avg = sum(r["fixed"]["avg_delay"] for r in results.values()) / len(results)
    va_avg = sum(r["adaptive"]["avg_delay"] for r in results.values()) / len(results)
    overall_improvement = ((fixed_avg - va_avg) / fixed_avg * 100) if fixed_avg > 0 else 0

    print(f"固定配时平均延误: {fixed_avg:.2f}s")
    print(f"VA 控制平均延误:  {va_avg:.2f}s")
    print(f"整体延误下降:     {overall_improvement:.1f}%")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VA vs 固定配时对比")
    parser.add_argument("--scenarios", nargs="+",
                        default=["balanced", "imbalanced", "tidal", "burst"])
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--fixed-green-x", type=float, default=20.0)
    parser.add_argument("--fixed-green-y", type=float, default=20.0)
    parser.add_argument("--min-green", type=float, default=10.0)
    parser.add_argument("--max-green", type=float, default=30.0)
    parser.add_argument("--max-red", type=float, default=25.0)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="GUI延迟(ms)，建议50-500")
    parser.add_argument("--pause", action="store_true",
                        help="GUI模式下切换场景前暂停，按Enter继续")

    args = parser.parse_args()

    compare(
        scenarios=args.scenarios,
        duration=args.duration,
        fixed_green_x=args.fixed_green_x,
        fixed_green_y=args.fixed_green_y,
        va_params={
            "min_green": args.min_green,
            "max_green": args.max_green,
            "max_red": args.max_red,
        },
        gui=args.gui,
        delay=args.delay,
        pause=args.pause,
    )
