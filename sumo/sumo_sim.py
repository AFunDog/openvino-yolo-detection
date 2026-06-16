"""
SUMO + TraCI 仿真主循环

将 SUMO 微观仿真与 VAController 连接：
- 从 SUMO 读取排队车辆数
- 调用 VAController 决策红绿灯
- 将红绿灯状态写回 SUMO
- 收集性能指标

需要: pip install traci
"""

import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from algorithm import VAController

SUMO_DIR = Path(__file__).parent
NET_FILE = SUMO_DIR / "network" / "intersection.net.xml"

SUMO_HOME = os.environ.get("SUMO_HOME", "")
SUMO_BIN = Path(SUMO_HOME) / "bin" / "sumo.exe" if SUMO_HOME else "sumo"


@dataclass
class SimulationMetrics:
    sim_time: float = 0.0
    arrived: float = 0.0
    total_delay: float = 0.0
    max_queue: float = 0.0
    switch_count: int = 0

    @property
    def avg_delay(self) -> float:
        return self.total_delay / self.arrived if self.arrived > 0 else 0.0


def get_queue_lengths(traci_conn) -> tuple:
    """获取 X/Y 方向排队车辆数"""
    x_edges = {"left0A0", "right0A0"}
    y_edges = {"bottom0A0", "top0A0"}

    queue_x = 0
    queue_y = 0

    for veh in traci_conn.vehicle.getIDList():
        try:
            edge = traci_conn.vehicle.getRoadID(veh)
            speed = traci_conn.vehicle.getSpeed(veh)
            if speed < 0.5:
                if edge in x_edges:
                    queue_x += 1
                elif edge in y_edges:
                    queue_y += 1
        except Exception:
            pass

    return queue_x, queue_y


def set_traffic_light(traci_conn, x_state: str, y_state: str):
    """
    设置交通灯。TLS ID = "A0", 12 位信号:
      0-2   top0A0 (Y), 3-5   right0A0 (X),
      6-8   bottom0A0 (Y), 9-11  left0A0 (X)
    """
    def sig(s):
        return {"green": "G", "red": "r", "yellow": "y"}.get(s, "r")

    x = sig(x_state)
    y = sig(y_state)

    tl_state = f"{y}{y}{y}{x}{x}{x}{y}{y}{y}{x}{x}{x}"
    try:
        traci_conn.trafficlight.setRedYellowGreenState("A0", tl_state)
    except Exception:
        pass


def run_simulation(
    scenario: str = "balanced",
    duration: float = 1800.0,
    controller_params: Optional[dict] = None,
    gui: bool = False,
    delay: float = 0.0,
    verbose: bool = True,
) -> SimulationMetrics:
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
        str(SUMO_BIN if not gui else Path(SUMO_HOME) / "bin" / "sumo-gui.exe"),
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

    metrics = SimulationMetrics()
    step = 0

    if verbose:
        print(f"仿真: scenario={scenario}, duration={duration}s")

    while traci.simulation.getMinExpectedNumber() > 0 and step < duration:
        traci.simulationStep()

        queue_x, queue_y = get_queue_lengths(traci)

        arrived = traci.simulation.getArrivedNumber()
        # 累计延误 = 排队车辆数 × 步长
        metrics.total_delay += (queue_x + queue_y) * 1.0
        metrics.max_queue = max(metrics.max_queue, queue_x + queue_y)
        metrics.arrived += traci.simulation.getArrivedNumber()

        x_light, y_light, _ = controller.step(queue_x, queue_y, 1.0)
        set_traffic_light(traci, x_light, y_light)

        step += 1

        if verbose and step % 100 == 0:
            state = controller.get_state()
            print(f"  t={step}s: qX={queue_x} qY={queue_y} "
                  f"phase={state['phase']} target={state['target_green']:.1f}s")

    metrics.sim_time = step
    metrics.switch_count = len(controller.phase_history)

    traci.close()

    if verbose:
        print(f"\n完成: {metrics.sim_time}s")
        print(f"  到达: {metrics.arrived:.0f}")
        print(f"  延误: {metrics.total_delay:.1f} 车秒, avg={metrics.avg_delay:.2f}s")
        print(f"  最大排队: {metrics.max_queue:.0f}")
        print(f"  切换: {metrics.switch_count}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SUMO + VAController")
    parser.add_argument("--scenario", default="balanced",
                        choices=["balanced", "imbalanced", "tidal", "burst"])
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--min-green", type=float, default=10.0)
    parser.add_argument("--max-green", type=float, default=30.0)
    parser.add_argument("--max-red", type=float, default=25.0)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="GUI延迟(ms)，值越大越慢，建议50-500")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    run_simulation(
        scenario=args.scenario,
        duration=args.duration,
        controller_params={
            "min_green": args.min_green,
            "max_green": args.max_green,
            "max_red": args.max_red,
        },
        gui=args.gui,
        delay=args.delay,
        verbose=not args.quiet,
    )
