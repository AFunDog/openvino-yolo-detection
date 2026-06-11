"""
SUMO 路由生成脚本

生成不同交通需求场景的路由文件。
edge 名称来自 netgenerate --grid 生成的网格路网：
  东西向 (X): left0A0 (西→中), right0A0 (东→中)
  南北向 (Y): bottom0A0 (南→中), top0A0 (北→中)
"""

import random
from pathlib import Path

SUMO_DIR = Path(__file__).parent

SCENARIOS = {
    "balanced": {"x": 600, "y": 600},
    "imbalanced": {"x": 900, "y": 300},
}

# 潮汐交通：前后两段
TIDAL_DURATION = 1800


def generate_route(scenario: str, duration: float = 1800.0, seed: int = 42):
    random.seed(seed)
    route_file = SUMO_DIR / f"routes_{scenario}.xml"

    # X 方向入口 → 只能去 Y 方向的出口（不能回头）
    x_from = ["left0A0", "right0A0"]
    y_from = ["bottom0A0", "top0A0"]

    # 出口（不能和入口同方向）
    x_exits = ["A0right0", "A0left0"]   # X 入口对应的出口
    y_exits = ["A0top0", "A0bottom0"]    # Y 入口对应的出口

    vehicles = []
    vid = 0

    if scenario in ("balanced", "imbalanced"):
        flow_x = SCENARIOS[scenario]["x"]
        flow_y = SCENARIOS[scenario]["y"]

        t = 0.0
        while t < duration:
            interval_x = 3600.0 / flow_x if flow_x > 0 else float("inf")
            if random.random() < (1.0 / interval_x):
                # X 方向入口 → 随机去 Y 方向出口（不能回头）
                vehicles.append({
                    "id": f"v{vid}",
                    "depart": t,
                    "from": random.choice(x_from),
                    "to": random.choice(y_exits),
                })
                vid += 1

            interval_y = 3600.0 / flow_y if flow_y > 0 else float("inf")
            if random.random() < (1.0 / interval_y):
                # Y 方向入口 → 随机去 X 方向出口
                vehicles.append({
                    "id": f"v{vid}",
                    "depart": t,
                    "from": random.choice(y_from),
                    "to": random.choice(x_exits),
                })
                vid += 1
            t += 1.0

    elif scenario == "tidal":
        t = 0.0
        while t < duration:
            half = duration / 2
            if t < half:
                flow_x, flow_y = 900, 200
            else:
                flow_x, flow_y = 200, 900

            interval_x = 3600.0 / flow_x
            if random.random() < (1.0 / interval_x):
                vehicles.append({
                    "id": f"v{vid}",
                    "depart": t,
                    "from": random.choice(x_from),
                    "to": random.choice(y_exits),
                })
                vid += 1

            interval_y = 3600.0 / flow_y
            if random.random() < (1.0 / interval_y):
                vehicles.append({
                    "id": f"v{vid}",
                    "depart": t,
                    "from": random.choice(y_from),
                    "to": random.choice(x_exits),
                })
                vid += 1
            t += 1.0

    elif scenario == "burst":
        flow_x = 400
        flow_y = 400
        t = 0.0
        while t < duration:
            interval_x = 3600.0 / flow_x
            if random.random() < (1.0 / interval_x):
                vehicles.append({
                    "id": f"v{vid}",
                    "depart": t,
                    "from": random.choice(x_from),
                    "to": random.choice(y_exits),
                })
                vid += 1

            interval_y = 3600.0 / flow_y
            if random.random() < (1.0 / interval_y):
                vehicles.append({
                    "id": f"v{vid}",
                    "depart": t,
                    "from": random.choice(y_from),
                    "to": random.choice(x_exits),
                })
                vid += 1
            t += 1.0

        # 突发车流：10-12 分钟 X 方向流量
        burst_start = 600.0
        burst_end = 720.0
        burst_flow = 1800
        t = burst_start
        while t < burst_end:
            interval = 3600.0 / burst_flow
            if random.random() < (1.0 / interval):
                vehicles.append({
                    "id": f"v{vid}",
                    "depart": t,
                    "from": random.choice(["left0A0", "right0A0"]),
                    "to": random.choice(["A0right0", "A0left0"]),
                })
                vid += 1
            t += 1.0

    # 写入 XML
    with open(route_file, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes>\n')
        f.write('    <vType id="car" length="5.0" accel="2.6" decel="4.5" maxSpeed="14.0" sigma="0.5"/>\n\n')

        for v in vehicles:
            f.write(
                f'    <vehicle id="{v["id"]}" type="car" depart="{v["depart"]:.1f}">\n'
                f'        <route edges="{v["from"]} {v["to"]}"/>\n'
                f'    </vehicle>\n'
            )

        f.write('</routes>\n')

    print(f"路由已生成: {route_file} ({len(vehicles)} 辆车)")
    return route_file


if __name__ == "__main__":
    for scenario in ["balanced", "imbalanced", "tidal", "burst"]:
        generate_route(scenario)
    print("\n所有场景路由生成完成！")
