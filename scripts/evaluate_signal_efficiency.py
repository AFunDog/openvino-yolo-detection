"""CLI entry for theoretical traffic-efficiency evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithm.traffic_efficiency import evaluate_signal_efficiency, format_report_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评估固定配时 vs 自适应配时的理论通行效率")
    parser.add_argument("summary", help="direction_pair 类型的 summary.json 路径")
    parser.add_argument("--fixed-green-x", type=float, default=20.0, help="固定配时 X 方向绿灯秒数")
    parser.add_argument("--fixed-green-y", type=float, default=20.0, help="固定配时 Y 方向绿灯秒数")
    parser.add_argument("--min-green", type=float, default=10.0, help="自适应最短绿灯秒数")
    parser.add_argument("--max-green", type=float, default=30.0, help="自适应最长绿灯秒数")
    parser.add_argument("--yellow", type=float, default=3.0, help="黄灯秒数")
    parser.add_argument("--sat-x", type=float, default=1.0, help="X 方向饱和放行率，单位 辆/秒")
    parser.add_argument("--sat-y", type=float, default=1.0, help="Y 方向饱和放行率，单位 辆/秒")
    parser.add_argument("--dt", type=float, default=0.5, help="离散仿真步长，单位 秒")
    parser.add_argument("--clearance", type=float, default=60.0, help="观测结束后的清空时间上限，单位 秒")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary_path = Path(args.summary)
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    report = evaluate_signal_efficiency(
        summary,
        fixed_green_x=args.fixed_green_x,
        fixed_green_y=args.fixed_green_y,
        min_green=args.min_green,
        max_green=args.max_green,
        yellow_duration=args.yellow,
        saturation_rate_x=args.sat_x,
        saturation_rate_y=args.sat_y,
        dt=args.dt,
        clearance_time=args.clearance,
    )

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return

    print(format_report_text(report))


if __name__ == "__main__":
    main()
