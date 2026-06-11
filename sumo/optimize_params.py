"""
VAController 参数优化脚本

使用网格搜索或贝叶斯优化寻找最优的 VA 控制器参数：
- min_green: 最短绿灯
- max_green: 最长绿灯
- max_red: 最长红灯

优化目标：最小化平均延误
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import asdict

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sumo.sumo_sim import run_simulation, SimulationMetrics


# 参数搜索空间
PARAM_GRID = {
    "min_green": [5.0, 10.0, 15.0],
    "max_green": [20.0, 30.0, 40.0, 50.0],
    "max_red": [30.0, 45.0, 60.0],
}


def grid_search(
    scenarios: List[str],
    duration: float = 1800.0,
    verbose: bool = True,
) -> Tuple[Dict, List[Dict]]:
    """
    网格搜索参数优化

    Args:
        scenarios: 要测试的交通场景列表
        duration: 每个场景的仿真时长
        verbose: 是否打印进度

    Returns:
        (best_params, all_results)
    """
    results = []
    best_avg_delay = float('inf')
    best_params = None

    # 生成所有参数组合
    param_combos = []
    for min_green in PARAM_GRID["min_green"]:
        for max_green in PARAM_GRID["max_green"]:
            for max_red in PARAM_GRID["max_red"]:
                if min_green >= max_green:
                    continue  # 跳过无效组合
                param_combos.append({
                    "min_green": min_green,
                    "max_green": max_green,
                    "max_red": max_red,
                })

    total = len(param_combos) * len(scenarios)
    current = 0

    for params in param_combos:
        scenario_delays = []

        for scenario in scenarios:
            current += 1
            if verbose:
                print(f"[{current}/{total}] scenario={scenario}, "
                      f"params={params}")

            try:
                metrics = run_simulation(
                    scenario=scenario,
                    duration=duration,
                    controller_params=params,
                    gui=False,
                    verbose=False,
                )
                scenario_delays.append(metrics.avg_delay)

                results.append({
                    "scenario": scenario,
                    "params": params,
                    "metrics": {
                        "avg_delay": metrics.avg_delay,
                        "total_delay": metrics.total_delay,
                        "passed_total": metrics.passed_total,
                        "max_queue_total": metrics.max_queue_total,
                        "switch_count": metrics.switch_count,
                    }
                })

            except Exception as e:
                print(f"  仿真失败: {e}")
                scenario_delays.append(float('inf'))

        # 计算平均延误
        avg_delay = sum(scenario_delays) / len(scenario_delays) if scenario_delays else float('inf')

        if avg_delay < best_avg_delay:
            best_avg_delay = avg_delay
            best_params = params.copy()
            if verbose:
                print(f"  -> 新最优: avg_delay={avg_delay:.2f}")

    return best_params, results


def optimize(
    scenarios: List[str] = None,
    duration: float = 1800.0,
    output_file: str = None,
) -> Dict:
    """
    运行参数优化

    Args:
        scenarios: 场景列表，默认所有场景
        duration: 仿真时长
        output_file: 结果输出文件

    Returns:
        最优参数和结果摘要
    """
    if scenarios is None:
        scenarios = ["balanced", "imbalanced", "tidal", "burst"]

    print("=" * 60)
    print("VAController 参数优化")
    print("=" * 60)
    print(f"场景: {scenarios}")
    print(f"仿真时长: {duration}s")
    print(f"参数空间: {PARAM_GRID}")
    print("=" * 60)

    best_params, all_results = grid_search(scenarios, duration)

    print("\n" + "=" * 60)
    print("优化结果")
    print("=" * 60)
    print(f"最优参数: {best_params}")

    # 计算最优参数在各场景的表现
    print("\n最优参数在各场景的表现:")
    for scenario in scenarios:
        for r in all_results:
            if r["scenario"] == scenario and r["params"] == best_params:
                m = r["metrics"]
                print(f"  {scenario}: avg_delay={m['avg_delay']:.2f}s, "
                      f"passed={m['passed_total']:.0f}, "
                      f"max_queue={m['max_queue_total']:.0f}")

    # 保存结果
    output = {
        "best_params": best_params,
        "param_grid": PARAM_GRID,
        "scenarios": scenarios,
        "duration": duration,
        "all_results": all_results,
    }

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {output_file}")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VAController 参数优化")
    parser.add_argument("--scenarios", nargs="+",
                        default=["balanced", "imbalanced", "tidal", "burst"],
                        help="交通场景列表")
    parser.add_argument("--duration", type=float, default=600.0,
                        help="仿真时长 (秒)")
    parser.add_argument("--output", default="sumo/optimization_results.json",
                        help="结果输出文件")

    args = parser.parse_args()

    optimize(
        scenarios=args.scenarios,
        duration=args.duration,
        output_file=args.output,
    )
