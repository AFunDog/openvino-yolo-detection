"""
Vehicle-Actuated 交通灯控制器

简化比较法 —— 仅根据 X/Y 两个方向的车辆数量比较来决定绿灯时长。

决策逻辑 (每决策步):
  1. 黄灯过渡中 → 自动推进
  2. 已过 < 最小绿灯 → KEEP (安全约束)
  3. 对向红灯时长 ≥ max_red → SWITCH (强制切换，防止无限等待)
  4. 计算当前相位的目标绿灯时长:
       target = min_green + (max_green - min_green) * current / max(current + other, 1)
     其中 current/other 只取当前方向与对向方向的车辆数量
  5. 当前绿灯已达到 target → SWITCH
  6. 其他 → KEEP

参数表:
  min_green: 10s  — 最短绿灯
  max_green: 30s  — 最长绿灯
  max_red:   45s  — 最长红灯（对向红灯超时强制切换）
  yellow:     3s  — 黄灯过渡
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class PhaseRecord:
    cycle: int
    phase: str
    duration: float
    reason: str
    queue_x: int
    queue_y: int


class VAController:
    """基于车辆数量比较的交通灯控制器"""

    def __init__(
        self,
        min_green: float = 10.0,
        max_green: float = 30.0,
        max_red: float = 45.0,
        yellow_duration: float = 3.0,
    ):
        self.min_green = min_green
        self.max_green = max_green
        self.max_red = max_red
        self.yellow_duration = yellow_duration

        self._phase: int = 0            # 0=X, 1=Y
        self._phase_elapsed: float = 0.0
        self._red_elapsed: float = 0.0
        self._in_yellow: bool = False
        self._yellow_elapsed: float = 0.0
        self._sim_time: float = 0.0
        self._cycle_num: int = 0
        self._target_green: float = min_green  # 当前相位的目标绿灯时长（切换时确定）
        self._target_reason: str = "初始默认"  # 目标绿灯的来源原因

        # 上一帧的特征 (用于显示)
        self.last_queue_x: int = 0
        self.last_queue_y: int = 0
        self.last_target_green: float = min_green
        self.last_compare_ratio: float = 0.5

        # 历史
        self.phase_history: List[PhaseRecord] = []

    # ── 主接口 ────────────────────────────────────────────

    def step(
        self,
        queue_x: int,
        queue_y: int,
        dt: float,
    ) -> Tuple[str, str, Optional[float]]:
        """
        推进仿真一步。

        Args:
            queue_x, queue_y: X/Y 路当前排队车辆数
            dt:               时间步长

        Returns:
            (x_light, y_light, countdown)
        """
        self._sim_time += dt

        self.last_queue_x = queue_x
        self.last_queue_y = queue_y

        # ── 黄灯过渡 ──
        if self._in_yellow:
            self._yellow_elapsed += dt
            remaining = self.yellow_duration - self._yellow_elapsed
            flash = int(self._yellow_elapsed * 3) % 2 == 0
            x_state = "yellow" if flash else "off"
            y_state = "yellow" if flash else "off"

            if self._yellow_elapsed >= self.yellow_duration:
                self._end_yellow()
            return x_state, y_state, max(0, remaining)

        # ── 正常决策 ──
        self._phase_elapsed += dt
        if self._phase == 0:
            self._red_elapsed = 0.0  # X绿时 Y红灯计时
        else:
            self._red_elapsed += dt

        should_switch, reason = self._decide()

        if should_switch:
            self.phase_history.append(PhaseRecord(
                cycle=self._cycle_num,
                phase="X" if self._phase == 0 else "Y",
                duration=self._phase_elapsed,
                reason=reason,
                queue_x=self.last_queue_x,
                queue_y=self.last_queue_y,
            ))
            self._in_yellow = True
            self._yellow_elapsed = 0.0
            return "yellow", "yellow", self.yellow_duration

        # 继续
        remaining = self._remaining_green()
        if self._phase == 0:
            return "green", "red", remaining
        else:
            return "red", "green", remaining

    # ── 决策 ──────────────────────────────────────────────

    def _decide(self) -> Tuple[bool, str]:
        # 1. 最小绿灯
        if self._phase_elapsed < self.min_green:
            return False, "最小绿灯"

        # 2. 对向红灯超时 → 强制切换
        if self._red_elapsed >= self.max_red:
            return True, f"对向红灯{self._red_elapsed:.0f}s≥{self.max_red:.0f}s"

        # 3. 目标绿灯时长（切换时已确定，此处直接使用）
        if self._phase_elapsed >= self._target_green:
            return True, self._target_reason

        return False, ""

    def _remaining_green(self) -> float:
        return max(0.0, self._target_green - self._phase_elapsed)

    def _compute_target_green(self):
        """根据切换时刻的车辆数计算目标绿灯时长"""
        q_cur  = self.last_queue_x if self._phase == 0 else self.last_queue_y
        q_other = self.last_queue_y if self._phase == 0 else self.last_queue_x
        total = q_cur + q_other

        if total <= 0:
            target_green = self.min_green
            ratio = 0.5
            reason = f"无车辆，最小绿灯{self.min_green:.0f}s"
        elif q_cur <= 0 and q_other > 0:
            target_green = self.min_green
            ratio = 0.0
            reason = f"本向无车({q_cur})，最小绿灯{self.min_green:.0f}s"
        elif q_other <= 0 and q_cur > 0:
            target_green = self.max_green
            ratio = 1.0
            reason = f"对向无车({q_other})，最大绿灯{self.max_green:.0f}s"
        else:
            ratio = q_cur / total
            target_green = self.min_green + (self.max_green - self.min_green) * ratio
            reason = f"比较法(当前{q_cur}/对向{q_other}={ratio:.0%} -> 目标{target_green:.1f}s)"

        self._target_green = target_green
        self._target_reason = reason
        self.last_target_green = target_green
        self.last_compare_ratio = ratio

    def _end_yellow(self):
        self._phase = 1 - self._phase
        self._phase_elapsed = 0.0
        self._in_yellow = False
        self._cycle_num += 1

        # 切换时根据当前车辆数确定下一次的目标绿灯时长
        self._compute_target_green()

    # ── 状态查询 ──────────────────────────────────────────

    def get_state(self) -> dict:
        return {
            "phase": "X" if self._phase == 0 else "Y",
            "phase_elapsed": self._phase_elapsed,
            "in_yellow": self._in_yellow,
            "yellow_elapsed": self._yellow_elapsed,
            "sim_time": self._sim_time,
            "cycle_num": self._cycle_num,
            "queue_x": self.last_queue_x,
            "queue_y": self.last_queue_y,
            "target_green": self.last_target_green,
            "compare_ratio": self.last_compare_ratio,
        }

    def get_history(self) -> List[PhaseRecord]:
        return self.phase_history

    def reset(self):
        self._phase = 0
        self._phase_elapsed = 0.0
        self._red_elapsed = 0.0
        self._in_yellow = False
        self._yellow_elapsed = 0.0
        self._sim_time = 0.0
        self._cycle_num = 0
        self._target_green = self.min_green
        self._target_reason = "初始默认"
        self.phase_history.clear()
        self.last_queue_x = 0
        self.last_queue_y = 0
        self.last_target_green = self.min_green
        self.last_compare_ratio = 0.5
