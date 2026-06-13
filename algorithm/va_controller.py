"""
Vehicle-Actuated 交通灯控制器

基于比较法的自适应控制，融合提前切换规则：
  1. 最小绿灯未过 → KEEP
  2. 对向红灯超时 → SWITCH
  3. 本向清空 + 对向有车 → SWITCH (提前让路)
  4. 对向排队远超本向 → SWITCH (压力失衡)
  5. 达到目标绿灯 → SWITCH (比例上限兜底)
  6. 其他 → KEEP

参数:
  min_green: 10s  — 最短绿灯
  max_green: 30s  — 最长绿灯
  max_red:   25s  — 最长红灯 (SUMO优化结果)
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
    """Vehicle-Actuated 交通灯控制器"""

    def __init__(
        self,
        min_green: float = 10.0,
        max_green: float = 30.0,
        max_red: float = 25.0,
        yellow_duration: float = 3.0,
    ):
        self.min_green = min_green
        self.max_green = max_green
        self.max_red = max_red
        self.yellow_duration = yellow_duration

        self._phase: int = 0
        self._phase_elapsed: float = 0.0
        self._red_elapsed: float = 0.0
        self._in_yellow: bool = False
        self._yellow_elapsed: float = 0.0
        self._sim_time: float = 0.0
        self._cycle_num: int = 0
        self._target_green: float = min_green
        self._target_reason: str = "初始默认"

        self.last_queue_x: int = 0
        self.last_queue_y: int = 0
        self.last_target_green: float = min_green
        self.last_compare_ratio: float = 0.5

        self.phase_history: List[PhaseRecord] = []

    # ── 主接口 ────────────────────────────────────────────

    def step(
        self,
        queue_x: int,
        queue_y: int,
        dt: float,
    ) -> Tuple[str, str, Optional[float]]:
        self._sim_time += dt
        self.last_queue_x = queue_x
        self.last_queue_y = queue_y

        if self._in_yellow:
            self._yellow_elapsed += dt
            remaining = self.yellow_duration - self._yellow_elapsed
            flash = int(self._yellow_elapsed * 3) % 2 == 0
            x_state = "yellow" if flash else "off"
            y_state = "yellow" if flash else "off"
            if self._yellow_elapsed >= self.yellow_duration:
                self._end_yellow()
            return x_state, y_state, max(0, remaining)

        self._phase_elapsed += dt
        if self._phase == 0:
            self._red_elapsed = 0.0   # X 绿灯，Y 红灯计时从零开始
        else:
            self._red_elapsed += dt   # Y 绿灯，X 红灯累计

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

        remaining = self._remaining_green()
        if self._phase == 0:
            return "green", "red", remaining
        else:
            return "red", "green", remaining

    # ── 决策 ──────────────────────────────────────────────

    def _decide(self) -> Tuple[bool, str]:
        q_cur   = self.last_queue_x if self._phase == 0 else self.last_queue_y
        q_other = self.last_queue_y if self._phase == 0 else self.last_queue_x

        if self._phase_elapsed < self.min_green:
            return False, "最小绿灯"

        if self._red_elapsed >= self.max_red:
            return True, f"对向红灯{self._red_elapsed:.0f}s≥{self.max_red:.0f}s"

        if q_cur == 0 and q_other > 0:
            return True, "本向清空，提前切换"

        if q_other > q_cur * 1.5 and q_other >= 3:
            return True, f"对向压力{q_other}>>本向{q_cur}"

        if self._phase_elapsed >= self._target_green:
            return True, self._target_reason

        return False, ""

    def _remaining_green(self) -> float:
        return max(0.0, self._target_green - self._phase_elapsed)

    def _compute_target_green(self):
        """根据切换时刻车辆数计算目标绿灯"""
        q_cur   = self.last_queue_x if self._phase == 0 else self.last_queue_y
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
        self._red_elapsed = 0.0
        self._in_yellow = False
        self._cycle_num += 1
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
