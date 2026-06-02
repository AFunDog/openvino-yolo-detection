"""
Vehicle-Actuated 交通灯控制器

感应式控制 —— 交通工程中最广泛使用的实战算法。

决策逻辑 (每决策步):
  1. 黄灯过渡中 → 自动推进
  2. 已过 < 最小绿灯 → KEEP (安全约束)
  3. 当前方向已清空 (gap > 阈值) → SWITCH
  4. 已过 >= 最大绿灯 → SWITCH (最大约束)
  5. 对向累计等待时间 > 当前 × 1.2 且 当前无排队 → SWITCH
  6. 其他 → KEEP

参数表:
  min_green: 10s  — 最短绿灯
  max_green: 30s  — 最长绿灯
  max_red:   45s  — 最长红灯
  yellow:     3s  — 黄灯过渡
  gap:        3s  — 连续无车判定为"已清空"
  wait_ratio: 1.2 — 对方等待超过当前 1.2 倍则触发切换
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class PhaseRecord:
    """每次相位切换的记录"""
    cycle: int
    phase: str                # 'X' or 'Y'
    duration: float           # 绿灯时长 (s)
    reason: str               # 切换原因
    queue_x: int
    queue_y: int
    wait_x: float
    wait_y: float


class VAController:
    """感应式交通灯控制器"""

    def __init__(
        self,
        min_green: float = 10.0,
        max_green: float = 30.0,
        max_red: float = 45.0,
        yellow_duration: float = 3.0,
        gap_seconds: float = 3.0,
        wait_ratio: float = 1.2,
    ):
        self.min_green = min_green
        self.max_green = max_green
        self.max_red = max_red
        self.yellow_duration = yellow_duration
        self.gap_seconds = gap_seconds
        self.wait_ratio = wait_ratio

        self._phase: int = 0            # 0=X, 1=Y
        self._phase_elapsed: float = 0.0
        self._red_elapsed: float = 0.0
        self._in_yellow: bool = False
        self._yellow_elapsed: float = 0.0
        self._sim_time: float = 0.0
        self._cycle_num: int = 0

        # 上一帧的特征 (用于显示)
        self.last_queue_x: int = 0
        self.last_queue_y: int = 0
        self.last_wait_x: float = 0.0
        self.last_wait_y: float = 0.0
        self.last_gap_x: float = 0.0
        self.last_gap_y: float = 0.0

        # 历史
        self.phase_history: List[PhaseRecord] = []

    # ── 主接口 ────────────────────────────────────────────

    def step(
        self,
        queue_x: int,
        queue_y: int,
        wait_x: float,
        wait_y: float,
        gap_x: float,
        gap_y: float,
        dt: float,
    ) -> Tuple[str, str, Optional[float]]:
        """
        推进仿真一步。

        Args:
            queue_x, queue_y: X/Y 路当前排队车辆数
            wait_x, wait_y:   X/Y 路总等待时间 (秒)
            gap_x, gap_y:     X/Y 路已清空时长 (秒)
            dt:               时间步长

        Returns:
            (x_light, y_light, countdown)
        """
        self._sim_time += dt

        self.last_queue_x = queue_x
        self.last_queue_y = queue_y
        self.last_wait_x = wait_x
        self.last_wait_y = wait_y
        self.last_gap_x = gap_x
        self.last_gap_y = gap_y

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
            # 记录本轮相位
            self.phase_history.append(PhaseRecord(
                cycle=self._cycle_num,
                phase="X" if self._phase == 0 else "Y",
                duration=self._phase_elapsed,
                reason=reason,
                queue_x=self.last_queue_x,
                queue_y=self.last_queue_y,
                wait_x=self.last_wait_x,
                wait_y=self.last_wait_y,
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
        phase_str = "X" if self._phase == 0 else "Y"

        # 当前方向的队列特征
        q_cur  = self.last_queue_x if self._phase == 0 else self.last_queue_y
        q_other = self.last_queue_y if self._phase == 0 else self.last_queue_x
        w_cur  = self.last_wait_x if self._phase == 0 else self.last_wait_y
        w_other = self.last_wait_y if self._phase == 0 else self.last_wait_x
        gap_cur = self.last_gap_x if self._phase == 0 else self.last_gap_y
        red_other = self._phase_elapsed  # 对方红灯持续时间 ≈ 当前绿灯已过时间

        # 1. 最小绿灯
        if self._phase_elapsed < self.min_green:
            return False, "最小绿灯"

        # 2. 清空检测: 当前方向连续无车 → 切换
        if gap_cur >= self.gap_seconds:
            return True, f"已清空({gap_cur:.1f}s)"

        # 3. 最大绿灯
        if self._phase_elapsed >= self.max_green:
            return True, "最大绿灯"

        # 4. 等待时间加权: 对方积累等待 > 当前 × ratio, 且当前无排队
        if q_cur == 0 and w_other > 5.0 and w_other > w_cur * self.wait_ratio:
            return True, f"等待加权(对方{w_other:.0f}s > 己方{w_cur:.0f}s×{self.wait_ratio})"

        return False, ""

    def _remaining_green(self) -> float:
        if self._phase_elapsed < self.min_green:
            return self.min_green - self._phase_elapsed
        return max(0, self.max_green - self._phase_elapsed)

    def _end_yellow(self):
        old = self._phase
        self._phase = 1 - self._phase
        self._phase_elapsed = 0.0
        self._in_yellow = False
        self._cycle_num += 1

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
            "wait_x": self.last_wait_x,
            "wait_y": self.last_wait_y,
            "gap_x": self.last_gap_x,
            "gap_y": self.last_gap_y,
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
        self.phase_history.clear()
        self.last_queue_x = 0
        self.last_queue_y = 0
        self.last_wait_x = 0.0
        self.last_wait_y = 0.0
        self.last_gap_x = 0.0
        self.last_gap_y = 0.0
