"""Theoretical traffic-efficiency evaluation for fixed-time vs adaptive control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

from .va_controller import VAController


@dataclass
class DemandProfile:
    duration_x: float
    duration_y: float
    arrival_rate_x: float
    arrival_rate_y: float
    total_x: float
    total_y: float

    @property
    def observed_duration(self) -> float:
        return max(self.duration_x, self.duration_y)

    @property
    def total_arrivals(self) -> float:
        return self.total_x + self.total_y


@dataclass
class SimulationMetrics:
    strategy: str
    sim_time: float
    arrived_x: float
    arrived_y: float
    passed_x: float
    passed_y: float
    total_delay: float
    avg_delay: float
    max_queue_x: float
    max_queue_y: float
    max_queue_total: float
    switch_count: int
    throughput_ratio: float

    @property
    def arrived_total(self) -> float:
        return self.arrived_x + self.arrived_y

    @property
    def passed_total(self) -> float:
        return self.passed_x + self.passed_y

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["arrived_total"] = self.arrived_total
        data["passed_total"] = self.passed_total
        return data


@dataclass
class EvaluationReport:
    assumptions: Dict[str, Any]
    demand: Dict[str, Any]
    fixed_time: SimulationMetrics
    adaptive: SimulationMetrics
    comparison: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assumptions": self.assumptions,
            "demand": self.demand,
            "fixed_time": self.fixed_time.to_dict(),
            "adaptive": self.adaptive.to_dict(),
            "comparison": self.comparison,
        }


@dataclass
class RealtimeEfficiencySnapshot:
    arrived_x: float
    arrived_y: float
    fixed_passed_x: float
    fixed_passed_y: float
    adaptive_passed_x: float
    adaptive_passed_y: float
    fixed_delay: float
    adaptive_delay: float
    fixed_max_queue: float
    adaptive_max_queue: float
    fixed_switch_count: int
    adaptive_switch_count: int

    @property
    def arrived_total(self) -> float:
        return self.arrived_x + self.arrived_y

    @property
    def fixed_passed_total(self) -> float:
        return self.fixed_passed_x + self.fixed_passed_y

    @property
    def adaptive_passed_total(self) -> float:
        return self.adaptive_passed_x + self.adaptive_passed_y

    @property
    def throughput_gain(self) -> float:
        return _gain(self.adaptive_passed_total, self.fixed_passed_total)

    @property
    def delay_reduction(self) -> float:
        return _reduction(self.adaptive_delay, self.fixed_delay)

    @property
    def fixed_avg_delay(self) -> float:
        return _safe_div(self.fixed_delay, self.arrived_total)

    @property
    def adaptive_avg_delay(self) -> float:
        return _safe_div(self.adaptive_delay, self.arrived_total)

    @property
    def avg_delay_reduction(self) -> float:
        return _reduction(self.adaptive_avg_delay, self.fixed_avg_delay)

    @property
    def max_queue_reduction(self) -> float:
        return _reduction(self.adaptive_max_queue, self.fixed_max_queue)


class FixedTimeController:
    """Simple two-phase fixed-time signal controller with yellow intervals."""

    def __init__(self, green_x: float = 20.0, green_y: float = 20.0, yellow_duration: float = 3.0):
        self.green_x = float(green_x)
        self.green_y = float(green_y)
        self.yellow_duration = float(yellow_duration)
        self.reset()

    def reset(self) -> None:
        self._phase = 0
        self._phase_elapsed = 0.0
        self._in_yellow = False
        self._yellow_elapsed = 0.0
        self.switch_count = 0

    def step(self, dt: float):
        if self._in_yellow:
            self._yellow_elapsed += dt
            remaining = max(0.0, self.yellow_duration - self._yellow_elapsed)
            if self._yellow_elapsed >= self.yellow_duration:
                self._in_yellow = False
                self._yellow_elapsed = 0.0
                self._phase = 1 - self._phase
                self._phase_elapsed = 0.0
                self.switch_count += 1
            return "yellow", "yellow", remaining

        self._phase_elapsed += dt
        target = self.green_x if self._phase == 0 else self.green_y
        remaining = max(0.0, target - self._phase_elapsed)
        if self._phase_elapsed >= target:
            self._in_yellow = True
            self._yellow_elapsed = 0.0
            return "yellow", "yellow", self.yellow_duration
        if self._phase == 0:
            return "green", "red", remaining
        return "red", "green", remaining


class RealtimeEfficiencyTracker:
    """Incremental theoretical comparison under the same observed arrivals."""

    def __init__(
        self,
        *,
        fixed_green_x: float = 20.0,
        fixed_green_y: float = 20.0,
        min_green: float = 10.0,
        max_green: float = 30.0,
        yellow_duration: float = 3.0,
        saturation_rate_x: float = 1.0,
        saturation_rate_y: float = 1.0,
    ):
        self.fixed_green_x = float(fixed_green_x)
        self.fixed_green_y = float(fixed_green_y)
        self.min_green = float(min_green)
        self.max_green = float(max_green)
        self.yellow_duration = float(yellow_duration)
        self.saturation_rate_x = float(saturation_rate_x)
        self.saturation_rate_y = float(saturation_rate_y)
        self.reset()

    def reset(self) -> None:
        self.fixed_controller = FixedTimeController(
            green_x=self.fixed_green_x,
            green_y=self.fixed_green_y,
            yellow_duration=self.yellow_duration,
        )
        self.adaptive_controller = VAController(
            min_green=self.min_green,
            max_green=self.max_green,
            yellow_duration=self.yellow_duration,
        )
        self.prev_observed_total_x = 0.0
        self.prev_observed_total_y = 0.0
        self.arrived_x = 0.0
        self.arrived_y = 0.0
        self.fixed_queue_x = 0.0
        self.fixed_queue_y = 0.0
        self.adaptive_queue_x = 0.0
        self.adaptive_queue_y = 0.0
        self.fixed_passed_x = 0.0
        self.fixed_passed_y = 0.0
        self.adaptive_passed_x = 0.0
        self.adaptive_passed_y = 0.0
        self.fixed_delay = 0.0
        self.adaptive_delay = 0.0
        self.fixed_max_queue = 0.0
        self.adaptive_max_queue = 0.0

    def update(
        self,
        *,
        observed_queue_x: float,
        observed_queue_y: float,
        observed_passed_x: float,
        observed_passed_y: float,
        dt: float,
    ) -> RealtimeEfficiencySnapshot:
        observed_total_x = float(observed_queue_x) + float(observed_passed_x)
        observed_total_y = float(observed_queue_y) + float(observed_passed_y)
        arrival_inc_x = max(0.0, observed_total_x - self.prev_observed_total_x)
        arrival_inc_y = max(0.0, observed_total_y - self.prev_observed_total_y)
        self.prev_observed_total_x = observed_total_x
        self.prev_observed_total_y = observed_total_y

        self.arrived_x += arrival_inc_x
        self.arrived_y += arrival_inc_y
        self.fixed_queue_x += arrival_inc_x
        self.fixed_queue_y += arrival_inc_y
        self.adaptive_queue_x += arrival_inc_x
        self.adaptive_queue_y += arrival_inc_y

        fixed_x_state, fixed_y_state, _ = self.fixed_controller.step(dt)
        adaptive_x_state, adaptive_y_state, _ = self.adaptive_controller.step(
            int(round(self.adaptive_queue_x)),
            int(round(self.adaptive_queue_y)),
            dt,
        )

        self.fixed_delay += (self.fixed_queue_x + self.fixed_queue_y) * dt
        self.adaptive_delay += (self.adaptive_queue_x + self.adaptive_queue_y) * dt

        if fixed_x_state == "green":
            depart_x = min(self.fixed_queue_x, self.saturation_rate_x * dt)
            self.fixed_queue_x -= depart_x
            self.fixed_passed_x += depart_x
        if fixed_y_state == "green":
            depart_y = min(self.fixed_queue_y, self.saturation_rate_y * dt)
            self.fixed_queue_y -= depart_y
            self.fixed_passed_y += depart_y

        if adaptive_x_state == "green":
            depart_x = min(self.adaptive_queue_x, self.saturation_rate_x * dt)
            self.adaptive_queue_x -= depart_x
            self.adaptive_passed_x += depart_x
        if adaptive_y_state == "green":
            depart_y = min(self.adaptive_queue_y, self.saturation_rate_y * dt)
            self.adaptive_queue_y -= depart_y
            self.adaptive_passed_y += depart_y

        self.fixed_max_queue = max(self.fixed_max_queue, self.fixed_queue_x + self.fixed_queue_y)
        self.adaptive_max_queue = max(self.adaptive_max_queue, self.adaptive_queue_x + self.adaptive_queue_y)

        return RealtimeEfficiencySnapshot(
            arrived_x=self.arrived_x,
            arrived_y=self.arrived_y,
            fixed_passed_x=self.fixed_passed_x,
            fixed_passed_y=self.fixed_passed_y,
            adaptive_passed_x=self.adaptive_passed_x,
            adaptive_passed_y=self.adaptive_passed_y,
            fixed_delay=self.fixed_delay,
            adaptive_delay=self.adaptive_delay,
            fixed_max_queue=self.fixed_max_queue,
            adaptive_max_queue=self.adaptive_max_queue,
            fixed_switch_count=self.fixed_controller.switch_count,
            adaptive_switch_count=len(self.adaptive_controller.get_history()),
        )


def _safe_rate(total: float, duration: float) -> float:
    return float(total) / float(duration) if duration > 0 else 0.0


def demand_from_summary(summary: Dict[str, Any]) -> DemandProfile:
    if summary.get("session_type") != "direction_pair":
        raise ValueError("仅支持 direction_pair 类型的 summary.json")

    x_info = dict(summary.get("direction_videos", {}).get("X", {}))
    y_info = dict(summary.get("direction_videos", {}).get("Y", {}))

    total_x = float(summary.get("track_count_x", summary.get("line_count_x", x_info.get("line_count_total", 0))))
    total_y = float(summary.get("track_count_y", summary.get("line_count_y", y_info.get("line_count_total", 0))))

    fps_x = float(x_info.get("video_info", {}).get("fps", 0.0) or 0.0)
    fps_y = float(y_info.get("video_info", {}).get("fps", 0.0) or 0.0)
    frames_x = float(x_info.get("total_frames", 0) or 0)
    frames_y = float(y_info.get("total_frames", 0) or 0)

    duration_x = frames_x / fps_x if fps_x > 0 else 0.0
    duration_y = frames_y / fps_y if fps_y > 0 else 0.0
    arrival_rate_x = _safe_rate(total_x, duration_x)
    arrival_rate_y = _safe_rate(total_y, duration_y)

    return DemandProfile(
        duration_x=duration_x,
        duration_y=duration_y,
        arrival_rate_x=arrival_rate_x,
        arrival_rate_y=arrival_rate_y,
        total_x=total_x,
        total_y=total_y,
    )


def _simulate_controller(
    controller: Any,
    profile: DemandProfile,
    saturation_rate_x: float,
    saturation_rate_y: float,
    dt: float,
    clearance_time: float,
    strategy_name: str,
) -> SimulationMetrics:
    queue_x = 0.0
    queue_y = 0.0
    arrived_x = 0.0
    arrived_y = 0.0
    passed_x = 0.0
    passed_y = 0.0
    total_delay = 0.0
    max_queue_x = 0.0
    max_queue_y = 0.0
    max_queue_total = 0.0
    t = 0.0
    observed_duration = profile.observed_duration
    max_time = observed_duration + clearance_time

    while t < max_time:
        if t < profile.duration_x:
            add_x = profile.arrival_rate_x * dt
            queue_x += add_x
            arrived_x += add_x
        if t < profile.duration_y:
            add_y = profile.arrival_rate_y * dt
            queue_y += add_y
            arrived_y += add_y

        if isinstance(controller, VAController):
            x_state, y_state, _ = controller.step(
                int(round(queue_x)),
                int(round(queue_y)),
                dt,
            )
            switch_count = len(controller.get_history())
        else:
            x_state, y_state, _ = controller.step(dt)
            switch_count = controller.switch_count

        total_delay += (queue_x + queue_y) * dt

        if x_state == "green":
            depart_x = min(queue_x, saturation_rate_x * dt)
            queue_x -= depart_x
            passed_x += depart_x
        if y_state == "green":
            depart_y = min(queue_y, saturation_rate_y * dt)
            queue_y -= depart_y
            passed_y += depart_y

        max_queue_x = max(max_queue_x, queue_x)
        max_queue_y = max(max_queue_y, queue_y)
        max_queue_total = max(max_queue_total, queue_x + queue_y)

        t += dt

        if t >= observed_duration and queue_x <= 1e-6 and queue_y <= 1e-6:
            break

    arrived_total = arrived_x + arrived_y
    passed_total = passed_x + passed_y
    return SimulationMetrics(
        strategy=strategy_name,
        sim_time=t,
        arrived_x=arrived_x,
        arrived_y=arrived_y,
        passed_x=passed_x,
        passed_y=passed_y,
        total_delay=total_delay,
        avg_delay=(total_delay / arrived_total) if arrived_total > 0 else 0.0,
        max_queue_x=max_queue_x,
        max_queue_y=max_queue_y,
        max_queue_total=max_queue_total,
        switch_count=switch_count,
        throughput_ratio=(passed_total / arrived_total) if arrived_total > 0 else 0.0,
    )


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _gain(new_value: float, base_value: float) -> float:
    if abs(base_value) <= 1e-9:
        return 0.0
    return (new_value - base_value) / base_value


def _reduction(new_value: float, base_value: float) -> float:
    if abs(base_value) <= 1e-9:
        return 0.0
    return (base_value - new_value) / base_value


def evaluate_signal_efficiency(
    summary: Dict[str, Any],
    *,
    fixed_green_x: float = 20.0,
    fixed_green_y: float = 20.0,
    min_green: float = 10.0,
    max_green: float = 30.0,
    yellow_duration: float = 3.0,
    saturation_rate_x: float = 1.0,
    saturation_rate_y: float = 1.0,
    dt: float = 0.5,
    clearance_time: float = 60.0,
) -> EvaluationReport:
    profile = demand_from_summary(summary)

    fixed_controller = FixedTimeController(
        green_x=fixed_green_x,
        green_y=fixed_green_y,
        yellow_duration=yellow_duration,
    )
    adaptive_controller = VAController(
        min_green=min_green,
        max_green=max_green,
        yellow_duration=yellow_duration,
    )

    fixed = _simulate_controller(
        fixed_controller,
        profile,
        saturation_rate_x=saturation_rate_x,
        saturation_rate_y=saturation_rate_y,
        dt=dt,
        clearance_time=clearance_time,
        strategy_name="fixed_time",
    )
    adaptive = _simulate_controller(
        adaptive_controller,
        profile,
        saturation_rate_x=saturation_rate_x,
        saturation_rate_y=saturation_rate_y,
        dt=dt,
        clearance_time=clearance_time,
        strategy_name="adaptive",
    )

    comparison = {
        "throughput_gain": _gain(adaptive.passed_total, fixed.passed_total),
        "delay_reduction": _reduction(adaptive.total_delay, fixed.total_delay),
        "avg_delay_reduction": _reduction(adaptive.avg_delay, fixed.avg_delay),
        "max_queue_reduction": _reduction(adaptive.max_queue_total, fixed.max_queue_total),
        "switch_count_reduction": _reduction(adaptive.switch_count, fixed.switch_count),
    }

    assumptions = {
        "arrival_model": "uniform_arrival_from_summary_counts",
        "dt_seconds": dt,
        "clearance_time_seconds": clearance_time,
        "fixed_green_x_seconds": fixed_green_x,
        "fixed_green_y_seconds": fixed_green_y,
        "adaptive_min_green_seconds": min_green,
        "adaptive_max_green_seconds": max_green,
        "yellow_seconds": yellow_duration,
        "saturation_rate_x_veh_per_sec": saturation_rate_x,
        "saturation_rate_y_veh_per_sec": saturation_rate_y,
        "note": "理论评估基于 summary.json 的总车流与视频时长，假设车流在观测时间内均匀到达。",
    }
    demand = {
        "duration_x_seconds": profile.duration_x,
        "duration_y_seconds": profile.duration_y,
        "observed_duration_seconds": profile.observed_duration,
        "arrival_rate_x_veh_per_sec": profile.arrival_rate_x,
        "arrival_rate_y_veh_per_sec": profile.arrival_rate_y,
        "total_x": profile.total_x,
        "total_y": profile.total_y,
        "total_arrivals": profile.total_arrivals,
    }
    return EvaluationReport(
        assumptions=assumptions,
        demand=demand,
        fixed_time=fixed,
        adaptive=adaptive,
        comparison=comparison,
    )


def format_report_text(report: EvaluationReport) -> str:
    fixed = report.fixed_time
    adaptive = report.adaptive
    comp = report.comparison
    demand = report.demand
    assumptions = report.assumptions
    return (
        "交通灯理论通行效率评估\n"
        f"观测时长: {demand['observed_duration_seconds']:.1f}s\n"
        f"X/Y 到达率: {demand['arrival_rate_x_veh_per_sec']:.3f} / {demand['arrival_rate_y_veh_per_sec']:.3f} 辆/s\n"
        f"饱和放行率假设: X={assumptions['saturation_rate_x_veh_per_sec']:.3f}, "
        f"Y={assumptions['saturation_rate_y_veh_per_sec']:.3f} 辆/s\n"
        "\n"
        "固定配时:\n"
        f"  通过量: {fixed.passed_total:.2f} / {fixed.arrived_total:.2f} 辆\n"
        f"  累计延误: {fixed.total_delay:.2f} 车秒\n"
        f"  平均等待时长: {fixed.avg_delay:.2f} s/辆\n"
        f"  最大排队: {fixed.max_queue_total:.2f} 辆\n"
        f"  切换次数: {fixed.switch_count}\n"
        "\n"
        "自适应配时:\n"
        f"  通过量: {adaptive.passed_total:.2f} / {adaptive.arrived_total:.2f} 辆\n"
        f"  累计延误: {adaptive.total_delay:.2f} 车秒\n"
        f"  平均等待时长: {adaptive.avg_delay:.2f} s/辆\n"
        f"  最大排队: {adaptive.max_queue_total:.2f} 辆\n"
        f"  切换次数: {adaptive.switch_count}\n"
        "\n"
        "对比结果:\n"
        f"  理论通过量提升: {comp['throughput_gain'] * 100:.2f}%\n"
        f"  累计延误下降: {comp['delay_reduction'] * 100:.2f}%\n"
        f"  平均等待时长下降: {comp['avg_delay_reduction'] * 100:.2f}%\n"
        f"  最大排队下降: {comp['max_queue_reduction'] * 100:.2f}%\n"
        f"  切换次数下降: {comp['switch_count_reduction'] * 100:.2f}%\n"
    )


def format_realtime_snapshot_text(snapshot: RealtimeEfficiencySnapshot) -> str:
    return (
        f"累计到达: X={snapshot.arrived_x:.1f}  Y={snapshot.arrived_y:.1f}  总={snapshot.arrived_total:.1f}\n"
        f"固定配时通过: {snapshot.fixed_passed_total:.1f}  自适应通过: {snapshot.adaptive_passed_total:.1f}\n"
        f"理论通过量提升: {snapshot.throughput_gain * 100:.2f}%\n"
        f"固定累计延误: {snapshot.fixed_delay:.1f}  自适应累计延误: {snapshot.adaptive_delay:.1f}  延误下降: {snapshot.delay_reduction * 100:.2f}%\n"
        f"固定平均等待: {snapshot.fixed_avg_delay:.2f}s/辆  自适应平均等待: {snapshot.adaptive_avg_delay:.2f}s/辆  平均等待下降: {snapshot.avg_delay_reduction * 100:.2f}%\n"
        f"固定最大排队: {snapshot.fixed_max_queue:.1f}  自适应最大排队: {snapshot.adaptive_max_queue:.1f}  排队下降: {snapshot.max_queue_reduction * 100:.2f}%\n"
        f"切换次数: 固定={snapshot.fixed_switch_count}  自适应={snapshot.adaptive_switch_count}"
    )
