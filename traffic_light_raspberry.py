#!/usr/bin/env python3
"""
树莓派交通灯控制器
通过GPIO控制红黄绿三色LED灯
"""

import time
import sys
from enum import Enum

# 尝试导入RPi.GPIO，如果在非树莓派环境则使用模拟模式
try:
    import RPi.GPIO as GPIO
    SIMULATION_MODE = False
except ImportError:
    print("警告: 未找到RPi.GPIO模块，运行在模拟模式")
    SIMULATION_MODE = True


class LightColor(Enum):
    """灯颜色枚举"""
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    OFF = "off"


class TrafficLightRaspberry:
    """树莓派交通灯控制器"""

    # GPIO引脚定义 (BCM编码)
    # X路红绿灯
    PIN_X_RED = 17      # GPIO17
    PIN_X_YELLOW = 27   # GPIO27
    PIN_X_GREEN = 22    # GPIO22

    # Y路红绿灯
    PIN_Y_RED = 23      # GPIO23
    PIN_Y_YELLOW = 24   # GPIO24
    PIN_Y_GREEN = 25    # GPIO25

    def __init__(self):
        # 标准红灯时长 (秒)
        self.std_light_on_seconds = 10.0
        # 最小亮灯时长
        self.min_light_on_seconds = 10.0
        # 最大亮灯时长
        self.max_light_on_seconds = 30.0

        # X路当前红灯时长
        self.cur_seconds_x_red = self.std_light_on_seconds
        # X路当前绿灯时长
        self.cur_seconds_x_green = self.std_light_on_seconds

        # Y路当前红灯时长
        self.cur_seconds_y_red = self.std_light_on_seconds
        # Y路当前绿灯时长
        self.cur_seconds_y_green = self.std_light_on_seconds

        # X路当前灯颜色
        self.cur_color_x = LightColor.RED
        # Y路当前灯颜色
        self.cur_color_y = LightColor.GREEN

        # 车辆计数
        self.total_car_x = 0
        self.total_car_y = 0

        # 初始化GPIO
        if not SIMULATION_MODE:
            self._setup_gpio()

    def _setup_gpio(self):
        """初始化GPIO设置"""
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        # 设置所有引脚为输出模式
        pins = [
            self.PIN_X_RED, self.PIN_X_YELLOW, self.PIN_X_GREEN,
            self.PIN_Y_RED, self.PIN_Y_YELLOW, self.PIN_Y_GREEN
        ]

        for pin in pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)  # 初始状态：全部熄灭

        print("GPIO初始化完成")

    def _set_light(self, road: str, color: LightColor):
        """设置指定路的灯状态"""
        if SIMULATION_MODE:
            # 模拟模式：只在控制台打印
            color_names = {
                LightColor.RED: "红灯",
                LightColor.YELLOW: "黄灯",
                LightColor.GREEN: "绿灯",
                LightColor.OFF: "熄灭"
            }
            print(f"  [{road}路] {color_names[color]}")
            return

        # 实际GPIO控制
        if road == "X":
            pins = {
                LightColor.RED: self.PIN_X_RED,
                LightColor.YELLOW: self.PIN_X_YELLOW,
                LightColor.GREEN: self.PIN_X_GREEN,
                LightColor.OFF: None
            }
            # 先熄灭所有X路灯
            GPIO.output(self.PIN_X_RED, GPIO.LOW)
            GPIO.output(self.PIN_X_YELLOW, GPIO.LOW)
            GPIO.output(self.PIN_X_GREEN, GPIO.LOW)
        else:
            pins = {
                LightColor.RED: self.PIN_Y_RED,
                LightColor.YELLOW: self.PIN_Y_YELLOW,
                LightColor.GREEN: self.PIN_Y_GREEN,
                LightColor.OFF: None
            }
            # 先熄灭所有Y路灯
            GPIO.output(self.PIN_Y_RED, GPIO.LOW)
            GPIO.output(self.PIN_Y_YELLOW, GPIO.LOW)
            GPIO.output(self.PIN_Y_GREEN, GPIO.LOW)

        # 点亮指定颜色的灯
        if pins[color] is not None:
            GPIO.output(pins[color], GPIO.HIGH)

    def control_light(self, road: str, color: LightColor):
        """控制指定路的灯"""
        if road == "X":
            self.cur_color_x = color
            # Y路颜色与X路相反（黄灯时Y路也黄灯）
            if color == LightColor.RED:
                self.cur_color_y = LightColor.GREEN
                self._set_light("Y", LightColor.GREEN)
            elif color == LightColor.GREEN:
                self.cur_color_y = LightColor.RED
                self._set_light("Y", LightColor.RED)
            else:  # YELLOW or OFF
                self.cur_color_y = color
                self._set_light("Y", color)

            self._set_light("X", color)
        else:
            self.cur_color_y = color
            self._set_light("Y", color)

    def set_total_car(self, car_count_x: int, car_count_y: int):
        """
        设置车辆数量并调整灯时长

        逻辑:
        - 基本红灯时长为10秒
        - 当X路车辆多于Y路时，X路红灯时长减少20%，Y路红灯时长增加20%
        - 最长不超过30秒，最短不低于10秒
        """
        print(f"\n>>> 更新车辆数: X路={car_count_x}, Y路={car_count_y}")
        self.total_car_x = car_count_x
        self.total_car_y = car_count_y

        if self.total_car_x > self.total_car_y:
            # X路车多，减少X路红灯时长，增加X路绿灯时长
            self.cur_seconds_x_red -= self.std_light_on_seconds * 0.2
            if self.cur_seconds_x_red < self.min_light_on_seconds:
                self.cur_seconds_x_red = self.min_light_on_seconds

            self.cur_seconds_x_green += self.std_light_on_seconds * 0.2
            if self.cur_seconds_x_green > self.max_light_on_seconds:
                self.cur_seconds_x_green = self.max_light_on_seconds
        else:
            # Y路车多或相等，增加X路红灯时长，减少X路绿灯时长
            self.cur_seconds_x_red += self.std_light_on_seconds * 0.2
            if self.cur_seconds_x_red > self.max_light_on_seconds:
                self.cur_seconds_x_red = self.max_light_on_seconds

            self.cur_seconds_x_green -= self.std_light_on_seconds * 0.2
            if self.cur_seconds_x_green < self.min_light_on_seconds:
                self.cur_seconds_x_green = self.min_light_on_seconds

        # Y路时长与X路相反
        self.cur_seconds_y_red = self.cur_seconds_x_green
        self.cur_seconds_y_green = self.cur_seconds_x_red

        print(f"    调整后 - X路红灯: {self.cur_seconds_x_red:.1f}s, X路绿灯: {self.cur_seconds_x_green:.1f}s")
        print(f"           Y路红灯: {self.cur_seconds_y_red:.1f}s, Y路绿灯: {self.cur_seconds_y_green:.1f}s")

    def yellow_transition(self):
        """黄灯过渡（3秒，每0.6秒闪烁一次，共5次）"""
        print("\n>>> 黄灯过渡阶段 (3秒)")
        splash_time = 3.0
        flash_count = 0

        while splash_time > 0:
            flash_count += 1
            # 黄灯亮
            self.control_light("X", LightColor.YELLOW)
            time.sleep(0.6)

            splash_time -= 0.6
            if splash_time <= 0:
                break

            # 灯灭
            self.control_light("X", LightColor.OFF)
            time.sleep(0.6)
            splash_time -= 0.6

        print(f">>> 黄灯闪烁 {flash_count} 次，过渡完成")

    def loop_lights(self):
        """主循环：控制交通灯切换"""
        print("\n" + "=" * 50)
        print("     交通灯控制系统启动")
        if SIMULATION_MODE:
            print("     [模拟模式]")
        else:
            print("     [GPIO控制模式]")
        print("=" * 50)

        try:
            while True:
                # 根据当前颜色决定亮灯时长
                if self.cur_color_x == LightColor.RED:
                    # X路红灯，Y路绿灯
                    self.control_light("X", LightColor.RED)
                    wait_seconds = self.cur_seconds_x_red
                    print(f"\n>>> X路红灯 ({wait_seconds:.1f}s), Y路绿灯")
                else:
                    # X路绿灯，Y路红灯
                    self.control_light("X", LightColor.GREEN)
                    wait_seconds = self.cur_seconds_x_green
                    print(f"\n>>> X路绿灯 ({wait_seconds:.1f}s), Y路红灯")

                # 等待
                time.sleep(wait_seconds)

                # 切换颜色
                if self.cur_color_x == LightColor.RED:
                    self.cur_color_x = LightColor.GREEN
                else:
                    self.cur_color_x = LightColor.RED

                # 黄灯过渡
                self.yellow_transition()

        except KeyboardInterrupt:
            print("\n\n>>> 交通灯控制系统已停止")
            self.cleanup()

    def cleanup(self):
        """清理GPIO资源"""
        if not SIMULATION_MODE:
            # 熄灭所有灯
            self._set_light("X", LightColor.OFF)
            self._set_light("Y", LightColor.OFF)
            GPIO.cleanup()
            print("GPIO资源已清理")

    def test_lights(self):
        """测试所有灯"""
        print("\n>>> 测试所有灯")

        # X路灯测试
        print("X路 - 红灯")
        self._set_light("X", LightColor.RED)
        time.sleep(1)

        print("X路 - 黄灯")
        self._set_light("X", LightColor.YELLOW)
        time.sleep(1)

        print("X路 - 绿灯")
        self._set_light("X", LightColor.GREEN)
        time.sleep(1)

        print("X路 - 熄灭")
        self._set_light("X", LightColor.OFF)

        # Y路灯测试
        print("Y路 - 红灯")
        self._set_light("Y", LightColor.RED)
        time.sleep(1)

        print("Y路 - 黄灯")
        self._set_light("Y", LightColor.YELLOW)
        time.sleep(1)

        print("Y路 - 绿灯")
        self._set_light("Y", LightColor.GREEN)
        time.sleep(1)

        print("Y路 - 熄灭")
        self._set_light("Y", LightColor.OFF)

        print(">>> 测试完成")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="树莓派交通灯控制器")
    parser.add_argument("--test", action="store_true", help="测试所有灯")
    parser.add_argument("--x", type=int, default=5, help="X路车辆数 (默认: 5)")
    parser.add_argument("--y", type=int, default=3, help="Y路车辆数 (默认: 3)")

    args = parser.parse_args()

    light = TrafficLightRaspberry()

    if args.test:
        light.test_lights()
    else:
        # 设置初始车辆数
        light.set_total_car(args.x, args.y)
        # 启动主循环
        light.loop_lights()


if __name__ == "__main__":
    main()
