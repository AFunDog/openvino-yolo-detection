# SUMO 交通仿真模块

使用 SUMO 微观交通仿真器验证和优化 VA 交通灯控制算法。

## 安装依赖

### 1. 安装 SUMO

从官网下载安装包：https://sumo.dlr.de/docs/Downloads.html

Windows 用户下载 `sumo-win64-xxx.zip`，解压后将 `bin` 目录添加到 PATH：

```powershell
# 例如解压到 C:\sumo
[Environment]::SetEnvironmentVariable("SUMO_HOME", "C:\sumo", "User")
[Environment]::SetEnvironmentVariable("PATH", "$env:PATH;C:\sumo\bin", "User")
```

### 2. 安装 Python 接口

```bash
pip install traci
```

### 3. 验证安装

```bash
sumo --version
python -c "import traci; print('traci OK')"
```

## 文件结构

```
sumo/
├── network/
│   ├── generate.py              # 路网生成 (netgenerate + netconvert)
│   └── intersection.net.xml     # 生成的路网文件
├── routes/
│   ├── generate.py              # 路由生成 (4 种交通场景)
│   ├── balanced.xml             # 均衡场景
│   ├── imbalanced.xml           # 不均衡场景
│   ├── tidal.xml                # 潮汐场景
│   └── burst.xml                # 突发场景
├── sumo_sim.py                  # TraCI 仿真主循环
├── compare_strategies.py        # VA vs 固定配时对比
├── optimize_params.py           # 参数优化
└── README.md
```

## 使用流程

### 1. 生成路网

```bash
python sumo/network/generate.py
```

### 2. 生成路由

```bash
python sumo/routes/generate.py
```

生成 4 种场景：
- `balanced`: X/Y 方向流量均衡 (600/600 辆/小时)
- `imbalanced`: X 方向流量远大于 Y (900/300)
- `tidal`: 潮汐交通（前 15 分钟 X 为主，后 15 分钟 Y 为主）
- `burst`: 突发车流（在 10-12 分钟时出现 1800 辆/小时的脉冲）

### 3. 运行仿真

```bash
# 基本运行
python sumo/sumo_sim.py --scenario balanced --duration 600

# 带 GUI
python sumo/sumo_sim.py --scenario balanced --gui

# 自定义参数
python sumo/sumo_sim.py --scenario imbalanced --min-green 8 --max-green 35 --max-red 50
```

### 4. 对比评估

```bash
python sumo/compare_strategies.py --duration 600
```

输出各场景下 VA 控制 vs 固定配时的：
- 平均延误
- 总通过量
- 最大排队
- 切换次数
- 延误下降百分比

### 5. 参数优化

```bash
# 网格搜索最优参数
python sumo/optimize_params.py --duration 600
```

搜索空间：
- `min_green`: [5, 10, 15]
- `max_green`: [20, 30, 40, 50]
- `max_red`: [30, 45, 60]

结果保存到 `sumo/optimization_results.json`。

## 仿真架构

```
SUMO 仿真器
    │
    ├── 路网 + 路由 (.net.xml + .rou.xml)
    │
    ├── TraCI 接口 (step_length=1s)
    │      │
    │      ├── 每步读取检测器 → queue_x, queue_y
    │      ├── 送入 VAController.step() → x_light, y_light
    │      └── 写回 SUMO 信号灯
    │
    └── 输出：逐车轨迹、延误、排队、通过量
```

## 交通灯信号格式

SUMO 使用 12 位字符串控制 4 个入口的信号灯：

```
位置:  012 345 678 9AB
入口:  东入 西入 南入 北入
车道:  直左右 直左右 直左右 直左右
```

X 方向控制东入(012)和西入(345)，Y 方向控制南入(678)和北入(9AB)。

## 注意事项

- SUMO 仿真比流体模型慢 10-100 倍，建议先用短时间（300s）验证
- 参数优化耗时较长，建议先用少量场景测试
- 确保 `intersection.net.xml` 已生成再运行仿真
