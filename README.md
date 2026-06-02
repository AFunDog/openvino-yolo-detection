# YOLO 智能交通灯控制系统

基于 YOLOv26 目标检测与 Vehicle-Actuated 感应控制的智能交通灯系统。从检测帧的 track 位移中自动标定道路方向、识别排队车辆，实时驱动红绿灯配时。

## 功能

- **目标检测**：支持图片、视频、摄像头、RTSP 流的 OpenVINO YOLOv26 实时检测
- **批量图片检测**：支持将连续图像序列转为训练用检测数据
- **数据记录**：逐帧保存检测框、类别、置信度、track_id 到 JSON/CSV
- **自动方向标定**：从车辆运动方向自动发现道路方向，无需人工标定
- **排队/通行分类**：基于 track 帧间位移判停排队与通行车辆
- **Vehicle-Actuated 控制**：根据实时排队特征动态决策绿灯切换时机
- **桌面 GUI**：PyQt6 可视化界面，十字路口动画、视频预览、实时状态监控
- **控制台模拟**：终端按时间线回放交通灯周期
- **树莓派控制**：GPIO 驱动实体 LED 交通灯

## 快速开始

### 安装依赖

```bash
uv sync
```

### 运行检测 → 生成数据

```bash
# 视频检测
python yolov26.py video test/input/traffic.mp4 test/output/output.mp4

# 图片序列检测（连续路口拍摄的图片）
python yolov26.py images test/input/intersection1 25
```

### 运行 GUI 仿真

```bash
python gui_app.py
```

进入「交通灯仿真」页面，选择检测数据源，点击「开始」启动控制。

## Vehicle-Actuated 控制原理

### 数据提取管线

```
YOLO 检测帧
    │
    ├── 1. 方向自动标定
    │      收集长 track 的位移向量 → 角度直方图 → 找两个峰值方向
    │      → X路方向向量 / Y路方向向量（任意摄像头角度自适应）
    │
    ├── 2. 每车方向分类
    │      track 位移向量 · X方向向量 vs Y方向向量 → 归入 X 或 Y
    │
    ├── 3. 排队判定
    │      同 track 帧间中心点 EMA 速度 < 2px/帧 → 排队
    │
    ├── 4. 特征聚合（每帧）
    │      queue_x = Σ 排队_X路车      wait_x = Σ 排队时间
    │      gap_x   = 连续无排队秒数     arrival_x = 新track到达率
    │
    └── 5. 控制器决策
          当前路已清空 → 切换
          当前路绿灯 ≥ 30s → 强制切换
          对方等待 > 己方 × 1.2 → 切换
```

### 决策逻辑

| 优先级 | 条件 | 动作 | 说明 |
|--------|------|------|------|
| 1 | 绿灯 < 10s | KEEP | 最小绿灯安全约束 |
| 2 | 当前路连续无排队 > 3s | SWITCH | 清空检测 |
| 3 | 绿灯 ≥ 30s | SWITCH | 最大绿灯约束 |
| 4 | 对方等待 > 己方 × 1.2 且己方无排队 | SWITCH | 等待加权 |

### 控制器参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| min_green | 10s | 最短绿灯 |
| max_green | 30s | 最长绿灯 |
| max_red | 45s | 最长红灯 |
| yellow_duration | 3s | 黄灯过渡 |
| gap_seconds | 3s | 连续无车判定为"已清空" |
| wait_ratio | 1.2 | 对方等待超过当前倍数则触发切换 |

## 项目结构

```
├── algorithm/
│   ├── __init__.py              # 算法模块入口
│   ├── data_extractor.py        # YOLO → 交通特征提取（方向标定、排队分类、等待时间）
│   └── va_controller.py         # Vehicle-Actuated 感应控制器
├── gui_app.py                   # PyQt6 桌面应用
├── yolov26.py                   # YOLOv26 检测脚本（含批量图片检测）
├── main.py                      # YOLOv3-tiny 检测脚本
├── traffic_light_console.py     # 控制台交通灯模拟
├── traffic_light_raspberry.py   # 树莓派 GPIO 控制
├── data/
│   ├── detection_*/             # 检测会话（自动生成）
│   │   ├── frames.json          # 帧级检测数据
│   │   └── summary.json         # 统计汇总
│   └── models/                  # 模型文件
├── public/
│   ├── yolo-v26/                # YOLOv26 OpenVINO IR 模型
│   └── yolo-v3-tiny-tf/         # YOLOv3-tiny 模型
├── scripts/
│   ├── downloader.ps1           # 模型下载脚本
│   └── converter.ps1            # 模型转换脚本
└── test/
    ├── input/                   # 测试输入视频/图片
    └── output/                  # 检测输出视频
```

## GUI 功能说明

### YOLO 视频分析

- 输入视频路径或浏览选择文件，一键启动 YOLOv26 检测
- 检测中实时显示帧画面、FPS、目标数
- 检测完成后自动播放输出视频（支持播放/暂停）
- 左侧查看 `data/` 目录下所有检测会话，支持删除
- 统计卡片：总帧数、检测数、车辆数、FPS
- 类别分布表：各类目标数量与占比

### 交通灯仿真

- 俯视十字路口 Canvas 动画（QPainter 绘制，支持缩放）
- 交通灯实时切换（红/黄/绿 + 发光效果）
- **Vehicle-Actuated 感应控制**：从检测数据中实时提取排队特征，驱动相位切换
- 车辆数分区显示（X路横向 / Y路纵向）
- 倒计时 + 进度条、速度调节（1x ~ 20x）
- 切换记录表格：每次相位切换的时长和原因

## 技术栈

- Python 3.10+
- OpenVINO（YOLOv26 模型推理）
- OpenCV（图像处理）
- PyQt6（桌面 GUI）
- NumPy
- RPi.GPIO（树莓派控制，可选）
