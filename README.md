# YOLO 智能交通灯控制系统

基于 OpenVINO 的目标检测与交通灯智能联动系统。通过 YOLO 模型检测视频中的车辆，自动分析车流并生成交通灯配时方案，提供桌面 GUI 可视化与控制台模拟。

## 功能

- **目标检测**：支持图片、视频、摄像头、RTSP 流的实时检测（YOLOv26 / YOLOv3-tiny）
- **数据记录**：逐帧保存检测框、类别、置信度到 JSON/CSV
- **交通灯算法**：根据画面左右区域车流量自动计算红绿灯时长
- **桌面 GUI**：Dear PyGui 可视化界面，含十字路口动画与数据统计
- **控制台模拟**：终端按时间线回放交通灯周期
- **树莓派控制**：GPIO 驱动实体 LED 交通灯

## 快速开始

### 安装依赖

```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install -r requirements.txt
pip install dearpygui
```

### 运行 GUI 应用

```bash
python gui_app.py
```

### 运行检测

```bash
# YOLOv26 视频检测
python yolov26.py video test/input/video.mp4 test/output/output.mp4

# YOLOv26 图片检测
python yolov26.py image input.png test/output/output.png

# YOLOv26 摄像头检测
python yolov26.py camera 0 test/output/output_camera.mp4

# YOLOv26 RTSP 流检测
python yolov26.py rtsp rtsp://stream_url test/output/output_rtsp.mp4
```

### 交通灯控制台

```bash
# 分析数据
python traffic_light_console.py --analyze

# 按时间线模拟
python traffic_light_console.py --simulate --speed 10

# 列出会话
python traffic_light_console.py --list
```

### 树莓派控制

```bash
# 测试灯光
python traffic_light_raspberry.py --test

# 按车流控制
python traffic_light_raspberry.py --x 5 --y 3
```

## 项目结构

```
├── gui_app.py                   # Dear PyGui 桌面应用
├── yolov26.py                   # YOLOv26 检测脚本（主入口）
├── main.py                      # YOLOv3-tiny 检测脚本
├── traffic_light_console.py      # 控制台交通灯模拟
├── traffic_light_raspberry.py   # 树莓派 GPIO 控制
├── pyproject.toml               # 项目配置与依赖
├── requirements.txt             # 最小依赖
├── public/
│   ├── yolo-v26/               # YOLOv26 模型
│   └── yolo-v3-tiny-tf/        # YOLOv3-tiny 模型
├── scripts/
│   ├── downloader.ps1          # 模型下载脚本
│   └── converter.ps1           # 模型转换脚本
├── data/                        # 检测数据（自动生成）
│   └── detection_YYYYMMDD_HHMMSS/
│       ├── frames.json
│       ├── frames.csv
│       └── summary.json
└── test/
    ├── input/                   # 测试输入
    └── output/                  # 检测输出
```

## GUI 功能说明

### YOLO 视频分析

- 查看 `data/` 目录下所有检测会话
- 统计信息：总帧数、检测数、车辆数、FPS
- 类别分布表：各类目标数量与占比

### 十字路口模拟

- 俯视十字路口 Canvas 动画
- 交通灯实时切换（红/黄/绿 + 发光效果）
- 车辆数分区显示（X路横向 / Y路纵向）
- 倒计时 + 进度条
- 速度调节（1x ~ 20x）
- 支持从检测数据生成配时时间线

## 交通灯算法

1. 以画面中线 `x = width/2` 划分 X路（左）和 Y路（右）
2. 统计每个灯周期内所有帧的车辆数取均值
3. 车流量大的路绿灯 +20%（最多 30s），车流量小的路绿灯 -20%（最少 10s）
4. 黄灯过渡 3 秒（闪烁）

## 技术栈

- Python 3.10+
- OpenVINO（模型推理）
- OpenCV（图像处理）
- Dear PyGui（桌面 GUI）
- NumPy
- RPi.GPIO（树莓派控制，可选）

## 模型说明

### YOLOv26

- 路径：`public/yolo-v26/ir_model/yolo26n.xml`
- 输入：640×640
- 默认设备：CPU（可切换 GPU/NPU）
- 跳帧检测：`SKIP_FRAMES=2`

### YOLOv3-tiny

- 路径：`public/yolo-v3-tiny-tf/ir_model/yolo-v3-tiny-tf.xml`
- 输入：416×416
- 使用 anchors + strides 手动解码

## 打包为 EXE

```bash
pip install pyinstaller
pyinstaller --onefile --name YOLO-Traffic gui_app.py
```
