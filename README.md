# 基于 OpenVINO 的 YOLO 检测与交通灯联动项目

## 项目概述

这是一个以 OpenVINO 推理部署为核心的实验型项目，当前包含两条主线：

- 目标检测：使用 OpenVINO + OpenCV 对图片、视频、摄像头和 RTSP 流做实时检测。
- 交通灯联动：把检测结果保存为结构化数据，再基于车辆数量做交通灯时序分析与模拟。

当前仓库的主要实现已经从单纯的 YOLO 演示，扩展为“检测 -> 数据记录 -> 交通灯分析/控制”的完整流程。

## 当前模块

- `yolov26.py`
  当前主入口。使用 `public/yolo-v26/ir_model/yolo26n.xml` 做推理，支持命令行模式切换，并会把检测结果写入 `data/`。

- `main.py`
  YOLOv3-tiny 参考实现。保留了更传统的 YOLO 后处理逻辑，适合做模型效果、速度和输出格式的对比。

- `traffic_light_console.py`
  读取 `data/` 中保存的检测结果，在控制台按时间线模拟交通灯周期，并输出配时分析。

- `traffic_light_raspberry.py`
  树莓派 GPIO 控制脚本。可在树莓派上驱动红黄绿 LED，也支持在非树莓派环境下退化为模拟模式。

## 项目结构

```text
src/
├─ main.py
├─ yolov26.py
├─ traffic_light_console.py
├─ traffic_light_raspberry.py
├─ pyproject.toml
├─ requirements.txt
├─ README.md
├─ public/
│  ├─ yolo-v26/
│  │  ├─ yolo26n.pt
│  │  ├─ yolo26n.onnx
│  │  └─ ir_model/
│  │     ├─ yolo26n.xml
│  │     └─ yolo26n.bin
│  ├─ yolo-v3-tiny-tf/
│  │  ├─ yolo-v3-tiny-tf.pb
│  │  ├─ yolo-v3-tiny-tf.json
│  │  └─ ir_model/
│  │     ├─ yolo-v3-tiny-tf.xml
│  │     └─ yolo-v3-tiny-tf.bin
│  └─ yolo-v3-tf/
├─ scripts/
│  ├─ downloader.ps1
│  └─ converter.ps1
├─ test/
│  ├─ input/
│  │  └─ video.mp4
│  └─ output/
│     └─ output_yolov26.mp4
├─ data/
│  └─ detection_YYYYMMDD_HHMMSS/
│     ├─ frames.json
│     ├─ frames.csv
│     └─ summary.json
├─ input.png
├─ input_video.mp4
├─ output.mp4
└─ output_yolov26.mp4
```

说明：

- `public/` 下保存模型源文件和 OpenVINO IR 文件。
- `test/` 用于存放测试输入和输出。
- `data/` 用于存放检测过程导出的结构化数据。

## 技术栈

- Python 3.10+
- OpenVINO
- OpenCV
- NumPy
- PowerShell
- `yt-dlp`（已出现在 `pyproject.toml` 依赖中）

## 环境与依赖

### 最小运行依赖

`requirements.txt` 当前内容：

```txt
openvino==2024.6.0
opencv-python==4.13.0.92
numpy==2.1.0
```

### `pyproject.toml` 中的额外依赖

项目的打包配置还声明了这些依赖：

- `yt-dlp>=2026.3.17`
- 开发可选依赖：`black`、`ruff`、`openvino-dev`、`onnx`、`onnxruntime`、`ultralytics`

如果你只是运行现有推理脚本，通常安装 `requirements.txt` 即可；如果需要模型转换、格式导出或开发打包，再参考 `pyproject.toml` 补齐依赖。

## 快速开始

在 `src/` 目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果你使用 `uv`，也可以按 `pyproject.toml` 安装项目依赖。

## 目标检测使用方式

### YOLOv26

`yolov26.py` 是当前建议使用的主脚本。

#### 图片检测

```powershell
python yolov26.py image input.png test/output/output_yolov26.png
```

#### 视频检测

推荐显式传入存在的输入路径和带目录的输出路径：

```powershell
python yolov26.py video test/input/video.mp4 test/output/output_yolov26.mp4
```

如果你要用顶层样例视频，也可以：

```powershell
python yolov26.py video input_video.mp4 test/output/output_yolov26.mp4
```

#### 摄像头检测

```powershell
python yolov26.py camera 0 test/output/output_camera_yolov26.mp4
```

#### RTSP 检测

```powershell
python yolov26.py rtsp rtsp://your_stream_url test/output/output_rtsp_yolov26.mp4
```

### YOLOv3-tiny

`main.py` 当前不是完整 CLI，而是通过文件底部硬编码入口运行。默认写法偏演示性质，必要时需要直接修改文件末尾调用参数。

直接运行：

```powershell
python main.py
```

如果默认样例路径不存在，请手动修改 [main.py](C:/Users/Zeng/Desktop/人工智能实践创新/大作业/src/main.py:425) 中的输入输出路径。

## 检测输出数据

`main.py` 和 `yolov26.py` 当前都会在推理结束后把检测结果写入 `data/`，每次运行生成一个独立会话目录：

```text
data/
└─ detection_20260519_134500/
   ├─ frames.json
   ├─ frames.csv
   └─ summary.json
```

各文件用途：

- `frames.json`：逐帧保存检测框、类别、置信度和坐标。
- `frames.csv`：便于表格分析和后处理。
- `summary.json`：保存视频信息、平均 FPS、总检测数、类别统计和阈值配置。

这部分数据是后续交通灯分析脚本的输入。

## 交通灯分析与控制

### 控制台模拟

`traffic_light_console.py` 会读取最近一次或指定会话的检测数据，按画面左右区域估算 X 路、Y 路车流，并生成交通灯配时建议。

列出已有检测会话：

```powershell
python traffic_light_console.py --list
```

只做分析，不按真实时间模拟：

```powershell
python traffic_light_console.py --analyze
```

按时间线模拟交通灯切换：

```powershell
python traffic_light_console.py --simulate --speed 10
```

分析指定会话：

```powershell
python traffic_light_console.py --analyze --session detection_20260519_134500
```

### 树莓派控制

`traffic_light_raspberry.py` 用于 GPIO 控制双色路口信号灯。

测试灯光：

```powershell
python traffic_light_raspberry.py --test
```

按给定车流启动控制：

```powershell
python traffic_light_raspberry.py --x 5 --y 3
```

在非树莓派环境下，如果没有 `RPi.GPIO`，脚本会自动切换到模拟模式。

## 模型与脚本说明

### YOLOv26

- 模型路径：`public/yolo-v26/ir_model/yolo26n.xml`
- 默认输入尺寸：`640 x 640`
- 默认设备：`CPU`
- 支持通过修改 `DEVICE` 切换到 `GPU`、`NPU` 等 OpenVINO 支持设备
- 使用跳帧检测参数 `SKIP_FRAMES=2` 来提高视频场景下的实时性

### YOLOv3-tiny

- 模型路径：`public/yolo-v3-tiny-tf/ir_model/yolo-v3-tiny-tf.xml`
- 默认输入尺寸：`416 x 416`
- 使用 anchors + strides 做手动解码

### 辅助 PowerShell 脚本

下载 Open Model Zoo 模型：

```powershell
.\scripts\downloader.ps1 yolo-v3-tiny-tf
```

转换模型到 OpenVINO IR：

```powershell
.\scripts\converter.ps1 yolo-v3-tiny-tf
```

说明：

- `converter.ps1` 当前按 TensorFlow `.pb` 模型流程编写。
- `yolo-v26` 目录已经包含 `.pt`、`.onnx` 和 IR 文件。

## 当前实现状态

从代码检查来看，当前项目已经具备这些能力：

- OpenVINO 模型推理
- 图片/视频/摄像头/RTSP 检测
- 检测结果可视化与视频保存
- 逐帧检测数据导出
- 基于检测数据的交通灯控制台分析
- 面向树莓派 GPIO 的交通灯控制脚本

但它仍然是脚本型项目，工程化程度有限，还没有抽成清晰的 Python 包结构。

## 已确认的问题与注意事项

以下问题已经在当前代码中确认，使用前应知晓：

- `yolov26.py` 的默认视频路径是 `test/input/input_video.mp4`，但当前仓库内实际样例文件是 `test/input/video.mp4`，默认值与仓库内容不一致。
- `main.py` 末尾默认调用的路径也是 `test/input/input_video.mp4`，同样可能找不到文件。
- `detect_video()` 和 `detect_image()` 中使用了 `os.makedirs(os.path.dirname(output_path), exist_ok=True)`。如果输出参数只有文件名而不带目录，例如 `output.mp4`，`os.path.dirname()` 会得到空字符串，可能报错。
- `yolov26.py` 中仍保留部分调试和实验痕迹，例如未使用的辅助函数、较强的输出 shape 假设，以及部分路径默认值不统一。
- `main.py` 仍然偏手工演示脚本，没有统一 CLI。
- 当前没有自动化测试，也没有针对模型输出格式做稳健校验。

## 建议的使用方式

为避免路径相关报错，推荐遵循两个约定：

- 输入路径始终显式传入，不依赖脚本默认值。
- 输出路径始终写成带目录的形式，例如 `test/output/output_yolov26.mp4`。

推荐命令：

```powershell
python yolov26.py video test/input/video.mp4 test/output/output_yolov26.mp4
python yolov26.py image input.png test/output/output_yolov26.png
python traffic_light_console.py --analyze
```

## 后续改进建议

- 用 `argparse` 或 `typer` 统一 `main.py` 和 `yolov26.py` 的 CLI。
- 抽出公共模块，复用模型加载、预处理、推理和数据保存逻辑。
- 统一样例输入输出目录，消除 `input_video.mp4`、`video.mp4`、`test/input/...` 混用问题。
- 在创建输出目录前判断目录名是否为空，修复纯文件名输出路径的错误。
- 给 `traffic_light_console.py` 和 `traffic_light_raspberry.py` 建立更明确的数据接口。
- 增加日志、测试和异常处理。
- 把当前的规则式交通灯策略升级为更清晰的配置化或可训练策略。

## 适合写进课程/答辩的定位

如果这个仓库用于课程设计或实验答辩，可以把它描述为：

“一个基于 OpenVINO 的目标检测与交通灯联动实验系统。前端通过 YOLO 模型对视频流中的车辆和交通场景目标进行检测，后端将检测结果结构化保存，并基于车流统计进行交通灯配时分析与控制模拟，同时预留了树莓派 GPIO 实物控制接口。”
