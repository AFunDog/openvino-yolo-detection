# 基于 OpenVINO 的 YOLO 目标检测项目

## 项目简介

这是一个基于 Python、OpenVINO 和 OpenCV 的目标检测项目，主要用于对图片、视频、摄像头画面以及 RTSP 视频流进行实时目标检测。

项目当前包含两套检测实现：

- `yolov26.py`：当前更完整的主实现，使用 `YOLOv26` 模型进行推理。
- `main.py`：基于 `YOLOv3-tiny` 的参考实现，适合对比不同模型的效果和速度。

整个项目的核心目标是：

- 使用 OpenVINO 对检测模型进行部署和推理加速
- 支持多种输入源的目标检测
- 输出带有检测框、类别标签和置信度的结果
- 为团队后续继续做模型对比、性能优化和功能扩展提供基础

## 项目功能

当前项目支持以下能力：

- 图片目标检测
- 本地视频文件目标检测
- 摄像头实时检测
- RTSP 视频流检测
- 检测结果可视化
- 检测结果保存为图片或视频文件
- 基于 OpenVINO 的 CPU 推理
- 预留 GPU / NPU 设备切换能力

## 项目结构

```text
src/
├─ main.py                         # YOLOv3-tiny 检测脚本
├─ yolov26.py                     # YOLOv26 检测脚本（当前主实现）
├─ public/                        # 模型文件目录
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
│  ├─ downloader.ps1              # Open Model Zoo 模型下载脚本
│  └─ converter.ps1               # 模型转换为 OpenVINO IR 的脚本
├─ openvino_env/                  # 本地 Python 虚拟环境（建议不要提交到 Git）
├─ test.png                       # 测试图片
├─ test_video.mp4                 # 测试视频
├─ result.png                     # 检测输出图片
├─ result_yolov26.png             # YOLOv26 输出图片
├─ output.mp4                     # YOLOv3-tiny 输出视频
└─ output_yolov26.mp4             # YOLOv26 输出视频
```

## 技术栈

- Python 3.10
- OpenVINO
- OpenCV
- NumPy
- PowerShell（用于运行辅助脚本）

## 运行环境

根据当前仓库中的本地虚拟环境，项目已验证的基础环境如下：

- 操作系统：Windows
- Python 版本：`3.10.11`
- OpenVINO 版本：`2024.6.0`
- OpenCV 版本：`4.13.0.92`
- NumPy 版本：`2.1.0`

推荐团队统一使用以下环境配置：

- Windows 10 / Windows 11
- Python `3.10.x`
- OpenVINO `2024.x`
- 使用独立虚拟环境运行项目

## 依赖项

项目运行至少需要以下 Python 依赖：

```txt
openvino==2024.6.0
opencv-python==4.13.0.92
numpy==2.1.0
```

如果需要重新导出模型、转换模型或继续训练/转换 YOLOv26，实际开发中还可能会用到：

- `openvino-dev`
- `onnx`
- `onnxruntime`
- `ultralytics`

说明：

- 仅运行现有检测脚本时，核心依赖主要是 `openvino`、`opencv-python` 和 `numpy`
- 如果只做推理，不一定需要安装完整训练工具链

## 环境搭建

### 1. 创建虚拟环境

在项目根目录执行：

```powershell
python -m venv .venv
```

### 2. 激活虚拟环境

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. 安装依赖

```powershell
pip install openvino==2024.6.0 opencv-python==4.13.0.92 numpy==2.1.0
```

如果后续需要做模型转换或相关开发，可以补充安装：

```powershell
pip install openvino-dev onnx onnxruntime ultralytics
```

## 如何运行

### 运行 YOLOv26 检测

#### 1. 检测图片

```powershell
python yolov26.py image test.png result_yolov26.png
```

#### 2. 检测视频

```powershell
python yolov26.py video test_video.mp4 output_yolov26.mp4
```

#### 3. 检测摄像头

```powershell
python yolov26.py camera 0 output_camera_yolov26.mp4
```

#### 4. 检测 RTSP 视频流

```powershell
python yolov26.py rtsp rtsp://your_stream_url output_rtsp_yolov26.mp4
```

### 运行 YOLOv3-tiny 检测

`main.py` 当前默认演示的是视频检测。直接运行：

```powershell
python main.py
```

如果需要切换到图片、摄像头或 RTSP，可以修改 `main.py` 文件底部的调用方式。

## 模型说明

### YOLOv26

- 模型路径：`public/yolo-v26/ir_model/yolo26n.xml`
- 当前脚本默认输入尺寸：`640 x 640`
- 当前默认推理设备：`CPU`
- 支持通过修改 `DEVICE` 参数切换为 `GPU` 或其他 OpenVINO 支持的设备

### YOLOv3-tiny

- 模型路径：`public/yolo-v3-tiny-tf/ir_model/yolo-v3-tiny-tf.xml`
- 当前脚本默认输入尺寸：`416 x 416`
- 适合做轻量级检测和速度对比实验

## 辅助脚本说明

### `scripts/downloader.ps1`

该脚本用于通过 Open Model Zoo 下载模型：

```powershell
.\scripts\downloader.ps1 yolo-v3-tiny-tf
```

### `scripts/converter.ps1`

该脚本用于将模型转换为 OpenVINO IR 格式：

```powershell
.\scripts\converter.ps1 yolo-v3-tiny-tf
```

注意：

- 这两个脚本依赖本机已正确安装 OpenVINO 工具链
- `converter.ps1` 当前更适合 TensorFlow `.pb` 模型的转换流程
- `YOLOv26` 目录下当前已经包含 `.onnx`、`.pt` 和转换后的 IR 文件

## 代码说明

### `yolov26.py`

主要流程包括：

- 加载 OpenVINO 模型
- 对输入图像做 resize、归一化和维度变换
- 执行推理
- 解析模型输出
- 做 NMS 非极大值抑制
- 将检测框映射回原图
- 在图像或视频帧上绘制结果

脚本内包含一些可以调整的关键参数：

- `DEVICE`：推理设备
- `INPUT_SIZE`：输入尺寸
- `CONF_THRESHOLD`：置信度阈值
- `IOU_THRESHOLD`：NMS 的 IoU 阈值
- `SKIP_FRAMES`：跳帧检测设置

### `main.py`

该脚本使用 YOLOv3-tiny 的输出格式进行手动解码，适合作为：

- 经典 YOLO 检测流程参考
- 与 YOLOv26 做推理效果对比
- OpenVINO 入门实验代码

## 团队协作建议

为了方便多人协作，建议遵循以下约定：

- 不要提交虚拟环境目录，例如 `openvino_env/`、`.venv/`
- 不要提交大体积推理输出文件，例如 `.mp4`、中间结果图片
- 尽量把依赖写入 `requirements.txt`
- 新功能开发尽量放在独立分支中完成
- 对模型文件、测试样例和输出结果做明确命名
- 提交前先确认脚本中的模型路径仍然有效

建议协作分工方向：

- 一部分同学负责模型与推理优化
- 一部分同学负责输入源扩展和结果展示
- 一部分同学负责文档、实验记录和测试

## 当前项目的已知注意事项

- `openvino_env/` 已被 `.gitignore` 忽略，但如果团队成员各自创建环境，仍建议统一使用 `.venv/`
- `public/` 目录当前没有被忽略，说明模型文件默认会保留在仓库中；如果仓库体积过大，可以考虑改为只保留下载说明
- `main.py` 和 `yolov26.py` 的默认模型、输入尺寸和输出格式不同，使用时需要注意区分
- 项目目前以脚本形式组织，还没有封装成模块化工程结构

## 后续可改进方向

- 增加 `requirements.txt`
- 增加命令行参数解析，例如 `argparse`
- 支持统一配置文件管理
- 增加日志系统
- 增加性能统计与实验报告
- 将图片检测、视频检测和模型加载封装为独立模块
- 增加更完整的错误处理

## 快速开始

如果你只是想快速跑通项目，可以按下面步骤执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install openvino==2024.6.0 opencv-python==4.13.0.92 numpy==2.1.0
python yolov26.py video test_video.mp4 output_yolov26.mp4
```

运行成功后，你应该能看到检测窗口，并在当前目录下得到输出视频文件。
