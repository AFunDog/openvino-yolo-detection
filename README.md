# YOLO 智能交通灯控制系统

基于 YOLOv26 目标检测的车辆监控与交通灯仿真系统。当前检测侧采用基于 `track_id` 的轨迹方向过滤，先跟踪车辆轨迹，再剔除与车流主方向偏差过大的轨迹，支持上传同一路口 `X / Y` 两个垂直方向的监控视频，并以此作为 `X/Y` 方向车辆数量判断依据。

## 功能

- **目标检测**：支持视频、摄像头、RTSP 流的 YOLOv26 实时检测（ONNX Runtime / OpenVINO 双后端）
- **轨迹方向过滤**：基于目标追踪与主方向估计，对车辆执行 `track_id` 级别方向过滤，统计有效车辆 / 过滤车辆 / 总车辆数
- **双方向视频上传**：GUI 支持分别上传同一路口 `X / Y` 两个垂直方向的视频，并自动聚合成一组双方向车辆统计
- **数据记录**：逐帧保存检测框、类别、置信度、track_id、轨迹过滤事件 到 `data/` 目录（JSON + CSV）
- **自动方向标定**：从车辆 track 位移向量自动发现两条主方向（角度直方图 + 峰值聚类），任意摄像机角度自适应
- **排队/通行分类**：基于 track 帧间中心点 EMA 速度判停排队车辆，聚合 X/Y 路排队数、等待时间、清空间隔、到达率
- **Vehicle-Actuated 控制**：清空检测 + 最大/最小绿灯约束 + 等待加权，动态切换红绿灯
- **实时联动仿真**：检测视频的同时，交通灯仿真自动跟随检测结果运行（墙钟同步，无需等待检测结束）
- **桌面 GUI**：PyQt6 可视化界面，十字路口动画、视频预览、实时状态监控、类别统计
- **Windows 明暗主题自适应**：颜色随系统主题自动切换（QPalette + 自定义调色板）
- **控制台模拟**：终端按时间线回放交通灯周期
- **树莓派控制**：GPIO 驱动实体 LED 交通灯

## 快速开始

### 安装依赖

```bash
uv sync
```

项目仅保留 `pyproject.toml + uv.lock` 作为依赖来源，不再维护 `requirements.txt`。

### 运行 GUI

```bash
python gui_app.py
```

#### 离线回放
1. 先在「YOLO 视频分析」页检测一段视频，生成检测数据
2. 切换到「交通灯仿真」页，数据源选择检测记录，点击「▶ 开始」

#### 实时联动
1. 「交通灯仿真」页 → 数据源选 **"实时检测"**
2. 切换到「YOLO 视频分析」页 → 选视频 → 点「开始检测」
3. 仿真自动启动，交通灯随检测结果实时切换

### 命令行检测

```bash
# 视频检测
python main.py video test/input/traffic.mp4 test/output/output.mp4

# 摄像头
python main.py camera 0 test/output/output.mp4
```

### 理论通行效率评估

```bash
python scripts/evaluate_signal_efficiency.py data/detection_pair_xxx/summary.json
```

可选参数示例：

```bash
python scripts/evaluate_signal_efficiency.py data/detection_pair_xxx/summary.json --fixed-green-x 20 --fixed-green-y 20 --sat-x 1.0 --sat-y 1.0 --json
```

说明：
- 输入必须是 `direction_pair` 类型的 `summary.json`
- 评估基于总车流量和视频时长推导出的均匀到达率，属于理论估算，不是逐车精确仿真
- 输出包含固定配时与自适应配时的通过量、累计延误、平均延误、最大排队和提升率

## 项目结构

```text
├── algorithm/
│   ├── __init__.py              # 算法模块入口
│   ├── data_extractor.py        # 方向标定、排队分类、特征提取
│   └── va_controller.py         # Vehicle-Actuated 控制器
├── gui_app.py                   # PyQt6 桌面应用
├── theme_manager.py             # 主题管理器（明暗调色板 + QPalette）
├── main.py                      # YOLOv26 检测（ONNX Runtime / OpenVINO）
├── traffic_light_console.py     # 控制台交通灯模拟
├── traffic_light_raspberry.py   # 树莓派 GPIO 控制
├── data/
│   ├── detection_*/             # 检测会话（自动生成）
│   │   ├── frames.json          # 逐帧检测数据
│   │   ├── frames.csv           # 表格检测数据
│   │   └── summary.json         # 统计汇总（类别计数、FPS、视频信息）
│   └── models/
├── public/
│   └── yolo-v26/                # YOLOv26 模型（ONNX + OpenVINO IR）
├── scripts/
│   ├── downloader.ps1           # 模型下载
│   └── converter.ps1            # 模型转换
└── test/
    └── output/                  # 检测输出
```

## Vehicle-Actuated 控制原理

### 数据提取管线

```
YOLO 检测帧
    │
    ├── 1. 方向自动标定（预热 150 帧后执行）
    │      收集所有长 track (>15 帧, 位移 >30px) 的总位移向量
    │      → 角度直方图 (36 bins, 0°–180°) → 平滑 → 找两个最高峰 (≥30° 间隔)
    │      → 更接近水平 (0°/180°) 的为 X 路，另一个为 Y 路
    │      → 不正交时用 X+90° 修正 Y
    │
    ├── 2. 每车方向分类
    │      track 总位移 · X方向向量 vs Y方向向量（点积取绝对值）
    │      → sim_x > sim_y 且 > 0.5 → X 路，反之 → Y 路
    │      标定前：|dx| > 2|dy| 临时归 X，|dy| > 2|dx| 临时归 Y
    │      标定后：_reclassify_all_tracks() 重新分类所有已有 track
    │
    ├── 3. 排队判定
    │      同 track 帧间中心点位移 → EMA 平滑速度 (α=0.3)
    │      速度 < 2 px/帧 连续 2+ 帧 → queued = True
    │      每帧排队 +1/fps 秒累计等待时间
    │
    ├── 4. 特征聚合（每帧）
    │      queue_x  = Σ 排队_X路车辆      wait_x  = Σ 排队_X路等待时间
    │      gap_x    = 连续无排队 X 秒数    arrival_x = 新 track 到达率 (EMA α=0.1)
    │      queue_y  = Σ 排队_Y路车辆      wait_y  = Σ 排队_Y路等待时间
    │      gap_y    = 连续无排队 Y 秒数    arrival_y = 新 track 到达率
    │
    └── 5. 控制器决策
          当前路绿灯 < 10s   → 保持（最小绿灯约束）
          当前路绿灯目标 = 10s + (30s-10s) × 当前路车辆数 / (X路车辆数 + Y路车辆数)
          当前路绿灯 ≥ 目标绿灯 → 切换
```

### 控制器参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| min_green | 10s | 最短绿灯 |
| max_green | 30s | 最长绿灯 |
| max_red | 45s | 最长红灯 |
| yellow_duration | 3s | 黄灯过渡 |
| gap_seconds | 3s | 保留为状态字段，当前比较法不再依赖 |
| wait_ratio | 1.2 | 保留为兼容参数，当前比较法不再依赖 |

## 实时联动架构

```
YOLO 检测线程                        GUI 主线程
─────────────                       ──────────
detect_video() 循环                 _sim_tick() 每 33ms
  │                                  │
  ├── process_frame()                ├── drain _live_frames (deque, 线程安全)
  ├── tracker.update()               ├── feature_extractor.process_frame() → 排队/方向
  ├── 构建 detections 列表            ├── va_controller.step(dt) → 红绿灯决策
  └── frame_callback(                └── _update_sim_ui() → Canvas + 指示灯 + 表格
        ..., detections, video_fps)
         │                                   │
         ▼                                   │
    _live_frames.append({...})  ─────────────┘
```

关键设计：
- 跳帧帧（SKIP_FRAMES=2 的非检测帧）不传入 `detections`，避免假排队
- 仿真用墙钟 `dt` 驱动（实时固定 1x），检测结束仿真继续运行
- 视频真实 FPS 自动同步到特征提取器

## GUI 功能

### YOLO 视频分析

- 输入视频路径或浏览选择，一键启动检测
- 检测中实时显示帧画面、FPS、目标数
- 检测完成后自动播放输出视频
- 左侧查看所有检测会话，支持删除
- 统计卡片：帧数 / 检测数 / 车辆数 / FPS
- 类别分布表：各类目标数量与占比

### 交通灯仿真

- 俯视十字路口 Canvas（QPainter 绘制，X/Y 路标注）
- 交通灯实时切换（红/黄/绿 + 发光效果）
- 数据源支持：历史检测记录回放 / 实时联动
- 排队车辆动画显示、倒计时 + 进度条
- 切换记录表格：每次相位切换的时长和原因
- 离线回放支持速度调节（1x~20x），实时模式固定 1x

## 技术栈

- Python 3.10+
- OpenVINO / ONNX Runtime（YOLOv26 推理）
- OpenCV（图像处理）
- PyQt6（桌面 GUI + Windows 明暗主题）
- NumPy
- RPi.GPIO（树莓派控制，可选）

## 参考资料

### 信号交叉口饱和流率

饱和流率指绿灯期间单车道每秒能通过的最大车辆数，是通行效率评估的核心参数。

| 来源 | 单车道饱和流率 |
|------|--------------|
| FHWA 信号配时手册（《Traffic Signal Timing Manual》） | 1900 veh/h/车道 ≈ **0.53 veh/s/车道** |
| FHWA  Highway Performance Monitoring System (HPMS) | 1900 pcphpl（基本饱和流率） |
| 中国《城市道路工程设计规范》CJJ 37-2012 | 1800 veh/h/车道 ≈ **0.5 veh/s/车道** |

**工程常用近似值：0.5 辆/秒/车道**（即约 2 秒通过 1 辆小客车）。

总放行能力由车道数决定：

```
Q = n × s × g
```

其中 `n` = 放行车道数，`s` ≈ 0.5 veh/s/车道，`g` = 有效绿灯时间（秒）。

| 场景 | 建议取值 |
|------|---------|
| 保守城市路口（大车多、干扰大） | 0.4 veh/s/车道 |
| 普通直行车道 | 0.5 veh/s/车道 |
| 条件较好、车辆启动快 | 0.55 veh/s/车道 |
| 有大车、左转、行人干扰、车道窄 | 0.25～0.45 veh/s/车道 |

**参考文档：**
- FHWA *Traffic Signal Timing Manual* – Chapter 3: [ops.fhwa.dot.gov](https://ops.fhwa.dot.gov/publications/fhwahop08024/chapter3.htm)
- FHWA *HPMS Field Manual* – Appendix N: [fhwa.dot.gov](https://www.fhwa.dot.gov/ohim/hpmsmanl/appn5.cfm)

### UA-DETRAC 车辆检测与跟踪数据集

本项目的训练数据与检测模型基于 **UA-DETRAC** 数据集，全称 **University at Albany DEtection and TRACking**。

#### 数据集来源与机构

由 **美国纽约州立大学奥尔巴尼分校（University at Albany, SUNY）** 牵头，多机构合作完成，包括 JD Finance America、UC San Diego、中国科学院自动化研究所、中国科学院大学、韩国汉阳大学、UC Merced。

- arXiv 论文：*UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking*
- 论文作者：Longyin Wen, Dawei Du, Zhaowei Cai, Zhen Lei, Ming-Ching Chang, Honggang Qi, Jongwoo Lim, Ming-Hsuan Yang, Siwei Lyu
- 官方页面：[albany.edu](https://www.albany.edu/cnse/research/computer-vision-machine-learning-lab)

#### 数据规格

| 属性 | 数值 |
|------|------|
| 视频序列 | 100 个 |
| 总时长 | > 10 小时 |
| 总帧数 | > 14 万帧 |
| 标注车辆 | 8,250 辆 |
| 标注边界框 | ≈ 121 万个 |
| 采集设备 | Canon EOS 550D |
| 帧率 | 25 fps |
| 分辨率 | 960 × 540 |
| 采集地点 | 中国北京、天津（24 个不同地点） |
| 场景类型 | 城市快速路、交通路口、T 型路口 |

#### 标注内容

- **车辆类型（4 类）**：car、bus、van、others
- **光照条件**：cloudy、night、sunny、rainy
- **遮挡程度**：无遮挡、部分遮挡、严重遮挡

#### 与交通灯控制项目的关系

UA-DETRAC **适合**作为本项目的上游视觉数据源：用于训练车辆检测模型（YOLOv26）、评估跟踪性能、统计车流密度与排队长度。

但需注意的局限：
- 不提供红绿灯相位、车道线拓扑、放行时长等信号控制标签
- 数据采集自北京、天津，交通行为与道路结构可能存在地域差异
- 论文定位是车辆检测与多目标跟踪（MOT）基准，**不是**交通信号控制数据集

**参考论文：**
- Wen et al., *UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking*, arXiv 1511.04136: [arxiv.org](https://arxiv.org/abs/1511.04136)
