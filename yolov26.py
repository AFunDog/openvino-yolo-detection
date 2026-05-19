"""
YOLOv26 实现代码
使用 OpenVINO 进行目标检测
"""

import openvino as ov
import cv2
import numpy as np
import os
import time
import json
import csv
from datetime import datetime

# ==================== 模型参数配置 ====================
# 模型文件路径
MODEL_XML_PATH = "public/yolo-v26/ir_model/yolo26n.xml"

# 设备选择: "CPU", "GPU", "NPU" 等
# 如果有独立显卡，改为 "GPU" 可显著提升速度
DEVICE = "CPU"  # 改为 "GPU" 如果有Intel集成显卡或独显

# 输入参数
INPUT_SIZE = 640  # 从640降低到320，速度提升约4倍，精度略降
INPUT_CHANNELS = 3

# 检测参数
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7
MAX_DETECTIONS = 300

# 性能优化参数
SKIP_FRAMES = 2  # 跳帧检测：每N帧检测一次（1=每帧检测，2=每隔1帧检测）

# 显示参数
DISPLAY_WINDOW_NAME = "YOLOv26 Detection"
DISPLAY_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
DISPLAY_LABEL_SCALE = 0.5
DISPLAY_LABEL_THICKNESS = 2
DISPLAY_BOX_THICKNESS = 2
DISPLAY_BOX_COLOR = (0, 255, 0)
DISPLAY_TEXT_COLOR = (0, 0, 0)

# COCO 类别标签 (80类)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


# 数据记录目录
DATA_DIR = "data"


def save_detection_data(source, frame_data, summary):
    """保存检测数据到 data 目录

    Args:
        source: 视频源标识
        frame_data: 帧级检测数据列表
        summary: 统计汇总信息
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(DATA_DIR, f"detection_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)

    # 保存帧级 JSON 数据
    json_path = os.path.join(session_dir, "frames.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(frame_data, f, ensure_ascii=False, indent=2)

    # 保存帧级 CSV 数据
    csv_path = os.path.join(session_dir, "frames.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "timestamp", "class", "class_id", "confidence", "x1", "y1", "x2", "y2"])
        for frame in frame_data:
            for det in frame["detections"]:
                writer.writerow([
                    frame["frame"], frame["timestamp"],
                    det["class"], det["class_id"], det["confidence"],
                    det["x1"], det["y1"], det["x2"], det["y2"]
                ])

    # 保存统计汇总
    summary_path = os.path.join(session_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"数据已保存到: {session_dir}/")
    print(f"  - frames.json  (帧级详细数据)")
    print(f"  - frames.csv   (帧级表格数据)")
    print(f"  - summary.json (统计汇总)")





def convert_boxes(x):
    """将中心点坐标 (cx, cy, w, h) 转换为 (x1, y1, x2, y2)"""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1 = cx - w/2
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1 = cy - h/2
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2 = cx + w/2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2 = cy + h/2
    return y


def nms(boxes, scores, iou_threshold):
    """非极大值抑制"""
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        scores.tolist(),
        CONF_THRESHOLD,
        iou_threshold
    )
    return indices.flatten() if len(indices) > 0 else []


def sigmoid(x):
    """Sigmoid 激活函数"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def dist2bbox(distance_points, anchor_points):
    """
    将距离预测转换为边界框坐标
    distance_points: (N, 4) - (l, t, r, b)
    anchor_points: (N, 2) - (cx, cy)
    """
    lt, rb = np.split(distance_points, 2, axis=-1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    return np.concatenate([x1y1, x2y2], axis=-1)


def decode_yolov26_output(output, img_width, img_height):
    """
    解码 YOLOv26 输出

    YOLOv26 输出格式: (batch, 300, 6)
    每个候选框: [x1, y1, x2, y2, conf, class_id]
    """
    print(f"输出shape: {output.shape}")

    batch, num_boxes, num_values = output.shape
    print(f"Batch: {batch}, Boxes: {num_boxes}, Values per box: {num_values}")

    boxes = []
    confidences = []
    class_ids = []

    # 遍历所有候选框
    for i in range(num_boxes):
        pred = output[0, i]  # (6,)

        # 提取边界框坐标 (x1, y1, x2, y2)
        x1, y1, x2, y2 = pred[:4]

        # 提取置信度和类别ID
        conf = pred[4]
        class_id = int(pred[5])

        # 只保留置信度超过阈值的框
        if conf > CONF_THRESHOLD:
            # 裁剪到图像范围内
            x1 = np.clip(x1, 0, img_width)
            y1 = np.clip(y1, 0, img_height)
            x2 = np.clip(x2, 0, img_width)
            y2 = np.clip(y2, 0, img_height)

            # 确保坐标顺序正确
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            boxes.append([x1, y1, x2, y2])
            confidences.append(float(conf))
            class_ids.append(class_id)

    return boxes, confidences, class_ids


def process_frame(frame, compiled_model, outputs):
    """
    处理单帧图像进行检测
    """
    orig_h, orig_w = frame.shape[:2]

    # 预处理
    img_resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))

    # 归一化 (0-255 -> 0-1)
    img_input = img_resized.astype(np.float32) / 255.0

    # HWC -> CHW 并添加batch维度
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0)

    # 推理
    results = compiled_model([img_input])

    # 解码输出
    all_boxes = []
    all_confidences = []
    all_class_ids = []

    for output in outputs:
        out_data = results[output]
        boxes, confidences, class_ids = decode_yolov26_output(
            out_data, INPUT_SIZE, INPUT_SIZE
        )
        all_boxes.extend(boxes)
        all_confidences.extend(confidences)
        all_class_ids.extend(class_ids)

    # NMS 非极大值抑制
    if len(all_boxes) > 0:
        indices = nms(np.array(all_boxes), np.array(all_confidences), IOU_THRESHOLD)
        final_boxes = [all_boxes[i] for i in indices]
        final_confidences = [all_confidences[i] for i in indices]
        final_class_ids = [all_class_ids[i] for i in indices]

        # 缩放检测框到原图尺寸
        scale_x = orig_w / INPUT_SIZE
        scale_y = orig_h / INPUT_SIZE
        for box in final_boxes:
            box[0] *= scale_x
            box[1] *= scale_y
            box[2] *= scale_x
            box[3] *= scale_y

        return final_boxes, final_confidences, final_class_ids

    return [], [], []


def draw_detections(img, boxes, confidences, class_ids, classes):
    """在图像上绘制检测结果"""
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)

        # 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), DISPLAY_BOX_COLOR, DISPLAY_BOX_THICKNESS)

        # 绘制标签
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(
            label, DISPLAY_LABEL_FONT, DISPLAY_LABEL_SCALE, DISPLAY_LABEL_THICKNESS
        )
        cv2.rectangle(
            img, (x1, y1 - label_h - 5), (x1 + label_w, y1),
            DISPLAY_BOX_COLOR, -1
        )
        cv2.putText(
            img, label, (x1, y1 - 5),
            DISPLAY_LABEL_FONT, DISPLAY_LABEL_SCALE, DISPLAY_TEXT_COLOR,
            DISPLAY_LABEL_THICKNESS
        )

    return img


def detect_video(source=0, output_path=None):
    """
    视频检测函数

    Args:
        source: 视频源，可以是:
            - 0, 1, 2... (摄像头编号)
            - "video.mp4" (视频文件路径)
            - "rtsp://..." (RTSP流地址)
        output_path: 输出视频保存路径，None则不保存
    """
    print("正在加载 YOLOv26 模型...")
    core = ov.Core()

    # 设置输入shape为固定值
    model = core.read_model(MODEL_XML_PATH)
    model.reshape({model.input().any_name: (1, 3, INPUT_SIZE, INPUT_SIZE)})

    # 性能优化配置
    config = {}

    if DEVICE == "CPU":
        # CPU优化设置
        config = {
            "PERFORMANCE_HINT": "LATENCY",  # 低延迟模式
            "NUM_STREAMS": "1",  # 单流处理
            "AFFINITY": "CORE"  # 绑定到物理核心
        }

    compiled_model = core.compile_model(model, DEVICE, config)
    outputs = compiled_model.outputs
    print(f"模型加载完成! 设备: {DEVICE}, 输出节点数: {len(outputs)}")
    print(f"输入尺寸: {INPUT_SIZE}x{INPUT_SIZE}, 跳帧: {SKIP_FRAMES}")

    # 打开视频源
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频源: {source}")

    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频信息: {width}x{height}, {fps} FPS")

    # 视频写入器
    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    fps_list = []
    frame_data = []
    class_counts = {}
    last_boxes = []
    last_confidences = []
    last_class_ids = []
    print("开始检测，按 'q' 退出...")

    while True:
        # 记录开始时间
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # 跳帧检测：每隔SKIP_FRAMES-1帧检测一次
        if frame_count % SKIP_FRAMES == 0:
            # 检测当前帧
            boxes, confidences, class_ids = process_frame(frame, compiled_model, outputs)
            last_boxes, last_confidences, last_class_ids = boxes, confidences, class_ids
        else:
            # 使用上一帧的检测结果
            boxes, confidences, class_ids = last_boxes, last_confidences, last_class_ids

        # 记录检测数据
        detections = []
        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            det = {
                "class": COCO_CLASSES[cls_id],
                "class_id": int(cls_id),
                "confidence": round(float(conf), 4),
                "x1": round(float(box[0]), 2),
                "y1": round(float(box[1]), 2),
                "x2": round(float(box[2]), 2),
                "y2": round(float(box[3]), 2),
            }
            detections.append(det)
            class_name = COCO_CLASSES[cls_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        frame_data.append({
            "frame": frame_count,
            "timestamp": round(time.time(), 3),
            "detections": detections,
            "num_objects": len(detections),
        })

        # 绘制结果
        if boxes:
            result_frame = draw_detections(
                frame.copy(), boxes, confidences, class_ids, COCO_CLASSES
            )
        else:
            result_frame = frame

        # 计算FPS
        end_time = time.time()
        fps = 1.0 / (end_time - start_time)
        fps_list.append(fps)
        if len(fps_list) > 30:  # 保留最近30帧的FPS
            fps_list.pop(0)
        avg_fps = sum(fps_list) / len(fps_list)

        # 在画面上显示FPS和检测数量
        fps_text = f"FPS: {avg_fps:.1f}"
        count_text = f"Objects: {len(boxes)}"
        cv2.putText(result_frame, fps_text, (10, 30),
                   DISPLAY_LABEL_FONT, 1.0, (0, 0, 255), 2)
        cv2.putText(result_frame, count_text, (10, 60),
                   DISPLAY_LABEL_FONT, 1.0, (0, 0, 255), 2)

        # 显示
        cv2.imshow(DISPLAY_WINDOW_NAME, result_frame)

        # 保存
        if writer:
            writer.write(result_frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"已处理 {frame_count} 帧, FPS: {avg_fps:.1f}...")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # 输出统计信息
    final_avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    print(f"检测完成! 共处理 {frame_count} 帧")
    print(f"平均 FPS: {final_avg_fps:.1f}")

    # 保存检测数据
    summary = {
        "source": str(source),
        "total_frames": frame_count,
        "avg_fps": round(final_avg_fps, 1),
        "total_detections": sum(f["num_objects"] for f in frame_data),
        "class_counts": class_counts,
        "video_info": {"width": width, "height": height, "fps": fps},
        "model": MODEL_XML_PATH,
        "confidence_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
    }
    save_detection_data(source, frame_data, summary)


def detect_image(image_path, output_path="test/output/result_yolov26.png"):
    """
    图片检测函数

    Args:
        image_path: 输入图片路径
        output_path: 输出图片保存路径
    """
    print("正在加载 YOLOv26 模型...")
    core = ov.Core()

    # 设置输入shape为固定值
    model = core.read_model(MODEL_XML_PATH)
    model.reshape({model.input().any_name: (1, 3, INPUT_SIZE, INPUT_SIZE)})

    # 性能优化配置
    config = {}

    if DEVICE == "CPU":
        # CPU优化设置
        config = {
            "PERFORMANCE_HINT": "LATENCY",  # 低延迟模式
            "NUM_STREAMS": "1",  # 单流处理
            "AFFINITY": "CORE"  # 绑定到物理核心
        }

    compiled_model = core.compile_model(model, DEVICE, config)
    outputs = compiled_model.outputs
    print(f"模型加载完成! 设备: {DEVICE}, 输出节点数: {len(outputs)}")
    print(f"输入尺寸: {INPUT_SIZE}x{INPUT_SIZE}, 跳帧: {SKIP_FRAMES}")

    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    print(f"正在检测图片: {image_path}")

    # 检测
    boxes, confidences, class_ids = process_frame(img, compiled_model, outputs)

    # 绘制结果
    if boxes:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_img = draw_detections(
            img.copy(), boxes, confidences, class_ids, COCO_CLASSES
        )
        cv2.imwrite(output_path, result_img)
        cv2.imshow("YOLOv26 Detection", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(f"检测到 {len(boxes)} 个目标:")
        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            print(f"  {i+1}. {COCO_CLASSES[cls_id]}: {conf:.2f}, 位置: {box}")
        print(f"结果已保存到: {output_path}")

        # 保存检测数据
        detections = []
        class_counts = {}
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            detections.append({
                "class": COCO_CLASSES[cls_id],
                "class_id": int(cls_id),
                "confidence": round(float(conf), 4),
                "x1": round(float(box[0]), 2),
                "y1": round(float(box[1]), 2),
                "x2": round(float(box[2]), 2),
                "y2": round(float(box[3]), 2),
            })
            class_name = COCO_CLASSES[cls_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        frame_data = [{
            "frame": 0,
            "timestamp": round(time.time(), 3),
            "detections": detections,
            "num_objects": len(detections),
        }]
        summary = {
            "source": image_path,
            "type": "image",
            "total_detections": len(detections),
            "class_counts": class_counts,
            "model": MODEL_XML_PATH,
            "confidence_threshold": CONF_THRESHOLD,
        }
        save_detection_data(image_path, frame_data, summary)
    else:
        print("未检测到目标")


def run_cli():
    """命令行入口函数"""
    import sys

    # 支持命令行参数
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "video"  # 默认视频模式

    # ===== 选择检测模式 =====

    if mode == "image":
        # 模式1: 检测图片
        image_path = sys.argv[2] if len(sys.argv) > 2 else "test/input/input.png"
        output_path = sys.argv[3] if len(sys.argv) > 3 else "test/output/output_yolov26.png"
        detect_image(image_path, output_path)

    elif mode == "camera":
        # 模式2: 检测摄像头
        camera_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        output_path = sys.argv[3] if len(sys.argv) > 3 else "test/output/output_camera_yolov26.mp4"
        detect_video(camera_id, output_path)

    elif mode == "video":
        # 模式3: 检测视频文件 (默认)
        video_path = sys.argv[2] if len(sys.argv) > 2 else "test/input/input_video.mp4"
        output_path = sys.argv[3] if len(sys.argv) > 3 else "test/output/output_yolov26.mp4"
        detect_video(video_path, output_path)

    elif mode == "rtsp":
        # 模式4: RTSP流
        rtsp_url = sys.argv[2] if len(sys.argv) > 2 else "rtsp://your_stream_url"
        output_path = sys.argv[3] if len(sys.argv) > 3 else "test/output/output_rtsp_yolov26.mp4"
        detect_video(rtsp_url, output_path)

    else:
        print("使用方法:")
        print("  yolo-v26 image [图片路径] [输出路径]")
        print("  yolo-v26 camera [摄像头ID] [输出路径]")
        print("  yolo-v26 video [视频路径] [输出路径]")
        print("  yolo-v26 rtsp [RTSP地址] [输出路径]")
        print("\n示例:")
        print("  yolo-v26 image test.png result.png")
        print("  yolo-v26 camera 0 output.mp4")
        print("  yolo-v26 video test_video.mp4 output.mp4")


if __name__ == "__main__":
    run_cli()
