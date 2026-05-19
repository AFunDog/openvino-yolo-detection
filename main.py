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
MODEL_XML_PATH = "public/yolo-v3-tiny-tf/ir_model/yolo-v3-tiny-tf.xml"
DEVICE = "CPU"

# 输入参数
INPUT_SIZE = 416

# 检测参数
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# YOLOv3-tiny anchors (宽, 高)
ANCHORS = [
    [[10, 14], [23, 27], [37, 58]],   # 小目标 (26x26)
    [[81, 82], [135, 169], [344, 319]]  # 大目标 (13x13)
]

STRIDES = [16, 32]  # 26x26 和 13x13 对应的 stride

# 显示参数
DISPLAY_WINDOW_NAME = "YOLO Video Detection"
DISPLAY_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
DISPLAY_LABEL_SCALE = 0.5
DISPLAY_LABEL_THICKNESS = 1
DISPLAY_BOX_THICKNESS = 2
DISPLAY_BOX_COLOR = (0, 255, 0)
DISPLAY_TEXT_COLOR = (0, 0, 0)

# COCO 类别标签
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





def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def decode_yolo_output(output, anchors, stride):
    """解码 YOLO 输出"""
    batch, channels, grid_h, grid_w = output.shape
    num_anchors = len(anchors)
    num_classes = channels // num_anchors - 5

    # 重塑输出: (batch, 255, grid_h, grid_w) -> (batch, 3, 85, grid_h, grid_w)
    output = output.reshape(batch, num_anchors, 5 + num_classes, grid_h, grid_w)
    output = output.transpose(0, 1, 3, 4, 2)  # (batch, anchors, h, w, 85)

    boxes = []
    confidences = []
    class_ids = []

    for anchor_idx in range(num_anchors):
        for row in range(grid_h):
            for col in range(grid_w):
                data = output[0, anchor_idx, row, col]

                obj_conf = sigmoid(data[4])
                if obj_conf < CONF_THRESHOLD:
                    continue

                class_scores = sigmoid(data[5:])
                class_id = np.argmax(class_scores)
                class_conf = class_scores[class_id]
                confidence = obj_conf * class_conf

                if confidence < CONF_THRESHOLD:
                    continue

                # 解码边界框
                cx = (sigmoid(data[0]) + col) * stride
                cy = (sigmoid(data[1]) + row) * stride
                w = np.exp(data[2]) * anchors[anchor_idx][0]
                h = np.exp(data[3]) * anchors[anchor_idx][1]

                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2

                boxes.append([x1, y1, x2, y2])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids


def draw_detections(img, boxes, confidences, class_ids, classes):
    """在图像上绘制检测结果"""
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), DISPLAY_BOX_COLOR, DISPLAY_BOX_THICKNESS)

        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, DISPLAY_LABEL_FONT, DISPLAY_LABEL_SCALE, DISPLAY_LABEL_THICKNESS)
        cv2.rectangle(img, (x1, y1 - label_h - 5), (x1 + label_w, y1), DISPLAY_BOX_COLOR, -1)
        cv2.putText(img, label, (x1, y1 - 5), DISPLAY_LABEL_FONT, DISPLAY_LABEL_SCALE, DISPLAY_TEXT_COLOR, DISPLAY_LABEL_THICKNESS)

    return img


def process_frame(frame, compiled_model, outputs):
    """处理单帧图像进行检测"""
    orig_h, orig_w = frame.shape[:2]
    img_resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img_resized = img_resized[:, :, ::-1]  # BGR -> RGB
    img_resized = np.expand_dims(img_resized, axis=0)  # HWC -> NHWC
    img_resized = img_resized / 255.0
    img_resized = img_resized.astype(np.float32)

    # 推理
    results = compiled_model([img_resized])

    # 解码输出
    all_boxes = []
    all_confidences = []
    all_class_ids = []

    for i, output in enumerate(outputs):
        out_data = results[output]
        boxes, confidences, class_ids = decode_yolo_output(out_data, ANCHORS[i], STRIDES[i])
        all_boxes.extend(boxes)
        all_confidences.extend(confidences)
        all_class_ids.extend(class_ids)

    # NMS 非极大值抑制
    if len(all_boxes) > 0:
        indices = cv2.dnn.NMSBoxes(all_boxes, all_confidences, CONF_THRESHOLD, NMS_THRESHOLD)
        final_boxes = [all_boxes[i] for i in indices.flatten()]
        final_confidences = [all_confidences[i] for i in indices.flatten()]
        final_class_ids = [all_class_ids[i] for i in indices.flatten()]

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


def detect_video(source=0, output_path=None):
    """视频检测函数

    Args:
        source: 视频源，可以是:
            - 0, 1, 2... (摄像头编号)
            - "video.mp4" (视频文件路径)
            - "rtsp://..." (RTSP流地址)
        output_path: 输出视频保存路径，None则不保存
    """
    # 加载模型
    core = ov.Core()
    model = core.read_model(MODEL_XML_PATH)
    compiled_model = core.compile_model(model, DEVICE)
    outputs = compiled_model.outputs

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
    print("开始检测，按 'q' 退出...")

    while True:
        # 记录开始时间
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # 检测当前帧
        boxes, confidences, class_ids = process_frame(frame, compiled_model, outputs)

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
            result_frame = draw_detections(frame.copy(), boxes, confidences, class_ids, COCO_CLASSES)
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
        "nms_threshold": NMS_THRESHOLD,
    }
    save_detection_data(source, frame_data, summary)


def detect_image(image_path, output_path="test/output/result.png"):
    """图片检测函数"""
    # 加载模型
    core = ov.Core()
    model = core.read_model(MODEL_XML_PATH)
    compiled_model = core.compile_model(model, DEVICE)
    outputs = compiled_model.outputs

    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    # 检测
    boxes, confidences, class_ids = process_frame(img, compiled_model, outputs)

    # 绘制结果
    if boxes:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_img = draw_detections(img.copy(), boxes, confidences, class_ids, COCO_CLASSES)
        cv2.imwrite(output_path, result_img)
        cv2.imshow("Detection", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(f"检测到 {len(boxes)} 个目标:")
        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            print(f"  {i+1}. {COCO_CLASSES[cls_id]}: {conf:.2f}, 位置: {box}")

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


if __name__ == "__main__":
    # ===== 选择检测模式 =====

    # 模式1: 检测图片
    # detect_image("test/input/test.png", "test/output/result.png")

    # 模式2: 检测摄像头 (默认摄像头为0)
    # detect_video(0, "test/output/output_camera.mp4")

    # 模式3: 检测视频文件
    detect_video("test/input/input_video.mp4", "test/output/output.mp4")

    # 模式4: RTSP流
    # detect_video("rtsp://your_stream_url", "test/output/output_rtsp.mp4")