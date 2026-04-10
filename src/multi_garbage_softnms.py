"""
Multi-Garbage Detection with Soft-NMS and Priority Sorting
多垃圾处理系统 - 含 Soft-NMS 后处理与优先级排序

本程序实现多垃圾场景的检测与处理，主要特点：
    1. Soft-NMS: 优化传统 NMS，降低漏检率
    2. 优先级排序: 结合垃圾面积和类别优先级选择处理目标
    3. 串口通信: 与下位机交互，支持 start 指令触发

Soft-NMS 原理：
    传统 NMS 直接删除重叠框，Soft-NMS 通过高斯加权降低重叠框的置信度，
    给它们重新竞争的机会，特别适合遮挡场景。

Author: yucheng
Date: 2025
Platform: MaixPy (Sipeed SG2002)
"""

from maix import camera, display, image, nn, app, uart, time
import math
import numpy as np

# ============================================================================
# 全局配置参数
# ============================================================================

# 串口配置
UART_DEVICE = "/dev/ttyS0"
UART_BAUDRATE = 115200

# 检测参数
CONF_THRESHOLD = 0.7
IOU_THRESHOLD = 0.55
STABLE_ROUNDS = 10

# ROI 裁剪区域
ROI_X, ROI_Y = 200, 130
ROI_W, ROI_H = 640, 640

# Soft-NMS 参数
SOFT_NMS_SIGMA = 0.5          # 高斯加权参数
SOFT_NMS_THRESHOLD = 0.3      # IoU 阈值
SOFT_NMS_CONF_THRESH = 0.001  # 最低置信度保留阈值

# 类别优先级配置（数值越大优先级越高）
# 优先级计算: 面积分(50%) + 类别分(50%)
LABEL_PRIORITY = {
    "kehuishou": 4,           # 可回收物 - 最高优先级
    "youhai": 3,              # 有害垃圾
    "chuyu": 2,               # 厨余垃圾
    "qita": 1                 # 其他垃圾
}

# 类别到发送数据映射
LABEL_SEND_MAPPING = {
    "kehuishou": 0,
    "youhai": 1,
    "chuyu": 2,
    "qita": 3
}

# ============================================================================
# 初始化
# ============================================================================

detector = nn.YOLO11(
    model="/root/myproject/my_mud.mud",
    dual_buff=False
)
cam = camera.Camera(900, 900, detector.input_format())
disp = display.Display()
serial = uart.UART(UART_DEVICE, UART_BAUDRATE)

# 状态变量
detect_enabled = False
stable_detection = False
last_detected_img = None
current_conf = CONF_THRESHOLD
iou_threshold = IOU_THRESHOLD
remaining_rounds = STABLE_ROUNDS


# ============================================================================
# 回调函数
# ============================================================================

def on_received(serial: uart.UART, data: bytes):
    """
    串口接收回调 - 解析控制指令
    
    支持的指令：
        - "start": 开始检测流程
    """
    global detect_enabled, stable_detection
    cmd = data.decode().strip().lower()
    
    if cmd == "start" and not detect_enabled:
        detect_enabled = True
        stable_detection = True
        print("[UART] Start command received. Beginning detection...")


def time_delay_function(rounds: int = 3):
    """
    延时稳定检测 - 等待图像稳定
    
    Args:
        rounds: 延时轮数
    """
    for i in range(rounds):
        try:
            new_img = cam.read()
            img_show = new_img.crop(ROI_X, ROI_Y, ROI_W, ROI_H)
            
            # 中间检测（仅显示）
            middle_objs = detector.detect(img_show, conf_th=0.6, iou_th=0.55)
            for obj in middle_objs:
                points = obj.get_obb_points()
                detector.draw_pose(
                    img_show, points,
                    8 if detector.input_width() > 480 else 4,
                    image.COLOR_BLACK,
                    close=True
                )
            
            disp.show(img_show)
            time.sleep_ms(20)
            print(f"[STABILIZE] Round {i + 1}/{rounds}")
        except Exception as e:
            print(f"[ERROR] time_delay_function: {e}")


# ============================================================================
# 后处理算法
# ============================================================================

def merge_overlapping_objects(objects: list, distance_threshold: float = 80) -> list:
    """
    基于距离的重叠物体过滤
    
    当两个同类物体中心距离小于阈值时，保留置信度高的。
    
    Args:
        objects: 检测到的物体列表
        distance_threshold: 中心点距离阈值（像素）
    
    Returns:
        list: 过滤后的物体列表
    """
    if not objects:
        return []
    
    # 按类别分组
    groups = {}
    for obj in objects:
        class_id = obj.class_id
        if class_id not in groups:
            groups[class_id] = []
        groups[class_id].append(obj)
    
    filtered_objects = []
    for class_id, objs in groups.items():
        # 按置信度排序（高的在前）
        sorted_objs = sorted(objs, key=lambda x: -x.score)
        kept = []
        
        for obj in sorted_objs:
            overlap = False
            for kept_obj in kept:
                dx = obj.x - kept_obj.x
                dy = obj.y - kept_obj.y
                distance = math.sqrt(dx ** 2 + dy ** 2)
                if distance < distance_threshold:
                    overlap = True
                    break
            if not overlap:
                kept.append(obj)
        
        filtered_objects.extend(kept)
    
    return filtered_objects


def calculate_real_angle(raw_angle: float, width: float, height: float) -> float:
    """
    计算实际旋转角度（0° ~ 180°）
    
    Args:
        raw_angle: 原始角度（度）
        width: 边界框宽度
        height: 边界框高度
    
    Returns:
        float: 实际角度
    """
    if (raw_angle < 10 and width > height) or (raw_angle > 80 and width < height):
        return 0.0
    elif (raw_angle < 10 and width < height) or (raw_angle > 80 and width > height):
        return 90.0
    elif raw_angle > 10 and width > height:
        return raw_angle
    else:
        return 90 + raw_angle


def get_label_by_id(class_id: int) -> str:
    """
    根据类别 ID 获取标签名称
    
    Args:
        class_id: 类别 ID (0-3)
    
    Returns:
        str: 标签名称
    """
    labels = list(LABEL_PRIORITY.keys())
    return labels[class_id] if 0 <= class_id < len(labels) else "unknown"


def calculate_priority(obj, max_area: float) -> float:
    """
    计算目标优先级分数
    
    优先级 = 面积分 * 0.5 + 类别分 * 0.5
    
    Args:
        obj: 检测到的物体对象
        max_area: 当前帧中最大物体的面积（用于归一化）
    
    Returns:
        float: 优先级分数 (0.0 ~ 1.0)
    """
    # 面积分：当前物体面积 / 最大面积
    area_score = (obj.w * obj.h / max_area) * 0.5 if max_area > 0 else 0
    
    # 类别分：根据 LABEL_PRIORITY 映射
    label = get_label_by_id(obj.class_id)
    label_score = LABEL_PRIORITY.get(label, 0) * 0.5 / 4  # 归一化到 0-0.5
    
    return area_score + label_score


# ============================================================================
# Soft-NMS 算法实现
# ============================================================================

def compute_iou(boxes: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    计算一组检测框与单个检测框的 IoU
    
    Args:
        boxes: 检测框数组，形状 (N, 4)，格式 [x1, y1, x2, y2]
        box: 单个检测框，形状 (4,)，格式 [x1, y1, x2, y2]
    
    Returns:
        np.ndarray: IoU 数组，形状 (N,)
    """
    # 计算交集坐标
    x1 = np.maximum(boxes[:, 0], box[0])
    y1 = np.maximum(boxes[:, 1], box[1])
    x2 = np.minimum(boxes[:, 2], box[2])
    y2 = np.minimum(boxes[:, 3], box[3])
    
    # 交集面积
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    
    # 各自面积
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    
    # 并集面积
    union = area_boxes + area_box - intersection
    
    return intersection / (union + 1e-6)  # 加小值防止除零


def soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    sigma: float = 0.5,
    iou_threshold: float = 0.3,
    conf_threshold: float = 0.001,
    method: int = 2
) -> np.ndarray:
    """
    Soft-NMS (Soft Non-Maximum Suppression)
    
    与传统 NMS 不同，Soft-NMS 不会直接删除重叠框，而是降低其置信度。
    这种方法在遮挡场景下效果更好。
    
    算法步骤：
        1. 按置信度排序所有框
        2. 选择置信度最高的框
        3. 计算该框与其他所有框的 IoU
        4. 根据 IoU 和权重函数降低其他框的置信度
        5. 重复直到所有框处理完毕
    
    权重函数：
        - method=1 (线性): weight = 1 - IoU if IoU > Nt else 1
        - method=2 (高斯): weight = exp(-(IoU^2) / sigma)
    
    Args:
        boxes: 检测框数组，形状 (N, 4)，格式 [x1, y1, x2, y2]
        scores: 置信度数组，形状 (N,)
        sigma: 高斯加权参数
        iou_threshold: IoU 阈值
        conf_threshold: 最低保留置信度
        method: 权重计算方法 (1=线性, 2=高斯)
    
    Returns:
        np.ndarray: 保留的索引数组
    """
    N = boxes.shape[0]
    if N == 0:
        return np.array([], dtype=int)
    
    # 添加索引列以便追踪原始位置
    indexes = np.arange(N).reshape(-1, 1)
    boxes_with_idx = np.hstack((boxes, indexes))
    
    for i in range(N):
        # 获取当前最高置信度框
        max_score = scores[i].item()
        max_pos = i
        
        # 在当前区域内寻找置信度最高的框
        pos = i + 1
        while pos < N:
            current_score = scores[pos].item()
            if max_score < current_score:
                max_score = current_score
                max_pos = pos
            pos += 1
        
        # 保存当前框坐标
        tx1, ty1, tx2, ty2 = boxes_with_idx[i, :4]
        
        # 交换到当前位置
        boxes_with_idx[i, :4], boxes_with_idx[max_pos, :4] = \
            boxes_with_idx[max_pos, :4].copy(), boxes_with_idx[i, :4].copy()
        scores[i], scores[max_pos] = scores[max_pos], scores[i]
        
        # 恢复坐标（因为后续需要原始坐标计算 IoU）
        boxes_with_idx[i, :4] = [tx1, ty1, tx2, ty2]
        scores[i] = max_score
        
        # 处理后续框
        pos = i + 1
        while pos < N:
            box = boxes_with_idx[pos, :4]
            overlap = compute_iou(np.array([boxes_with_idx[i, :4]]), box)[0]
            
            # 计算权重
            if method == 1:  # 线性
                weight = 1 - overlap if overlap > iou_threshold else 1
            elif method == 2:  # 高斯
                weight = np.exp(-(overlap ** 2) / sigma)
            else:
                raise ValueError(f"Invalid NMS method: {method}")
            
            # 降低置信度
            scores_pos = scores[pos].item()
            new_score = scores_pos * weight
            scores[pos] = new_score
            
            # 删除低置信度框
            if new_score < conf_threshold:
                boxes_with_idx[pos] = boxes_with_idx[-1]
                scores[pos] = scores[-1]
                boxes_with_idx = boxes_with_idx[:-1]
                scores = scores[:-1]
                N -= 1
                pos -= 1
            pos += 1
    
    # 返回保留的原始索引
    return boxes_with_idx[:, 4].astype(int)


# ============================================================================
# 主检测流程
# ============================================================================

def process_detection(img) -> None:
    """
    多垃圾处理流程
    
    处理流程：
        1. 稳定检测阶段
        2. 延时等待
        3. 最终检测
        4. Soft-NMS 后处理
        5. 优先级排序
        6. 发送最优目标
    """
    global stable_detection, detect_enabled, remaining_rounds
    global iou_threshold, current_conf
    
    use_soft_nms = True  # 启用 Soft-NMS
    
    # ------------------------------------------------------------------------
    # 阶段 1: 稳定检测
    # ------------------------------------------------------------------------
    if detect_enabled and stable_detection:
        try:
            img_roi = img.crop(ROI_X, ROI_Y, ROI_W, ROI_H)
            objs = detector.detect(img_roi, conf_th=0.6, iou_th=0.55)
            
            if objs:
                stable_detection = False
                disp.show(img_roi)
                time.sleep_ms(20)
                remaining_rounds = 5
            else:
                disp.show(img_roi)
        except Exception as e:
            print(f"[ERROR] Stable detection: {e}")
    
    # ------------------------------------------------------------------------
    # 阶段 2: 确认检测与后处理
    # ------------------------------------------------------------------------
    if detect_enabled and not stable_detection:
        try:
            time_delay_function(remaining_rounds)
            
            final_img_raw = cam.read()
            final_img = final_img_raw.crop(ROI_X, ROI_Y, ROI_W, ROI_H)
            
            # 检测
            final_objs = detector.detect(final_img, conf_th=current_conf, iou_th=iou_threshold)
            
            # Soft-NMS 后处理
            if use_soft_nms and final_objs:
                # 提取框坐标和置信度
                boxes = np.array([
                    [obj.x, obj.y, obj.x + obj.w, obj.y + obj.h]
                    for obj in final_objs
                ])
                scores = np.array([obj.score for obj in final_objs])
                
                # 执行 Soft-NMS
                keep = soft_nms(boxes, scores, Nt=iou_threshold)
                final_objs = [final_objs[i] for i in keep]
            
            if final_objs:
                # 计算最大面积用于优先级归一化
                max_area = max([obj.w * obj.h for obj in final_objs])
                
                # 计算每个目标的优先级
                priorities = [calculate_priority(obj, max_area) for obj in final_objs]
                
                # 选择优先级最高的目标
                selected_index = priorities.index(max(priorities))
                selected_obj = final_objs[selected_index]
                
                # 可视化所有检测到的目标
                for obj in final_objs:
                    points = obj.get_obb_points()
                    center_x = sum(points[::2]) / 4
                    center_y = sum(points[1::2]) / 4
                    
                    # 根据是否为选中目标使用不同颜色
                    if obj == selected_obj:
                        cross_color = image.COLOR_RED
                        pose_color = image.COLOR_RED
                    else:
                        cross_color = image.COLOR_BLACK
                        pose_color = image.COLOR_BLACK
                    
                    final_img.draw_cross(
                        int(center_x), int(center_y),
                        cross_color, size=25, thickness=3
                    )
                    detector.draw_pose(final_img, points, 10, pose_color, close=True)
                
                # 准备发送数据
                points = selected_obj.get_obb_points()
                center_x = sum(points[::2]) / 4
                center_y = sum(points[1::2]) / 4
                
                real_angle = calculate_real_angle(
                    selected_obj.angle * 180,
                    selected_obj.w,
                    selected_obj.h
                )
                
                label = get_label_by_id(selected_obj.class_id)
                class_id = LABEL_SEND_MAPPING.get(label, 0)
                
                # 坐标补偿
                center_x += 20
                center_y -= 5
                
                # 生成标志位
                flag = 0 if len(final_objs) != 1 else (0 if selected_obj.class_id == 2 else 1)
                
                # 发送数据
                data_str = (
                    f"{center_x:.0f},{center_y:.0f},"
                    f"{selected_obj.w:.0f},{selected_obj.h:.0f},"
                    f"{real_angle:.0f},"
                    f"{class_id},"
                    f"1,"
                    f"{flag}\n"
                )
                send_data(data_str)
                disp.show(final_img)
                
                # 重置状态
                detect_enabled = False
                stable_detection = True
                print("[INFO] Detection result sent successfully (Soft-NMS applied)")
                current_conf = CONF_THRESHOLD
            else:
                stable_detection = True
                current_conf -= 0.05
                print(f"[WARN] No target detected, retrying with conf={current_conf:.2f}...")
        except Exception as e:
            print(f"[ERROR] Final detection: {e}")


def send_data(data_str: str) -> None:
    """发送数据到串口"""
    try:
        serial.write(data_str.encode())
        print(f"[UART] Sent: {data_str.strip()}")
    except Exception as e:
        print(f"[ERROR] Failed to send data: {e}")


# ============================================================================
# 主程序入口
# ============================================================================

serial.set_received_callback(on_received)

print("[INIT] System ready")
serial.write("enabled serial ready\n".encode())

while not app.need_exit():
    try:
        img = cam.read()
        process_detection(img)
        time.sleep_ms(20)
    except Exception as e:
        print(f"[ERROR] Main loop: {e}")

print("[EXIT] System Exit")
