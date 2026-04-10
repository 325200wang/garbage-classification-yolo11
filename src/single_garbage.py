"""
Single Garbage Detection with UART Communication
单垃圾处理系统 - 含串口通信

本程序实现单个垃圾的检测与定位，通过 UART 串口与下位机（如 STM32/Arduino）通信，
发送垃圾的位置、角度、类别等信息，用于控制机械臂进行分拣。

主要功能：
    1. 接收 "start" 指令后开始检测
    2. 多帧稳定检测机制，避免误检
    3. 自适应置信度调整
    4. 通过串口发送检测结果

通信协议：
    接收: "start\n"  - 开始检测
    发送: "x,y,w,h,angle,class,num,flag\n" - 检测结果

Author: yucheng
Date: 2025
Platform: MaixPy (Sipeed SG2002)
"""

from maix import camera, display, image, nn, app, uart, time
import math

# ============================================================================
# 全局配置参数
# ============================================================================

# 串口配置
UART_DEVICE = "/dev/ttyS0"      # 串口设备路径
UART_BAUDRATE = 115200          # 波特率

# 检测参数
CONF_THRESHOLD = 0.7            # 初始置信度阈值
IOU_THRESHOLD = 0.55            # NMS IoU 阈值
STABLE_ROUNDS = 10              # 稳定检测帧数

# ROI 裁剪区域 (根据实际摄像头安装位置调整)
ROI_X, ROI_Y = 200, 130         # ROI 左上角坐标
ROI_W, ROI_H = 640, 640         # ROI 宽高

# 类别优先级映射（用于多目标时排序，本程序主要处理单目标）
# 数值越小优先级越高
PRIORITY_ORDER = {
    'kehuishou': 0,             # 可回收物 - 最高优先级
    'chuyu': 1,                 # 厨余垃圾
    'youhai': 2,                # 有害垃圾
    'qita': 3                   # 其他垃圾
}

# ============================================================================
# 初始化
# ============================================================================

# 初始化 YOLO11-OBB 检测器
# dual_buff=False: 单缓冲模式，适合串口通信的同步场景
detector = nn.YOLO11(
    model="/root/myproject/my_mud.mud",
    dual_buff=False
)

# 初始化摄像头
cam = camera.Camera(900, 900, detector.input_format())
disp = display.Display()

# 初始化串口通信
serial = uart.UART(UART_DEVICE, UART_BAUDRATE)

# ============================================================================
# 状态变量
# ============================================================================

detect_enabled = False          # 检测使能标志（由串口指令控制）
stable_detection = False        # 稳定检测状态标志
last_detected_img = None        # 上一次检测结果图像（调试用）

# 自适应参数
current_conf = CONF_THRESHOLD   # 当前置信度阈值（会动态调整）
current_iou = IOU_THRESHOLD     # 当前 IoU 阈值
remaining_rounds = STABLE_ROUNDS  # 剩余稳定检测轮数


# ============================================================================
# 回调函数
# ============================================================================

def on_received(serial: uart.UART, data: bytes):
    """
    串口数据接收回调函数
    
    解析下位机发送的指令：
        - "start": 开始检测流程
    
    Args:
        serial: UART 对象
        data: 接收到的字节数据
    """
    global detect_enabled, stable_detection
    
    # 解码并清理指令
    cmd = data.decode().strip().lower()
    
    if cmd == "start" and not detect_enabled:
        detect_enabled = True
        stable_detection = True
        print("[UART] Received 'start' command. Beginning stabilization detection...")


def time_delay_function(rounds: int = 3):
    """
    延时稳定检测函数
    
    在正式检测前进行多轮预热和稳定检查，确保：
        1. 摄像头自动曝光/白平衡稳定
        2. 垃圾停止晃动
        3. 获取更准确的检测框
    
    Args:
        rounds: 延时轮数，每轮约 20ms
    
    Note:
        此函数会显示中间过程，方便调试观察。
    """
    for i in range(rounds):
        # 读取新帧
        new_img = cam.read()
        img_show = new_img.crop(ROI_X, ROI_Y, ROI_W, ROI_H)
        
        # 执行中间检测（仅用于显示，不保存结果）
        middle_objs = detector.detect(img_show, conf_th=0.6, iou_th=0.55)
        
        # 可视化中间结果
        for obj in middle_objs:
            points = obj.get_obb_points()
            detector.draw_pose(
                img_show, points,
                8 if detector.input_width() > 480 else 4,
                image.COLOR_RED,
                close=True
            )
        
        disp.show(img_show)
        time.sleep_ms(20)
        print(f"[STABILIZE] Round {i + 1}/{rounds}")


def merge_overlapping_objects(objects: list, distance_threshold: float = 80) -> list:
    """
    合并重叠物体（同一类别且中心距离小于阈值时，保留面积大的）
    
    应用场景：
        - 当一个垃圾被多次检测时，保留最准确的一次
        - 根据面积大小选择（通常大框更准确）
    
    Args:
        objects: 检测到的物体列表
        distance_threshold: 中心点距离阈值（像素）
    
    Returns:
        list: 过滤后的物体列表
    """
    if not objects:
        return []
    
    # Step 1: 按类别分组
    groups = {}
    for obj in objects:
        class_id = obj.class_id
        if class_id not in groups:
            groups[class_id] = []
        groups[class_id].append(obj)
    
    # Step 2: 对每个类别组进行过滤
    filtered_objects = []
    for class_id, objs in groups.items():
        # 按面积降序排序（大的在前）
        sorted_objs = sorted(objs, key=lambda x: -(x.w * x.h))
        
        kept = []
        for obj in sorted_objs:
            overlap = False
            for kept_obj in kept:
                # 计算两个物体中心点的欧氏距离
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
    计算实际旋转角度（将模型输出映射到 0°~180°）
    
    Args:
        raw_angle: 模型输出的原始角度（度）
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


def process_detection(img) -> None:
    """
    处理单帧图像的检测流程
    
    状态机说明：
        1. 等待使能 (detect_enabled=False)
        2. 稳定检测阶段 (detect_enabled=True, stable_detection=True)
        3. 确认检测阶段 (detect_enabled=True, stable_detection=False)
    
    Args:
        img: 原始图像
    """
    global stable_detection, detect_enabled, remaining_rounds, current_iou, current_conf
    
    # ------------------------------------------------------------------------
    # 阶段 1: 稳定检测
    # ------------------------------------------------------------------------
    if detect_enabled and stable_detection:
        # 裁剪 ROI
        img_roi = img.crop(ROI_X, ROI_Y, ROI_W, ROI_H)
        
        # 初步检测（较低阈值，提高检出率）
        objs = detector.detect(img_roi, conf_th=0.6, iou_th=0.55)
        
        if len(objs) > 0:
            # 检测到物体，进入确认阶段
            stable_detection = False
            disp.show(img_roi)
            time.sleep_ms(20)
            remaining_rounds = 4  # 设置确认检测轮数
        else:
            disp.show(img_roi)
    
    # ------------------------------------------------------------------------
    # 阶段 2: 确认检测并发送结果
    # ------------------------------------------------------------------------
    if detect_enabled and not stable_detection:
        # 执行延时稳定
        time_delay_function(remaining_rounds)
        
        # 重新读取帧，确保获取最新图像
        final_img_raw = cam.read()
        final_img = final_img_raw.crop(ROI_X, ROI_Y, ROI_W, ROI_H)
        
        # 最终检测（使用当前置信度阈值）
        final_objs = detector.detect(final_img, conf_th=current_conf, iou_th=current_iou)
        
        if final_objs:
            # 选择置信度最高的目标
            selected_obj = max(final_objs, key=lambda obj: obj.score)
            
            # 可视化所有检测到的物体（用于调试）
            for obj in final_objs:
                points = obj.get_obb_points()
                center_x = sum(points[::2]) / 4
                center_y = sum(points[1::2]) / 4
                final_img.draw_cross(
                    int(center_x), int(center_y),
                    image.Color.from_rgb(0, 0, 0),
                    size=25, thickness=3
                )
                detector.draw_pose(final_img, points, 10, image.COLOR_BLACK, close=True)
            
            # 准备发送数据
            points = selected_obj.get_obb_points()
            center_x = (points[0] + points[2] + points[4] + points[6]) / 4
            center_y = (points[1] + points[3] + points[5] + points[7]) / 4
            
            # 计算实际角度
            real_angle = calculate_real_angle(
                selected_obj.angle * 180,
                selected_obj.w,
                selected_obj.h
            )
            
            class_id = selected_obj.class_id
            
            # 坐标补偿（根据实际机械臂校准结果调整）
            center_x += 20
            center_y += 5
            
            # 生成标志位
            flag = 0 if class_id == 2 else 1
            
            # 构建发送数据字符串
            # 格式: x,y,w,h,angle,class,num,flag\n
            data_str = (
                f"{center_x:.0f},{center_y:.0f},"
                f"{selected_obj.w:.0f},{selected_obj.h:.0f},"
                f"{real_angle:.0f},"
                f"{class_id},"
                f"1,"  # num=1 表示单垃圾
                f"{flag}\n"
            )
            
            send_data(data_str)
            disp.show(final_img)
            
            # 重置状态
            detect_enabled = False
            stable_detection = True
            print("[INFO] Stable detection result sent successfully")
            current_conf = CONF_THRESHOLD  # 恢复默认置信度
            
        else:
            # 未检测到稳定目标，降低阈值重试
            stable_detection = True
            current_conf -= 0.05
            print(f"[WARN] No stable target detected, retrying with conf={current_conf:.2f}...")


def send_data(data_str: str) -> None:
    """
    通过串口发送数据
    
    Args:
        data_str: 要发送的字符串
    """
    serial.write(data_str.encode())
    print(f"[UART] Sent: {data_str.strip()}")


# ============================================================================
# 主程序
# ============================================================================

# 设置串口接收回调
serial.set_received_callback(on_received)

# 发送系统就绪信号
ready_msg = "enabled serial ready\n"
print(f"[INIT] {ready_msg.strip()}")
serial.write(ready_msg.encode())

# 主循环
while not app.need_exit():
    # 读取图像
    img = cam.read()
    
    # 处理检测流程
    process_detection(img)
    
    # 帧率控制（约 50 FPS）
    time.sleep_ms(20)

print("[EXIT] System Exit")
