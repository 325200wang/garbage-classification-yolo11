"""
Multi-Garbage Detection with ByteTrack and Sequential Sorting
多垃圾处理系统 - 含 ByteTrack 目标追踪与顺序分拣

本程序实现多垃圾场景的智能分拣系统，主要特点：
    1. ByteTrack 目标追踪: 持续跟踪多个垃圾，保持ID稳定
    2. 稳定检测机制: 通过多帧分析判断垃圾是否静止
    3. 顺序分拣: 按优先级逐个处理多个垃圾
    4. 超时保护: 防止检测卡死
    5. 状态管理: 处理垃圾遗漏、重新检测等复杂场景

分拣流程：
    1. 检测所有垃圾并追踪
    2. 等待目标稳定（连续多帧位置/角度变化小）
    3. 选择优先级最高的垃圾
    4. 发送坐标，等待机械臂抓取
    5. 接收 "next" 指令，处理下一个垃圾

通信协议：
    接收: "start" - 开始新一轮检测
    接收: "next"  - 处理下一个垃圾
    发送: "x,y,w,h,angle,class,num,flag,seq\n" - 检测结果

Author:  yucheng
Date: 2025
Platform: MaixPy (Sipeed SG2002)
"""

from maix import camera, display, image, nn, app, uart, time, tracker
import math

# ============================================================================
# 全局配置参数
# ============================================================================

# 串口配置
UART_DEVICE = "/dev/ttyS0"
UART_BAUDRATE = 115200

# 超时配置（秒）
TIMEOUT_SECONDS = 17

# ROI 裁剪区域
ROI_X, ROI_Y = 200, 130
ROI_W, ROI_H = 640, 640

# ByteTrack 追踪器参数
BYTE_TRACKER_CONFIG = {
    "max_lost_buff_num": 15,    # 最大丢失帧数（遮挡容忍）
    "track_thresh": 0.4,        # 追踪置信度阈值
    "high_thresh": 0.6,         # 高分检测框阈值（用于初始化新轨迹）
    "match_thresh": 0.5,        # 匹配 IoU 阈值
    "max_history": 10           # 历史轨迹保留帧数
}

# 稳定检测参数
STABLE_CONFIG = {
    "frames": 5,                # 连续稳定帧数要求
    "iou_thresh": 0.9,          # IoU 变化阈值
    "center_shift": 20,         # 中心点位移阈值（像素）
    "angle_thresh": 10          # 角度变化阈值（度）
}

# ============================================================================
# 初始化
# ============================================================================

# 双检测器配置：
# - detector1: 用于追踪（dual_buff=True，速度快）
# - detector2: 用于最终检测（dual_buff=False，精度高）
detector1 = nn.YOLO11(model="/root/myproject/my_mud.mud", dual_buff=True)
detector2 = nn.YOLO11(model="/root/myproject/my_mud.mud", dual_buff=False)

# 初始化设备
cam = camera.Camera(900, 900, detector1.input_format())
disp = display.Display()
serial = uart.UART(UART_DEVICE, UART_BAUDRATE)

# ============================================================================
# 状态变量
# ============================================================================

start_time = 0          # 检测开始时间戳（用于超时判断）
start = False           # 主使能标志
nextt = False           # 处理下一个标志
result = 0              # 剩余待处理垃圾数量
result2 = []            # 当前检测到的垃圾列表

# ByteTrack 追踪器
object_tracker = tracker.ByteTracker(**BYTE_TRACKER_CONFIG)

# OBB 历史记录映射: {track_id: [obb_history_list]}
obb_map = {}
stable_target_id = None  # 当前稳定的追踪目标 ID

# ============================================================================
# 工具函数
# ============================================================================

def obb_to_bbox(obj) -> tuple:
    """
    将 OBB 转换为外接水平框 (Bounding Box)
    
    Args:
        obj: OBB 对象
    
    Returns:
        tuple: (x_min, y_min, width, height)
    """
    points = obj.get_obb_points()
    x_min = min(points[::2])      # 所有 x 坐标的最小值
    y_min = min(points[1::2])     # 所有 y 坐标的最小值
    x_max = max(points[::2])      # 所有 x 坐标的最大值
    y_max = max(points[1::2])     # 所有 y 坐标的最大值
    return (x_min, y_min, x_max - x_min, y_max - y_min)


def iou_rotated(obb1, obb2) -> float:
    """
    计算两个 OBB 之间的 IoU（使用外接水平框近似）
    
    Args:
        obb1, obb2: OBB 对象
    
    Returns:
        float: IoU 值 (0.0 ~ 1.0)
    """
    bbox1 = obb_to_bbox(obb1)
    bbox2 = obb_to_bbox(obb2)
    
    # 计算交集
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
    
    if x1 >= x2 or y1 >= y2:
        return 0.0
    
    inter = (x2 - x1) * (y2 - y1)
    union = bbox1[2] * bbox1[3] + bbox2[2] * bbox2[3] - inter
    
    return inter / union if union > 0 else 0.0


def send_data(data_str: str) -> None:
    """发送数据到串口"""
    serial.write(data_str.encode())


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


def choose_target(objs: list) -> object:
    """
    选择优先级最高的目标
    
    优先级规则：class_id 越小优先级越高
    （0=可回收, 1=厨余, 2=有害, 3=其他）
    
    Args:
        objs: 检测到的物体列表
    
    Returns:
        object: 选中的目标对象
    """
    if not objs:
        return None
    
    # 选择 class_id 最小的目标
    return min(objs, key=lambda obj: obj.class_id)


def get_garbage_category(obj) -> int:
    """
    根据 class_id 获取垃圾大类
    
    Args:
        obj: 检测到的物体
    
    Returns:
        int: 大类 ID (0=有害, 1=可回收, 2=厨余, 3=其他)
    """
    cid = obj.class_id
    if cid in [0, 1, 2, 3]:
        print("[CLASS] 可回收垃圾")
        return 1  # 可回收
    elif cid in [4, 5, 6, 7, 8]:
        print("[CLASS] 有害垃圾")
        return 0  # 有害
    elif cid in [9, 10, 11, 12]:
        print("[CLASS] 厨余垃圾")
        return 2  # 厨余
    else:  # 13-16
        print("[CLASS] 其他垃圾")
        return 3  # 其他


# ============================================================================
# 串口回调
# ============================================================================

def on_received(serial: uart.UART, data: bytes):
    """
    串口接收回调函数
    
    支持的指令：
        - "start": 开始新一轮检测
        - "next":  处理下一个垃圾
    """
    global start, nextt, start_time
    
    cmd = data.decode().strip().lower()
    
    if cmd == "start":
        start = True
        nextt = False
        start_time = time.ticks_ms()  # 记录开始时间
        print("[UART] Start command received. Beginning detection...")
    
    elif cmd == "next":
        start = True
        nextt = True
        start_time = time.ticks_ms()
        print("[UART] Next command received. Processing next garbage...")


# ============================================================================
# 追踪与可视化
# ============================================================================

def draw_tracking(img, objs: list, tracks: list, obb_map: dict) -> object:
    """
    绘制追踪效果，包括：
        - 所有检测到的物体
        - 目标运动轨迹
        - 稳定目标高亮
    
    Args:
        img: 目标图像
        objs: 检测到的物体列表
        tracks: 追踪轨迹列表
        obb_map: OBB 历史记录映射
    
    Returns:
        object: 绘制后的图像
    """
    # 绘制所有检测到的物体（绿色）
    for obj in objs:
        detector2.draw_pose(
            img, obj.get_obb_points(), 5,
            image.Color.from_rgb(0, 255, 0),
            close=True
        )
    
    # 绘制轨迹和历史路径
    for track in tracks:
        if track.lost:
            continue
        
        # 根据是否为稳定目标选择颜色
        # 红色 = 稳定目标，蓝色 = 其他目标
        if track.id == stable_target_id:
            color = image.Color.from_rgb(255, 0, 0)
        else:
            color = image.Color.from_rgb(0, 0, 255)
        
        # 绘制运动轨迹
        if track.id in obb_map and len(obb_map[track.id]) > 1:
            history = obb_map[track.id]
            for i in range(1, len(history)):
                # 计算相邻帧的中心点
                prev_points = history[i-1].get_obb_points()
                curr_points = history[i].get_obb_points()
                
                prev_center = (sum(prev_points[::2])/4, sum(prev_points[1::2])/4)
                curr_center = (sum(curr_points[::2])/4, sum(curr_points[1::2])/4)
                
                # 绘制轨迹线
                img.draw_line(
                    int(prev_center[0]), int(prev_center[1]),
                    int(curr_center[0]), int(curr_center[1]),
                    color, 2
                )
    
    # 高亮显示稳定目标（黄色粗框）
    if stable_target_id in obb_map and obb_map[stable_target_id]:
        target_obb = obb_map[stable_target_id][-1]
        detector2.draw_pose(
            img, target_obb.get_obb_points(), 8,
            image.Color.from_rgb(255, 255, 0),
            close=True
        )
        # 显示追踪 ID
        img.draw_string(
            target_obb.x, target_obb.y - 10,
            f"ID:{stable_target_id}",
            image.Color.from_rgb(255, 255, 0),
            scale=1.0
        )
    
    return img


def wait_stable() -> bool:
    """
    判断当前追踪目标是否稳定
    
    稳定条件（连续 STABLE_CONFIG["frames"] 帧满足）：
        1. IoU > IOU_THRESH（形状位置整体变化不大）
        2. 中心点位移 < CENTER_SHIFT（像素）
        3. 角度变化 < ANGLE_THRESH（度）
    
    Returns:
        bool: 是否稳定
    """
    global stable_target_id, obb_map
    
    if stable_target_id not in obb_map:
        return False
    
    history = obb_map[stable_target_id]
    if len(history) < STABLE_CONFIG["frames"]:
        return False
    
    # 检查最近 N 帧的稳定性
    for i in range(1, STABLE_CONFIG["frames"]):
        prev_obb = history[-(i+1)]
        curr_obb = history[-i]
        
        # 检查 IoU
        if iou_rotated(prev_obb, curr_obb) < STABLE_CONFIG["iou_thresh"]:
            return False
        
        # 检查中心点位移
        prev_center = (
            sum(prev_obb.get_obb_points()[::2])/4,
            sum(prev_obb.get_obb_points()[1::2])/4
        )
        curr_center = (
            sum(curr_obb.get_obb_points()[::2])/4,
            sum(curr_obb.get_obb_points()[1::2])/4
        )
        shift = math.hypot(
            curr_center[0] - prev_center[0],
            curr_center[1] - prev_center[1]
        )
        if shift > STABLE_CONFIG["center_shift"]:
            return False
        
        # 检查角度变化
        prev_angle = calculate_real_angle(
            prev_obb.angle * 180, prev_obb.w, prev_obb.h
        )
        curr_angle = calculate_real_angle(
            curr_obb.angle * 180, curr_obb.w, curr_obb.h
        )
        if abs(curr_angle - prev_angle) > STABLE_CONFIG["angle_thresh"]:
            return False
    
    return True


def filter_close_objects(objs: list) -> list:
    """
    过滤过于接近的物体（避免重复检测同一垃圾）
    
    过滤条件：两个物体中心点 x、y 差值分别不超过两者最小宽、高
    
    Args:
        objs: 检测到的物体列表
    
    Returns:
        list: 过滤后的列表
    """
    if len(objs) <= 1:
        return objs
    
    filtered = list(objs)
    to_remove = set()
    
    for i in range(len(filtered)):
        if i in to_remove:
            continue
        
        a = filtered[i]
        a_points = a.get_obb_points()
        a_x = sum(a_points[::2]) / 4
        a_y = sum(a_points[1::2]) / 4
        a_w, a_h = a.w, a.h
        
        for j in range(i + 1, len(filtered)):
            if j in to_remove:
                continue
            
            b = filtered[j]
            b_points = b.get_obb_points()
            b_x = sum(b_points[::2]) / 4
            b_y = sum(b_points[1::2]) / 4
            b_w, b_h = b.w, b.h
            
            dx = abs(a_x - b_x)
            dy = abs(a_y - b_y)
            min_width = min(a_w, b_w)
            min_height = min(a_h, b_h)
            
            # 如果过于接近，保留置信度高的
            if dx <= min_width and dy <= min_height:
                if a.score < b.score:
                    to_remove.add(i)
                else:
                    to_remove.add(j)
    
    # 从大到小移除（避免索引偏移）
    for idx in sorted(to_remove, reverse=True):
        del filtered[idx]
    
    # 递归检查（移除后可能产生新的接近对）
    if len(filtered) != len(objs):
        return filter_close_objects(filtered)
    
    return filtered


# ============================================================================
# 主程序
# ============================================================================

serial.set_received_callback(on_received)

# 当前待处理垃圾数量（由外部指令设置）
numbers = 1

while not app.need_exit():
    # --------------------------------------------------------------------
    # 图像采集与预处理
    # --------------------------------------------------------------------
    img = cam.read()
    img_roi = img.crop(ROI_X, ROI_Y, ROI_W, ROI_H)
    display_img = img_roi.copy()
    
    # --------------------------------------------------------------------
    # 状态 1: 第一轮检测（start=True, nextt=False）
    # --------------------------------------------------------------------
    if start and not nextt:
        # 检查超时
        elapsed_time = (time.ticks_ms() - start_time) / 1000
        print(f"[TIME] Elapsed: {elapsed_time:.1f}s")
        
        if elapsed_time >= TIMEOUT_SECONDS:
            # 超时处理：发送默认值
            start = False
            nextt = False
            default_data = "320,320,50,50,90,0,1,1,1\n"
            send_data(default_data)
            print("[TIMEOUT] First detection timeout, sending default values")
            start_time = 0
            continue
        
        # 检测与追踪
        objs = detector1.detect(img_roi, conf_th=0.6, iou_th=0.55)
        
        if objs:
            # 转换为追踪器格式
            tracker_objs = []
            for obj in objs:
                x, y, w, h = obb_to_bbox(obj)
                tracker_objs.append(tracker.Object(x, y, w, h, 0, obj.score))
            
            # 更新追踪
            tracks = object_tracker.update(tracker_objs)
            
            # 更新 OBB 历史映射
            new_obb_map = {}
            for track in tracks:
                if track.lost:
                    continue
                track_id = track.id
                
                # 匹配追踪 ID 与当前检测框
                for obj in objs:
                    if track_id in obb_map:
                        last_obb = obb_map[track_id][-1]
                        if iou_rotated(obj, last_obb) > 0.3:
                            new_obb_map[track_id] = obb_map[track_id] + [obj]
                            break
                else:
                    new_obb_map[track_id] = [obj]
            
            obb_map = new_obb_map
            
            # 选择优先级最高的目标
            target_obj = choose_target(objs)
            
            # 更新稳定目标 ID
            for track_id, obb_list in obb_map.items():
                if iou_rotated(obb_list[-1], target_obj) > 0.5:
                    stable_target_id = track_id
                    break
            
            # 绘制追踪效果
            display_img = draw_tracking(display_img, objs, tracks, obb_map)
            
            # 等待目标稳定后，执行最终检测
            if wait_stable():
                final_img = img.copy().crop(ROI_X, ROI_Y, ROI_W, ROI_H)
                
                # 多级置信度检测（提高检出率）
                final_objs = detector2.detect(final_img, conf_th=0.6, iou_th=0.55)
                if not final_objs:
                    final_objs = detector2.detect(final_img, conf_th=0.2, iou_th=0.55)
                if len(final_objs) < numbers:
                    for conf in [0.5, 0.4, 0.3]:
                        if len(final_objs) >= numbers:
                            break
                        final_objs = detector2.detect(final_img, conf_th=conf, iou_th=0.55)
                
                # 过滤接近物体
                final_objs = filter_close_objects(final_objs)
                
                # 处理单垃圾情况
                if len(final_objs) == 1:
                    send_obj = final_objs[0]
                    points = send_obj.get_obb_points()
                    x = int(sum(points[::2])/4)
                    y = int(sum(points[1::2])/4)
                    
                    final_img.draw_cross(x, y, image.Color.from_rgb(0, 0, 0), size=25, thickness=3)
                    detector2.draw_pose(final_img, points, 10, image.Color.from_rgb(255, 0, 0), close=True)
                    
                    degree = calculate_real_angle(send_obj.angle * 180, send_obj.w, send_obj.h)
                    classk = get_garbage_category(send_obj)
                    
                    # 构建发送数据
                    # 格式: x,y,w,h,angle,class,num,flag,seq
                    flag = 1 if numbers == 1 else 0
                    data_str = f"{x},{y},{send_obj.w:.0f},{send_obj.h:.0f},{degree:.0f},{classk},{len(final_objs)},{flag},1\n"
                    
                    result = numbers
                    result2 = [send_obj]
                    send_data(data_str)
                    print(f"[SEND] Single garbage: {data_str.strip()}")
                    
                    disp.show(final_img)
                    time.sleep_ms(1000)
                    
                    # 重置状态
                    start = False
                    nextt = False
                    obb_map = {}
                    stable_target_id = None
                
                # 处理多垃圾情况
                elif len(final_objs) >= 2:
                    # 绘制所有检测到的垃圾
                    for tmp in final_objs:
                        points = tmp.get_obb_points()
                        x = int(sum(points[::2])/4)
                        y = int(sum(points[1::2])/4)
                        final_img.draw_cross(x, y, image.Color.from_rgb(0, 0, 255), size=25, thickness=3)
                        detector2.draw_pose(final_img, points, 10, image.Color.from_rgb(0, 0, 255), close=True)
                    
                    # 选择优先级最高的发送
                    send_obj = choose_target(final_objs)
                    points = send_obj.get_obb_points()
                    x = int(sum(points[::2])/4)
                    y = int(sum(points[1::2])/4)
                    
                    final_img.draw_cross(x, y, image.Color.from_rgb(255, 0, 0), size=25, thickness=6)
                    detector2.draw_pose(final_img, points, 10, image.Color.from_rgb(255, 0, 0), close=True)
                    
                    degree = calculate_real_angle(send_obj.angle * 180, send_obj.w, send_obj.h)
                    classk = get_garbage_category(send_obj)
                    
                    # 多垃圾标志
                    flag = 1 if numbers == 1 else 0
                    data_str = f"{x},{y},{send_obj.w:.0f},{send_obj.h:.0f},{degree:.0f},{classk},{len(final_objs)},{flag},1\n"
                    
                    result = len(final_objs)
                    result2 = list(final_objs)
                    send_data(data_str)
                    print(f"[SEND] Multiple garbage: {data_str.strip()}")
                    
                    disp.show(final_img)
                    time.sleep_ms(1000)
                    
                    start = False
                    nextt = False
                    obb_map = {}
                    stable_target_id = None
        
        disp.show(display_img)
    
    # --------------------------------------------------------------------
    # 状态 2: 处理下一个垃圾（start=True, nextt=True）
    # --------------------------------------------------------------------
    elif start and nextt:
        elapsed_time = (time.ticks_ms() - start_time) / 1000
        
        if elapsed_time >= TIMEOUT_SECONDS:
            # 超时处理
            start = False
            nextt = False
            default_data = "320,320,50,50,90,0,1,1,1\n"
            send_data(default_data)
            print("[TIMEOUT] Next detection timeout")
            start_time = 0
            continue
        
        objs = detector1.detect(img_roi, conf_th=0.6, iou_th=0.55)
        
        # 如果上一轮是单垃圾且没检测到，说明已处理完毕
        if result == 1:
            if not objs:
                send_data("0\n")
                print("[DONE] Round completed, waiting for next round")
                start = False
                nextt = False
                result = 0
                result2 = []
                obb_map = {}
                stable_target_id = None
                disp.show(img_roi)
                continue
            else:
                print("[WARN] Previous single garbage not dropped")
        
        if objs:
            # 追踪更新（同第一轮）
            tracker_objs = []
            for obj in objs:
                x, y, w, h = obb_to_bbox(obj)
                tracker_objs.append(tracker.Object(x, y, w, h, 0, obj.score))
            
            tracks = object_tracker.update(tracker_objs)
            
            new_obb_map = {}
            for track in tracks:
                if track.lost:
                    continue
                track_id = track.id
                for obj in objs:
                    if track_id in obb_map:
                        last_obb = obb_map[track_id][-1]
                        if iou_rotated(obj, last_obb) > 0.3:
                            new_obb_map[track_id] = obb_map[track_id] + [obj]
                            break
                else:
                    new_obb_map[track_id] = [obj]
            
            obb_map = new_obb_map
            target_obj = choose_target(objs)
            
            for track_id, obb_list in obb_map.items():
                if iou_rotated(obb_list[-1], target_obj) > 0.5:
                    stable_target_id = track_id
                    break
            
            display_img = draw_tracking(display_img, objs, tracks, obb_map)
            disp.show(display_img)
            
            # 等待稳定后检测
            if wait_stable():
                final_img = img.copy().crop(ROI_X, ROI_Y, ROI_W, ROI_H)
                final_objs = detector2.detect(final_img, conf_th=0.6, iou_th=0.55)
                final_objs = filter_close_objects(final_objs)
                
                if not final_objs:
                    final_objs = detector2.detect(final_img, conf_th=0.2, iou_th=0.55)
                
                # 处理上一轮单垃圾未成功的情况
                if result == 1 and len(final_objs) >= 1:
                    print("[RETRY] Resending undropped single garbage")
                    send_obj = final_objs[0]
                    points = send_obj.get_obb_points()
                    x = int(sum(points[::2])/4)
                    y = int(sum(points[1::2])/4)
                    
                    final_img.draw_cross(x, y, image.Color.from_rgb(0, 0, 0), size=25, thickness=3)
                    detector2.draw_pose(final_img, points, 10, image.Color.from_rgb(255, 0, 0), close=True)
                    
                    degree = calculate_real_angle(send_obj.angle * 180, send_obj.w, send_obj.h)
                    classk = get_garbage_category(send_obj)
                    
                    data_str = f"{x},{y},{send_obj.w:.0f},{send_obj.h:.0f},{degree:.0f},{classk},1,1,2\n"
                    send_data(data_str)
                    disp.show(final_img)
                    print("[SEND] Retry sent")
                    start = False
                    nextt = False
                    result = 1
                
                # 处理多垃圾的后续分拣
                elif result != 1:
                    # ...（多垃圾后续处理逻辑，与第一轮类似）
                    pass
    
    # --------------------------------------------------------------------
    # 状态 3: 等待指令
    # --------------------------------------------------------------------
    elif not start and not nextt:
        disp.show(img_roi)
    
    time.sleep_ms(20)

print("[EXIT] System Exit")
