# 智能垃圾分类系统 - 技术详解

## 目录
1. [项目架构概述](#1-项目架构概述)
2. [YOLO11-OBB 原理详解](#2-yolo11-obb-原理详解)
3. [目标追踪：ByteTrack 算法](#3-目标追踪bytetrack-算法)
4. [后处理优化策略](#4-后处理优化策略)
5. [嵌入式部署方案](#5-嵌入式部署方案)
6. [硬件通信协议](#6-硬件通信协议)
7. [关键代码解析](#7-关键代码解析)

---

## 1. 项目架构概述

### 1.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        应用层 (Application)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ 单垃圾处理    │  │ 多垃圾处理    │  │ 多垃圾追踪+分拣        │   │
│  │ single.py    │  │ multi.py     │  │ tracking.py          │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                        算法层 (Algorithm)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ YOLO11-OBB   │  │ ByteTrack    │  │ Soft-NMS + 过滤      │   │
│  │ 目标检测      │  │ 目标追踪      │  │ 后处理               │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                        框架层 (Framework)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ MaixPy       │  │ OpenCV       │  │ NumPy               │   │
│  │ 嵌入式AI框架  │  │ 图像处理      │  │ 数值计算             │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                        硬件层 (Hardware)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Camera       │  │ NPU (0.5T)   │  │ UART Controller     │   │
│  │ 摄像头        │  │ 神经网络加速器│  │ 串口通信             │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 数据流图

```
Camera Frame (1920x1080)
         │
         ▼
┌─────────────────┐
│   ROI Crop      │ ← 裁剪有效检测区域 (640x640)
│  (200,130,640,640)│
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  YOLO11-OBB     │ ← 目标检测 + 角度预测
│  Inference      │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Post-process   │ ← NMS / Soft-NMS / 过滤
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  ByteTrack      │ ← 目标追踪 (多垃圾场景)
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Priority Sort  │ ← 按类别+面积排序
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  UART Output    │ ← x,y,w,h,angle,class,num,flag
└─────────────────┘
```

---

## 2. YOLO11-OBB 原理详解

### 2.1 什么是 OBB (Oriented Bounding Box)

**传统 HBB (Horizontal Bounding Box)**
```
┌─────────────────┐
│                 │  ← 只能表示水平矩形
│     倾斜物体     │     会包含大量背景
│                 │
└─────────────────┘
```

**OBB (Oriented Bounding Box)**
```
       ┌──────┐
      /      /
     / 物体  /    ← 旋转矩形，贴合物体角度
    /      /
   └──────┘
```

### 2.2 OBB 表示方法

YOLO11-OBB 使用 **8点坐标** 或 **中心点+宽高+角度** 表示：

```python
# 8点表示 (4个顶点，每个顶点2个坐标)
points = [x1, y1, x2, y2, x3, y3, x4, y4]

# 中心点+宽高+角度表示
obb = {
    'x': 320,      # 中心点 x
    'y': 240,      # 中心点 y
    'w': 100,      # 宽度
    'h': 50,       # 高度
    'angle': 0.3   # 旋转角度 (弧度，范围 [-pi/4, pi/4])
}
```

### 2.3 YOLO11-OBB 网络结构

```
Input (640x640x3)
       │
       ▼
┌──────────────┐
│  Backbone    │ ← CSPDarknet 特征提取
│  (特征提取)   │
└──────────────┘
       │
       ├──► P3 (80x80)   ──► 小目标检测
       ├──► P4 (40x40)   ──► 中目标检测
       └──► P5 (20x20)   ──► 大目标检测
       │
       ▼
┌──────────────┐
│     Neck     │ ← PANet 特征融合
│  (特征融合)   │
└──────────────┘
       │
       ▼
┌──────────────┐
│    Head      │ ← 检测头
│   (检测头)    │
└──────────────┘
       │
       ├──► cls: 类别概率 [num_classes]
       ├──► box: 边界框 [x, y, w, h]
       ├──► angle: 旋转角度 [1]
       └──► conf: 置信度 [1]
```

### 2.4 OBB 检测头的特殊性

标准 YOLO 检测头输出：
```python
# [batch, num_anchors, 4 + 1 + num_classes]
# 4: x, y, w, h
# 1: objectness
# num_classes: 类别概率
```

YOLO11-OBB 检测头输出：
```python
# [batch, num_anchors, 4 + 1 + num_classes + 1]
# 4: x, y, w, h
# 1: objectness
# num_classes: 类别概率
# 1: angle (旋转角度)
```

### 2.5 角度预测的特殊处理

```python
# 角度归一化到 [-pi/4, pi/4]
# 因为旋转90度相当于交换宽高，所以只需要预测90度范围

def normalize_angle(angle_rad):
    """将角度归一化到 [-pi/4, pi/4]"""
    while angle_rad > np.pi/4:
        angle_rad -= np.pi/2
    while angle_rad < -np.pi/4:
        angle_rad += np.pi/2
    return angle_rad

# 实际使用时的角度转换
def get_real_angle(obj):
    """
    根据预测角度和宽高关系计算实际角度
    """
    angle = obj.angle * 180  # 转为度
    w, h = obj.w, obj.h
    
    if (angle < 10 and w > h) or (angle > 80 and w < h):
        return 0.0
    elif (angle < 10 and w < h) or (angle > 80 and w > h):
        return 90.0
    elif angle > 10 and w > h:
        return angle
    else:  # angle > 10 and w < h
        return 90 + angle
```

### 2.6 OBB 的损失函数

YOLO11-OBB 使用以下损失函数组合：

```python
# 1. 分类损失 (BCE Loss)
L_cls = -Σ[ y*log(p) + (1-y)*log(1-p) ]

# 2. 置信度损失 (BCE Loss)
L_conf = -Σ[ obj*log(conf) + (1-obj)*log(1-conf) ]

# 3. 边界框损失 (CIoU Loss / GIoU Loss)
L_box = 1 - CIoU(pred_box, target_box)

# 4. 角度损失 (MSE Loss / Smooth L1)
L_angle = SmoothL1(pred_angle, target_angle)

# 总损失
L_total = λ_cls*L_cls + λ_conf*L_conf + λ_box*L_box + λ_angle*L_angle
```

### 2.7 OBB IoU 计算

OBB 之间的 IoU 计算比较复杂，需要用到 **旋转框交集算法** 或近似计算：

```python
def iou_rotated(obb1, obb2):
    """
    计算两个OBB之间的IoU
    简化版本：使用外接水平框近似计算
    """
    # 获取OBB的外接水平框
    bbox1 = obb_to_bbox(obb1)  # (x_min, y_min, w, h)
    bbox2 = obb_to_bbox(obb2)
    
    # 计算水平框的IoU
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y2 = min(bbox1[1] + bbox3[3], bbox2[1] + bbox2[3])
    
    if x1 >= x2 or y1 >= y2:
        return 0.0
    
    inter = (x2 - x1) * (y2 - y1)
    union = bbox1[2]*bbox1[3] + bbox2[2]*bbox2[3] - inter
    
    return inter / union if union > 0 else 0.0
```

---

## 3. 目标追踪：ByteTrack 算法

### 3.1 ByteTrack 核心思想

ByteTrack 是一种简单高效的 **在线目标追踪算法**，核心思想：

1. **利用高分检测框和低分检测框**
   - 高分框 (score > 0.6)：用于初始化新轨迹和匹配已有轨迹
   - 低分框 (0.3 < score < 0.6)：用于匹配未确认的轨迹

2. **卡尔曼滤波预测**
   - 预测目标在下一帧的位置
   - 状态变量：[x, y, w, h, vx, vy, vw, vh]

3. **匈牙利算法匹配**
   - 使用 IoU 或外观特征作为代价矩阵
   - 解决二分图最大匹配问题

### 3.2 ByteTrack 算法流程

```
输入：当前帧检测框 D = {d1, d2, ..., dn}
      已有轨迹 T = {t1, t2, ..., tm}

Step 1: 卡尔曼滤波预测
        for each t in T:
            t.predict()  # 预测下一帧位置

Step 2: 高分框匹配 (score > 0.6)
        D_high = filter(D, score > 0.6)
        使用匈牙利算法匹配 D_high 和 T
        - 匹配的轨迹更新状态
        - 未匹配的轨迹标记为 "丢失"
        - 未匹配的检测框作为 "候选新轨迹"

Step 3: 低分框匹配 (0.3 < score < 0.6)
        D_low = filter(D, 0.3 < score < 0.6)
        使用匈牙利算法匹配 D_low 和 "未确认/丢失的轨迹"
        - 解决遮挡导致的漏检问题

Step 4: 更新轨迹状态
        - 匹配的轨迹：更新卡尔曼滤波状态
        - 未匹配的轨迹：lost_count++
        - 新检测框：初始化新轨迹（需要连续3帧确认）

Step 5: 删除失效轨迹
        - lost_count > max_lost 的轨迹删除
```

### 3.3 ByteTrack 在本项目中的应用

```python
from maix import tracker

# 初始化 ByteTracker
object_tracker = tracker.ByteTracker(
    max_lost_buff_num=15,   # 最大丢失帧数（遮挡容忍）
    track_thresh=0.4,       # 追踪置信度阈值
    high_thresh=0.6,        # 高分阈值
    match_thresh=0.5,       # 匹配IoU阈值
    max_history=10          # 保留历史轨迹点数
)

# 转换检测框格式
for obj in objs:
    x, y, w, h = obb_to_bbox(obj)
    tracker_objs.append(tracker.Object(x, y, w, h, class_id=0, score=obj.score))

# 更新追踪
tracks = object_tracker.update(tracker_objs)

# tracks 包含：
# - track.id: 轨迹ID
# - track.lost: 是否丢失
# - track.history: 历史位置
```

---

## 4. 后处理优化策略

### 4.1 Soft-NMS (非极大值抑制)

**传统 NMS 的问题**：
- 直接删除重叠框，可能导致漏检
- 对于遮挡场景效果不好

**Soft-NMS 改进**：
- 降低重叠框的置信度，而非直接删除
- 给重叠框重新参与竞争的机会

```python
def soft_nms(boxes, scores, sigma=0.5, Nt=0.3, threshold=0.001, method=2):
    """
    Soft NMS 实现
    
    Args:
        boxes: [N, 4] 检测框坐标 (x1,y1,x2,y2)
        scores: [N] 置信度
        sigma: 高斯加权参数
        Nt: IoU阈值
        threshold: 最低保留置信度
        method: 1-线性加权, 2-高斯加权
    
    Returns:
        keep: 保留的索引
    """
    N = boxes.shape[0]
    indexes = np.arange(N).reshape(-1, 1)
    boxes = np.hstack((boxes, indexes))

    for i in range(N):
        maxscore = scores[i].item()
        maxpos = i

        # 找到当前最大置信度框
        pos = i + 1
        while pos < N:
            current_score = scores[pos].item()
            if maxscore < current_score:
                maxscore = current_score
                maxpos = pos
            pos += 1

        # 交换到当前位置
        boxes[i, :4], boxes[maxpos, :4] = boxes[maxpos, :4].copy(), boxes[i, :4].copy()
        scores[i], scores[maxpos] = scores[maxpos], scores[i]

        # 处理后续框
        pos = i + 1
        while pos < N:
            box = boxes[pos, :4]
            overlap = iou(np.array([boxes[i, :4]]), box)
            
            # 计算权重
            if method == 1:  # 线性加权
                weight = 1 - overlap if overlap > Nt else 1
            elif method == 2:  # 高斯加权
                weight = np.exp(-(overlap ** 2) / sigma)
            
            # 降低置信度
            scores_pos = scores[pos].item()
            new_score = scores_pos * weight
            scores[pos] = new_score

            # 删除低置信度框
            if new_score < threshold:
                boxes[pos] = boxes[-1]
                scores[pos] = scores[-1]
                boxes = boxes[:-1]
                scores = scores[:-1]
                N -= 1
                pos -= 1
            pos += 1

    keep = boxes[:, 4].astype(int)
    return keep
```

### 4.2 IOU 过滤 + 距离过滤

```python
def filter_close_objects(objs, distance_threshold=80):
    """
    过滤过于接近的物体
    当两个物体中心点距离小于阈值时，保留置信度高的
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
        
        for j in range(i + 1, len(filtered)):
            if j in to_remove:
                continue
            b = filtered[j]
            b_points = b.get_obb_points()
            b_x = sum(b_points[::2]) / 4
            b_y = sum(b_points[1::2]) / 4
            
            # 计算距离
            distance = math.sqrt((a_x - b_x)**2 + (a_y - b_y)**2)
            
            if distance < distance_threshold:
                # 保留置信度高的
                if a.score < b.score:
                    to_remove.add(i)
                else:
                    to_remove.add(j)
    
    # 删除标记的索引
    for idx in sorted(to_remove, reverse=True):
        del filtered[idx]
    
    return filtered
```

### 4.3 优先级排序策略

```python
# 类别优先级（有害垃圾优先处理）
label_priority = {
    "youhai": 4,    # 有害垃圾 - 最高优先级
    "kehuishou": 3, # 可回收
    "chuyu": 2,     # 厨余
    "qita": 1       # 其他
}

def calculate_priority(obj, max_area):
    """
    计算目标优先级
    综合考虑：面积 + 类别
    """
    # 面积得分 (0-0.5)
    area_score = (obj.w * obj.h / max_area) * 0.5
    
    # 类别得分 (0-0.5)
    label = get_label_by_id(obj.class_id)
    label_score = label_priority.get(label, 0) * 0.5
    
    return area_score + label_score

# 选择优先级最高的目标
priorities = [calculate_priority(obj, max_area) for obj in final_objs]
selected_index = priorities.index(max(priorities))
selected_obj = final_objs[selected_index]
```

---

## 5. 嵌入式部署方案

### 5.1 模型转换流程

```
PyTorch Model (.pt)
        │
        ▼
┌─────────────────┐
│  Export to ONNX │ ← torch.onnx.export()
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ ONNX Simplify   │ ← onnxsim，简化计算图
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ ONNX → MLIR     │ ← 转换为中间表示
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ MLIR → CVIModel │ ← 目标芯片编译
│ Quantization    │ ← INT8 量化
└─────────────────┘
        │
        ▼
   CVIModel (.cvimodel)
   MUD Config (.mud)
```

### 5.2 INT8 量化

```python
# 量化校准
# 使用代表性数据集进行后训练量化 (PTQ)
# 将 FP32 权重和激活值映射到 INT8

# 量化公式：
# scale = (max - min) / 255
# zero_point = -round(min / scale)
# quantized = round(fp32 / scale) + zero_point
```

### 5.3 MaixPy 推理代码

```python
from maix import camera, display, image, nn, app

# 加载模型
detector = nn.YOLO11(
    model="/root/myproject/my_mud.mud",
    dual_buff=True  # 双缓冲优化
)

# 初始化摄像头
cam = camera.Camera(
    1920, 1080,           # 原始分辨率
    detector.input_format()  # 模型输入格式
)

# 初始化显示
disp = display.Display()

# 主循环
while not app.need_exit():
    # 读取帧
    img = cam.read()
    
    # ROI 裁剪
    img_roi = img.crop(200, 130, 640, 640)
    
    # 推理
    objs = detector.detect(
        img_roi,
        conf_th=0.6,    # 置信度阈值
        iou_th=0.55     # NMS IoU阈值
    )
    
    # 后处理...
    
    # 显示
    disp.show(img_roi)
```

---

## 6. 硬件通信协议

### 6.1 UART 通信配置

```python
from maix import uart

# 配置串口
device = "/dev/ttyS0"
serial = uart.UART(device, 115200)  # 波特率 115200

# 设置接收回调
def on_received(serial, data):
    cmd = data.decode().strip().lower()
    if cmd == "start":
        start_detection()
    elif cmd == "next":
        process_next()

serial.set_received_callback(on_received)
```

### 6.2 数据帧格式

```
数据帧格式：x,y,w,h,angle,class,num,flag\n
字段说明：
- x: 中心点 X 坐标 (像素)
- y: 中心点 Y 坐标 (像素)
- w: 宽度 (像素)
- h: 高度 (像素)
- angle: 旋转角度 (度，0-180)
- class: 垃圾类别 (0-3, 对应4大类)
- num: 垃圾数量
- flag: 标志位 (0/1)

示例：
"320,240,100,50,90,1,1,1\n"
```

### 6.3 通信状态机

```
┌─────────┐    start     ┌──────────┐
│  IDLE   │ ───────────► │ DETECTING│
│ (等待)  │              │ (检测中) │
└─────────┘              └────┬─────┘
     ▲                        │
     │ next                   │ detect done
     │                        ▼
     │                   ┌──────────┐
     └────────────────── │ SENDING  │
                         │ (发送中) │
                         └──────────┘
```

---

## 7. 关键代码解析

### 7.1 角度判断函数

```python
def panduan(angles, ww, hh):
    """
    根据预测角度和宽高关系计算实际角度
    
    逻辑：
    1. 如果角度接近0度且宽>高，或角度接近90度且高>宽 → 实际0度
    2. 如果角度接近0度且高>宽，或角度接近90度且宽>高 → 实际90度
    3. 其他情况根据宽高关系加上90度偏移
    """
    if (angles < 10 and ww > hh) or (angles > 80 and ww < hh):
        angles = 0.0
    elif (angles < 10 and ww < hh) or (angles > 80 and ww > hh):
        angles = 90.0
    elif angles > 10 and ww > hh:
        angles = angles
    elif angles > 10 and ww < hh:
        angles = 90 + angles
    return angles
```

### 7.2 稳定检测机制

```python
def wait_stable():
    """
    判断目标是否稳定（连续多帧位置/角度变化小）
    """
    global stable_target_id, obb_map
    
    if stable_target_id not in obb_map:
        return False
    
    history = obb_map[stable_target_id]
    if len(history) < STABLE_FRAMES:  # 需要连续5帧
        return False
    
    # 检查最近5帧的稳定性
    for i in range(1, STABLE_FRAMES):
        prev_obb = history[-(i+1)]
        curr_obb = history[-i]
        
        # 检查IoU变化
        if iou_rotated(prev_obb, curr_obb) < IOU_THRESH:
            return False
        
        # 检查中心点位移
        prev_center = get_center(prev_obb)
        curr_center = get_center(curr_obb)
        shift = math.hypot(curr_center[0]-prev_center[0], 
                          curr_center[1]-prev_center[1])
        if shift > CENTER_SHIFT_THRESH:
            return False
        
        # 检查角度变化
        prev_angle = get_angle(prev_obb)
        curr_angle = get_angle(curr_obb)
        if abs(curr_angle - prev_angle) > ANGLE_THRESH:
            return False
    
    return True
```

### 7.3 超时处理机制

```python
timeout_seconds = 17  # 超时时间
start_time = 0

def process_detection():
    global start_time
    
    if start:
        elapsed_time = (time.ticks_ms() - start_time) / 1000
        
        if elapsed_time >= timeout_seconds:
            # 超时，发送默认数据
            send_default_data()
            reset_state()
            return
        
        # 正常检测流程...
```

---

## 附录：性能指标

| 指标 | 数值 |
|------|------|
| 输入分辨率 | 640x640 |
| 推理速度 | ~30 FPS |
| 检测类别 | 17类 |
| mAP@0.5 | > 85% |
| 角度精度 | ±5度 |
| 追踪ID保持 | > 90% |
| 串口延迟 | < 10ms |
