"""
OBB (Oriented Bounding Box) Detection Demo
有向边界框检测演示程序

本程序演示如何使用 YOLO11-OBB 模型进行垃圾检测，并输出旋转角度信息。
适用于需要获取垃圾倾斜角度的场景，如机械臂抓取姿态规划。

Author: yucheng
Date: 2025
Platform: MaixPy (Sipeed SG2002)
"""

from maix import camera, display, image, nn, app
from maix import image

# ============================================================================
# 模型初始化
# ============================================================================

# 加载 YOLO11-OBB 模型
# dual_buff=True 启用双缓冲，提高推理吞吐量
# 模型文件 .cvimodel 和 .mud 需要预先部署到设备指定路径
detector = nn.YOLO11(
    model="/root/myproject/my_mud.mud",
    dual_buff=True
)

# 初始化摄像头
# 参数说明：
#   - 1920, 1080: 摄像头原始分辨率
#   - detector.input_format(): 模型输入格式（RGB/BGR等）
# 使用高分辨率输入可以获得更大的视野，后续通过 crop 裁剪 ROI
cam = camera.Camera(1920, 1080, detector.input_format())

# 初始化显示设备
disp = display.Display()

# 打印模型输入尺寸信息（用于调试）
print(f"[INFO] Model input width: {detector.input_width()}")
print(f"[INFO] Model input height: {detector.input_height()}")


# ============================================================================
# 角度计算函数
# ============================================================================

def calculate_real_angle(raw_angle: float, width: float, height: float) -> float:
    """
    根据原始预测角度和边界框宽高，计算实际旋转角度。
    
    YOLO-OBB 输出的角度范围是 [-45°, 45°]，且与宽高无关。
    本函数根据宽高关系将角度映射到 [0°, 180°] 范围。
    
    Args:
        raw_angle: 模型输出的原始角度（度）
        width: 边界框宽度
        height: 边界框高度
    
    Returns:
        float: 实际旋转角度（0° ~ 180°）
    
    逻辑说明：
        - 如果 raw_angle < 10° 且 width > height → 视为水平，返回 0°
        - 如果 raw_angle < 10° 且 width < height → 视为垂直，返回 90°
        - 如果 raw_angle > 10° → 根据宽高关系加上 90° 偏移
    """
    if (raw_angle < 10 and width > height) or (raw_angle > 80 and width < height):
        return 0.0
    elif (raw_angle < 10 and width < height) or (raw_angle > 80 and width > height):
        return 90.0
    elif raw_angle > 10 and width > height:
        return raw_angle
    else:  # raw_angle > 10 and width < height
        return 90 + raw_angle


# ============================================================================
# 主程序循环
# ============================================================================

while not app.need_exit():
    # ------------------------------------------------------------------------
    # 1. 图像采集与预处理
    # ------------------------------------------------------------------------
    
    # 读取一帧图像
    img = cam.read()
    
    # ROI 裁剪：从 (555, 150) 开始，裁剪 768x768 区域
    # 目的：
    #   1. 减少模型输入尺寸，提高推理速度
    #   2. 聚焦到感兴趣区域（如传送带中央）
    #   3. 避免边缘畸变影响检测精度
    img_roi = img.crop(555, 150, 768, 768)
    
    # ------------------------------------------------------------------------
    # 2. 模型推理
    # ------------------------------------------------------------------------
    
    # 执行检测
    # 参数说明：
    #   - conf_th=0.7: 置信度阈值，过滤低置信度检测结果
    #   - iou_th=0.5:  NMS IoU阈值，控制重叠框合并程度
    detected_objects = detector.detect(img_roi, conf_th=0.7, iou_th=0.5)
    
    # ------------------------------------------------------------------------
    # 3. 后处理与可视化
    # ------------------------------------------------------------------------
    
    for obj in detected_objects:
        # 获取 OBB 的 8 个顶点坐标 [x1,y1,x2,y2,x3,y3,x4,y4]
        points = obj.get_obb_points()
        
        # 计算实际旋转角度
        raw_angle = obj.angle * 180  # 模型输出为弧度，转为度
        real_angle = calculate_real_angle(raw_angle, obj.w, obj.h)
        
        # 计算 OBB 中心点坐标（4个顶点坐标的平均值）
        # points[::2] 取所有 x 坐标（偶数索引）
        # points[1::2] 取所有 y 坐标（奇数索引）
        center_x = (points[0] + points[2] + points[4] + points[6]) / 4
        center_y = (points[1] + points[3] + points[5] + points[7]) / 4
        
        # 打印检测信息（用于调试）
        print(f"[DETECT] Center: ({center_x:.0f}, {center_y:.0f}), "
              f"Class: {detector.labels[obj.class_id]}, "
              f"Score: {obj.score:.2f}, "
              f"Angle: {real_angle:.1f}°")
        
        # -------------------------------------------------------------------
        # 可视化绘制
        # -------------------------------------------------------------------
        
        # 准备显示文本
        label_text = f'{detector.labels[obj.class_id]}: {obj.score:.2f}, {real_angle:.1f}°'
        coord_text = f'X:{center_x:.0f}\nY:{center_y:.0f}'
        
        # 在边界框角点绘制类别和置信度信息
        # 使用红色和蓝色区分不同角点，便于观察旋转方向
        img_roi.draw_string(
            points[4], points[5],  # 绘制位置（第3个顶点）
            label_text,
            color=image.COLOR_RED,
            scale=2
        )
        img_roi.draw_string(
            points[0], points[1],  # 绘制位置（第1个顶点）
            label_text,
            color=image.COLOR_BLUE,
            scale=2
        )
        
        # 在图像底部中央绘制坐标信息
        img_roi.draw_string(
            580, 650,
            coord_text,
            color=image.COLOR_RED,
            scale=3,
            thickness=2
        )
        
        # 绘制 OBB 边界框
        # 参数说明：
        #   - img_roi: 目标图像
        #   - points:  8个顶点坐标
        #   - 8:       线条宽度（根据输入尺寸自适应调整）
        #   - image.COLOR_RED: 线条颜色
        #   - close=True: 闭合多边形
        detector.draw_pose(
            img_roi, points,
            8 if detector.input_width() > 480 else 4,
            image.COLOR_RED,
            close=True
        )
    
    # ------------------------------------------------------------------------
    # 4. 显示结果
    # ------------------------------------------------------------------------
    
    disp.show(img_roi)
