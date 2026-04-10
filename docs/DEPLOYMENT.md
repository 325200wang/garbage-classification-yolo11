# 模型部署指南

本文档介绍如何将 YOLO11-OBB 模型部署到嵌入式设备。

---

## 📋 部署流程概览

```
PyTorch (.pt)
    │
    ▼
ONNX (.onnx) ──► ONNX Simplify
    │
    ▼
MLIR (.mlir) ──► 量化 (INT8/BF16)
    │
    ▼
CVIModel (.cvimodel) + MUD (.mud)
    │
    ▼
嵌入式推理 (MaixPy)
```

---

## 🛠️ 环境准备

### 主机端 (模型转换)

```bash
# 安装模型转换工具链 (以 Sophgo 为例)
# 具体工具链根据你的嵌入式平台选择

# 1. 安装依赖
pip install onnx onnxoptimizer onnxsim numpy

# 2. 下载模型转换工具链
# 通常由芯片厂商提供，如 Sophgo、Rockchip 等
```

### 设备端 (推理)

```bash
# MaixPy 环境
# 参考: https://wiki.sipeed.com/maixpy/
```

---

## 📦 模型转换

### 步骤 1: PyTorch → ONNX

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/train/yolo11s_obb_garbage/weights/best.pt')

# 导出 ONNX
model.export(
    format='onnx',
    imgsz=640,
    simplify=True,      # 使用 onnxsim 简化
    opset=12,           # ONNX opset 版本
    dynamic=False       # 静态输入尺寸
)

# 输出: best.onnx
```

### 步骤 2: ONNX 优化

```bash
# 使用 onnxsim 简化模型
onnxsim best.onnx best_sim.onnx

# 或使用 onnxoptimizer
python -m onnxoptimizer best.onnx best_opt.onnx
```

### 步骤 3: ONNX → MLIR → CVIModel

使用芯片厂商提供的工具链进行转换：

```bash
# Sophgo 工具链示例
# 1. 转换为 MLIR
model_transform.py \
    --model_name yolo11s_obb \
    --model_def best_sim.onnx \
    --input_shapes [[1,3,640,640]] \
    --mean 0.0,0.0,0.0 \
    --scale 1.0,1.0,1.0 \
    --keep_aspect_ratio \
    --pixel_format rgb \
    --output best.mlir

# 2. INT8 量化
# 准备校准数据集 (约 100-200 张代表性图片)
mkdir calibration_data
# 复制图片到 calibration_data/

# 运行量化
run_calibration.py \
    best.mlir \
    --dataset calibration_data \
    --input_num 100 \
    -o best_cali_table

# 3. 编译为 CVIModel
model_deploy.py \
    --mlir best.mlir \
    --quantize INT8 \
    --calibration_table best_cali_table \
    --chip cv181x \
    --model best_int8.cvimodel
```

### 量化类型选择

| 类型 | 精度 | 速度 | 模型大小 | 适用场景 |
|------|------|------|----------|----------|
| FP32 | 最高 | 慢 | 大 | 服务器端 |
| BF16 | 高 | 较快 | 中 | 高性能设备 |
| INT8 | 中高 | 快 | 小 | 边缘设备 |
| INT4 | 中 | 最快 | 最小 | 极低功耗设备 |

---

## 📂 MUD 配置文件

MUD (Model Description) 文件描述模型输入输出：

```yaml
# deployment/yolo_mud_config.yaml
type: yolo11_obb
name: yolo11s-obb-garbage
version: 1.0.0

model_path: ./my_model.cvimodel

nms_threshold: 0.7
confidence_threshold: 0.58

classes:
  - kehuishou
  - chuyu
  - youhai
  - qita

input:
  width: 640
  height: 640
  format: RGB
```

---

## 🚀 嵌入式推理

### MaixPy 代码示例

```python
from maix import camera, display, image, nn, app, uart, time

# 加载模型
detector = nn.YOLO11(
    model="/root/myproject/my_mud.mud",
    dual_buff=True  # 双缓冲优化
)

# 初始化摄像头
cam = camera.Camera(1920, 1080, detector.input_format())
disp = display.Display()

# 初始化串口
serial = uart.UART("/dev/ttyS0", 115200)

def process_detection(objs):
    """处理检测结果"""
    for obj in objs:
        # 获取 OBB 顶点
        points = obj.get_obb_points()
        
        # 计算中心点
        x = sum(points[::2]) / 4
        y = sum(points[1::2]) / 4
        
        # 获取角度
        angle = obj.angle * 180
        
        # 获取类别
        class_id = obj.class_id
        class_name = detector.labels[class_id]
        
        # 发送数据
        data = f"{x:.0f},{y:.0f},{obj.w:.0f},{obj.h:.0f},{angle:.0f},{class_id},1,1\n"
        serial.write(data.encode())
        
        # 绘制结果
        detector.draw_pose(img, points, 8, image.COLOR_RED)

# 主循环
while not app.need_exit():
    img = cam.read()
    
    # ROI 裁剪 (只检测中间区域)
    img_roi = img.crop(200, 130, 640, 640)
    
    # 推理
    objs = detector.detect(img_roi, conf_th=0.6, iou_th=0.55)
    
    # 处理结果
    if objs:
        process_detection(objs)
    
    # 显示
    disp.show(img_roi)
    
    time.sleep_ms(20)
```

---

## ⚡ 性能优化

### 1. 双缓冲 (Dual Buffer)

```python
detector = nn.YOLO11(model="my_mud.mud", dual_buff=True)
```

双缓冲可以在 NPU 推理的同时准备下一帧数据，提高吞吐量。

### 2. ROI 裁剪

```python
# 只处理感兴趣区域，减少计算量
img_roi = img.crop(200, 130, 640, 640)
objs = detector.detect(img_roi, ...)
```

### 3. 动态阈值

```python
# 根据场景动态调整阈值
conf_th = 0.6 if bright else 0.4
objs = detector.detect(img, conf_th=conf_th)
```

### 4. 帧率控制

```python
# 控制推理频率，避免过载
if frame_count % skip_frames == 0:
    objs = detector.detect(img)
```

---

## 📊 性能参考

在 MaixCam (SG2002) 上的性能：

| 模型 | 输入尺寸 | 量化 | FPS | mAP@0.5 |
|------|----------|------|-----|---------|
| yolo11n-obb | 640×640 | INT8 | ~35 | 0.82 |
| yolo11s-obb | 640×640 | INT8 | ~25 | 0.88 |
| yolo11s-obb | 640×640 | BF16 | ~18 | 0.91 |

---

## 🔧 常见问题

### 1. 量化后精度下降

**解决方案：**
- 使用更多校准数据 (200+ 张)
- 选择代表性图片 (覆盖各种场景)
- 使用敏感层分析，某些层保持 FP32
- 尝试 BF16 量化

### 2. 推理速度慢

**解决方案：**
- 减小输入尺寸 (640→480)
- 使用更小的模型 (s→n)
- 启用双缓冲
- ROI 裁剪

### 3. 模型加载失败

**解决方案：**
- 检查 MUD 文件路径和格式
- 确认模型版本与固件匹配
- 检查模型文件是否损坏

---

## 📚 参考资源

- [MaixPy 文档](https://wiki.sipeed.com/maixpy/)
- [Sophgo 工具链文档](https://developer.sophgo.com/)
- [ONNX Runtime](https://onnxruntime.ai/)
