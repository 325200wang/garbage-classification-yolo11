# 训练指南

本文档介绍如何使用 Ultralytics YOLO 训练垃圾分类 OBB 模型。

---

## 📋 环境准备

### 1. 安装依赖

```bash
# 创建虚拟环境
conda create -n yolo python=3.10
conda activate yolo

# 安装 PyTorch (根据 CUDA 版本选择)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 Ultralytics
pip install ultralytics

# 安装其他依赖
pip install onnx onnxsim opencv-python
```

### 2. 验证安装

```bash
yolo check
```

---

## 📁 数据集准备

### 数据集结构

```
dataset/
├── images/
│   ├── train/          # 训练集图片
│   ├── val/            # 验证集图片
│   └── test/           # 测试集图片 (可选)
└── labels/
    ├── train/          # 训练集标注
    ├── val/            # 验证集标注
    └── test/           # 测试集标注 (可选)
```

### OBB 标注格式

YOLO OBB 格式使用归一化的 8 点坐标：

```
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

示例 (`1.txt`)：
```
0 0.45 0.32 0.52 0.30 0.53 0.38 0.46 0.40
1 0.60 0.70 0.75 0.65 0.78 0.80 0.63 0.85
```

标注工具推荐：
- [Roboflow](https://roboflow.com/) - 在线标注，支持 OBB
- [LabelImg](https://github.com/tzutalin/labelImg) - 桌面工具
- [CVAT](https://cvat.org/) - 开源标注平台

---

## 🚀 训练模型

### 1. 配置文件检查

确保 `training/configs/garbage.yaml` 中的路径正确：

```yaml
train: ./dataset/images/train
val: ./dataset/images/val
nc: 4
names: ['youhai', 'kehuishou', 'chuyu', 'qita']
```

### 2. 开始训练

```bash
cd training/scripts
python train.py
```

或使用命令行：

```bash
yolo obb train \
  data=garbage.yaml \
  model=yolo11s-obb.pt \
  epochs=180 \
  imgsz=640 \
  batch=16 \
  device=0 \
  project=runs/train \
  name=yolo11s_obb_garbage
```

### 3. 训练参数说明

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `epochs` | 训练轮数 | 100-300 |
| `imgsz` | 输入尺寸 | 640 |
| `batch` | 批量大小 | 根据显存调整 (8-32) |
| `device` | 设备 | 0 (GPU), -1 (CPU) |
| `workers` | 数据加载线程 | 8 |
| `patience` | 早停耐心值 | 50-100 |
| `translate` | 平移增强 | 0.1 |
| `degrees` | 旋转增强 | 5.0 |

---

## 📊 训练监控

### 使用 TensorBoard

```bash
tensorboard --logdir runs/train
```

### 关键指标

- **mAP50**: IoU=0.5 时的平均精度 (目标: > 0.85)
- **mAP50-95**: IoU 从 0.5 到 0.95 的平均精度
- **Precision**: 精确率
- **Recall**: 召回率
- **Box Loss**: 边界框回归损失
- **Cls Loss**: 分类损失

---

## 💾 导出模型

### 导出 ONNX

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/train/yolo11s_obb_garbage/weights/best.pt')

# 导出 ONNX
model.export(
    format='onnx',
    imgsz=640,
    simplify=True,
    opset=12
)
```

### 支持的导出格式

```python
# 导出不同格式
model.export(format='onnx')      # ONNX
model.export(format='engine')    # TensorRT
model.export(format='tflite')    # TensorFlow Lite
model.export(format='openvino')  # OpenVINO
```

---

## 🔧 模型选择建议

| 模型 | 参数量 | FLOPs | 速度 | 适用场景 |
|------|--------|-------|------|----------|
| yolo11n-obb | 2.7M | 6.9G | 最快 | 边缘设备，低延迟 |
| yolo11s-obb | 9.7M | 22.7G | 快 | 平衡速度和精度 |
| yolo11m-obb | 21M | 72.2G | 中等 | 高精度需求 |
| yolo11l-obb | 26M | 91.3G | 慢 | 服务器端 |

对于嵌入式部署，建议使用 `yolo11s-obb` 或 `yolo11n-obb`。

---

## 🐛 常见问题

### 1. CUDA 内存不足

```bash
# 减小 batch size
batch=8  # 或更小

# 或减小输入尺寸
imgsz=480
```

### 2. 训练不收敛

- 检查标注格式是否正确
- 降低学习率
- 增加数据增强
- 检查数据集是否有问题

### 3. 过拟合

```bash
# 增加正则化
yolo obb train data=garbage.yaml model=yolo11s-obb.pt dropout=0.1 weight_decay=0.0005

# 增加数据增强
yolo obb train data=garbage.yaml model=yolo11s-obb.pt mosaic=1.0 mixup=0.1
```

---

## 📚 参考资源

- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [YOLO OBB Docs](https://docs.ultralytics.com/tasks/obb/)
- [Training Tips](https://docs.ultralytics.com/guides/model-training-tips/)
