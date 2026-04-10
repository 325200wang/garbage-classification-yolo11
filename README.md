# 🗑️ 智能垃圾分类系统 - YOLO11-OBB

基于 **YOLO11-OBB** (有向边界框) 的嵌入式智能垃圾分类系统，支持单/多垃圾检测、目标追踪、实时分拣控制。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-8A2BE2.svg)](https://ultralytics.com)
[![Platform](https://img.shields.io/badge/Platform-MaixPy-green.svg)](https://maixpy.sipeed.com/)

**[中文文档](./README.md)** | **[技术详解](./docs/TECHNICAL_DETAILS.md)** | **[训练指南](./docs/TRAINING_GUIDE.md)** | **[部署指南](./docs/DEPLOYMENT.md)**

---

## 📖 项目简介

本项目是一个完整的**边缘AI垃圾分类解决方案**，从模型训练到嵌入式部署的全流程实现：

- 🎯 **YOLO11-OBB**：使用有向边界框检测倾斜放置的垃圾，输出精确角度
- 🏃 **ByteTrack**：多目标追踪算法，支持多垃圾场景持续跟踪
- 🧠 **后处理优化**：Soft-NMS、IOU过滤、优先级排序
- 🔧 **完整训练流程**：数据准备 → 模型训练 → 量化部署
- 🔌 **硬件通信**：UART串口与执行机构实时交互

---

## 🎬 演示

```
┌─────────────────────────────────────────┐
│  检测画面                                  │
│                                         │
│      ┌──────┐                           │
│     / 可乐瓶 /  ← kehuishou: 0.92       │
│    /      /     angle: 15°              │
│   └──────┘                              │
│        ┌────────┐                       │
│       /  电池   /  ← youhai: 0.88       │
│      /        /   angle: 90°            │
│     └────────┘                          │
│                                         │
│  UART → "320,240,100,50,15,0,2,1\n"      │
└─────────────────────────────────────────┘
```

---

## 🏗️ 系统架构

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Dataset   │───▶│   YOLO11    │───▶│   ONNX      │
│  (Images +  │    │   Training  │    │   Export    │
│   Labels)   │    │             │    │             │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                              │
                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   CVIModel  │◀───│   INT8      │◀───│   MLIR      │
│   (.cvimodel)    │   Quantize  │    │   Convert   │
└──────┬──────┘    └─────────────┘    └─────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│                Embedded Device                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  Camera  │─▶│  NPU     │─▶│  Post-processing │ │
│  │          │  │  Inference   │  (Soft-NMS +     │ │
│  └──────────┘  └──────────┘  │   ByteTrack)     │ │
│                               └────────┬─────────┘ │
│                                        │           │
│                               ┌────────▼─────────┐ │
│                               │  UART Output     │ │
│                               │  (x,y,w,h,angle) │ │
│                               └──────────────────┘ │
└─────────────────────────────────────────────────────┘
```

---

## 📂 项目结构

```
garbage-classification-yolo11/
├── 📁 training/                    # 训练相关
│   ├── configs/
│   │   ├── garbage.yaml           # 数据集配置
│   │   └── yolo11n-obb.yaml       # 模型架构配置
│   └── scripts/
│       └── train.py               # 训练脚本
│
├── 📁 src/                         # 推理代码
│   ├── obb_detection.py           # OBB基础检测
│   ├── single_garbage.py          # 单垃圾处理
│   ├── multi_garbage_softnms.py   # 多垃圾+Soft-NMS
│   └── multi_garbage_tracking.py  # 完整版(追踪+分拣)
│
├── 📁 deployment/                  # 部署相关
│   ├── quantization/              # 量化工作目录
│   └── yolo_mud_config.yaml       # MUD配置文件模板
│
├── 📁 docs/                        # 文档
│   ├── TECHNICAL_DETAILS.md       # 详细技术文档(2.3万字)
│   ├── YOLO_PRINCIPLES.md         # YOLO算法原理
│   ├── INTERVIEW_PREP.md          # 面试准备指南
│   ├── TRAINING_GUIDE.md          # 训练指南
│   └── DEPLOYMENT.md              # 部署指南
│
├── 📁 models/                      # 模型文件 (大文件，不上传Git)
├── 📁 assets/                      # 图片资源
│
├── README.md                       # 本文件
├── requirements.txt                # Python依赖
├── .gitignore                      # Git忽略配置
└── MISSING_CONTENT.md              # 缺失内容清单
```

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/你的用户名/garbage-classification-yolo11.git
cd garbage-classification-yolo11
```

### 2. 安装依赖

```bash
# 创建虚拟环境
conda create -n yolo python=3.10
conda activate yolo

# 安装依赖
pip install -r requirements.txt
```

### 3. 准备数据集

```bash
# 数据集结构
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### 4. 训练模型

```bash
cd training/scripts
python train.py
```

或使用命令行：
```bash
yolo obb train data=../configs/garbage.yaml model=yolo11s-obb.pt epochs=180 imgsz=640
```

### 5. 导出 ONNX

```bash
yolo obb export model=runs/train/yolo11s_obb_garbage/weights/best.pt format=onnx simplify=True
```

### 6. 嵌入式部署

参考 [部署指南](./docs/DEPLOYMENT.md) 进行模型量化和设备部署。

---

## 📊 检测类别

| 大类 | 英文 | 小类示例 |
|------|------|----------|
| 有害垃圾 | youhai | 电池、灯泡、药品、油漆桶 |
| 可回收物 | kehuishou | 可乐瓶、纸张、塑料瓶、易拉罐 |
| 厨余垃圾 | chuyu | 苹果、香蕉、菜叶、剩饭 |
| 其他垃圾 | qita | 烟头、尘土、陶瓷、卫生纸 |

---

## 🔧 核心功能

### 1. 有向边界框检测 (OBB)

```python
# 获取OBB的8个顶点
points = obj.get_obb_points()  # [x1,y1,x2,y2,x3,y3,x4,y4]

# 计算中心点
center_x = sum(points[::2]) / 4
center_y = sum(points[1::2]) / 4

# 获取角度
angle = obj.angle * 180  # 度
```

### 2. ByteTrack 多目标追踪

```python
from maix import tracker

object_tracker = tracker.ByteTracker(
    max_lost_buff_num=15,   # 最大丢失帧数
    track_thresh=0.4,       # 追踪置信度阈值
    high_thresh=0.6,        # 高分阈值
    match_thresh=0.5        # 匹配IoU阈值
)

tracks = object_tracker.update(detection_objects)
```

### 3. Soft-NMS 后处理

```python
def soft_nms(boxes, scores, sigma=0.5):
    """Soft-NMS: 降低重叠框置信度而非直接删除"""
    for i in range(N):
        for j in range(i+1, N):
            overlap = iou(boxes[i], boxes[j])
            weight = np.exp(-(overlap ** 2) / sigma)
            scores[j] *= weight
```

---

## 📈 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 输入分辨率 | 640×640 | 正方形输入 |
| 训练轮数 | 180 epochs | 早停 patience=100 |
| 检测类别 | 4大类 | 国标垃圾分类 |
| mAP@0.5 | > 85% | 验证集精度 |
| 角度精度 | ±5° | 旋转角度误差 |
| 推理速度 | ~25 FPS | yolo11s-obb INT8 @ SG2002 |

---

## 📚 文档目录

- **[TECHNICAL_DETAILS.md](./docs/TECHNICAL_DETAILS.md)** - 2.3万字技术详解，包含：
  - 项目架构设计
  - YOLO11-OBB 原理详解
  - ByteTrack 算法解析
  - 后处理优化策略
  - 嵌入式部署方案
  - 关键代码解析

- **[YOLO_PRINCIPLES.md](./docs/YOLO_PRINCIPLES.md)** - YOLO系列算法演进：
  - YOLOv1 到 YOLO11 发展历程
  - 网络结构对比
  - 损失函数详解
  - 面试常见问题

- **[INTERVIEW_PREP.md](./docs/INTERVIEW_PREP.md)** - 面试准备指南：
  - 项目介绍话术
  - 技术问题详解
  - 常见面试题回答

- **[TRAINING_GUIDE.md](./docs/TRAINING_GUIDE.md)** - 训练指南：
  - 环境配置
  - 数据集准备
  - 训练流程
  - 参数调优

- **[DEPLOYMENT.md](./docs/DEPLOYMENT.md)** - 部署指南：
  - 模型转换
  - INT8 量化
  - 嵌入式推理
  - 性能优化

---

## ⚠️ 已知问题与改进

1. **光照敏感**：在不同光照条件下检测效果差异较大
   - 解决：增加光照变化的数据增强

2. **遮挡处理**：严重遮挡时可能出现ID切换
   - 解决：优化 ByteTrack 参数或使用 ReID

3. **小目标检测**：对远距离小垃圾检测效果有待提升
   - 解决：增加小目标检测层或使用更高分辨率

---

## 🔮 未来计划

- [x] 添加训练代码和配置文件
- [x] 添加模型转换和量化说明
- [ ] 添加评估脚本和可视化工具
- [ ] 支持更多硬件平台 (Jetson、RK3588等)
- [ ] 添加 Web 可视化界面
- [ ] 支持动态类别扩展

---

## 🛠️ 开发环境

- **训练环境**
  - Python 3.10
  - PyTorch 2.0+
  - Ultralytics 8.0+
  - CUDA 11.8

- **部署环境**
  - MaixPy (Python 3.11)
  - Sophgo SG2002 / CV181x
  - UART 串口通信

---

## 📄 许可证

MIT License - 详见 [LICENSE](./LICENSE)

---

## 👨‍💻 作者

- **王昱程**
- GitHub: [@你的用户名](https://github.com/你的用户名)

如果这个项目对你有帮助，请给个 ⭐ Star！

---

## 🙏 致谢

- [Ultralytics](https://ultralytics.com/) - YOLO 框架
- [Sipeed](https://sipeed.com/) - MaixPy 嵌入式平台
- [ByteTrack](https://github.com/ifzhang/ByteTrack) - 目标追踪算法
