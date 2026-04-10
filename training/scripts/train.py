"""
YOLO11-OBB 垃圾分类训练脚本
使用 Ultralytics YOLO 框架训练有向边界框检测模型
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 解决 OpenMP 冲突

from ultralytics import YOLO


def train_model():
    """
    训练 YOLO11-OBB 模型
    """
    # 加载预训练的 YOLO11-OBB 模型
    # 可选: yolo11n-obb.pt (nano), yolo11s-obb.pt (small), yolo11m-obb.pt (medium)
    model = YOLO(r"yolo11s-obb.pt")
    
    # 训练模型
    results = model.train(
        data=r"../configs/garbage.yaml",    # 数据集配置文件
        epochs=180,                          # 训练轮数
        imgsz=640,                           # 输入图片尺寸
        batch=0.98,                          # 批量大小 (0.98 表示使用 98% 显存)
        device=0,                            # GPU 设备号，-1 表示 CPU
        
        # 输出配置
        project='runs/train',                # 训练结果保存路径
        name='yolo11s_obb_garbage',          # 实验名称
        exist_ok=True,                       # 允许覆盖已有目录
        
        # 数据增强配置
        augment=True,                        # 启用数据增强
        rect=True,                           # 矩形训练 (适合非方形图像)
        workers=8,                           # 数据加载线程数
        
        # 几何变换
        translate=0.1,                       # 平移 (默认: 0.1)
        degrees=5.0,                         # 旋转角度 (默认: 0.0)
        
        # HSV 颜色空间增强
        hsv_h=0.02,                          # 色调 (默认: 0.015)
        hsv_s=0.5,                           # 饱和度 (默认: 0.7)
        hsv_v=0.4,                           # 亮度 (默认: 0.4)
        
        # 早停设置
        patience=100,                        # 100 轮无改善则停止
        
        # 混合精度训练 (加速)
        amp=True,
    )
    
    print(f"训练完成! 结果保存在: {results.save_dir}")
    return results


def export_model(model_path):
    """
    导出模型为 ONNX 格式
    """
    # 加载训练好的模型
    model = YOLO(model_path)
    
    # 导出 ONNX
    model.export(
        format='onnx',
        imgsz=640,
        simplify=True,      # 使用 onnxsim 简化模型
        opset=12,           # ONNX opset 版本
    )
    print(f"模型已导出到 ONNX 格式")


if __name__ == "__main__":
    # 训练模型
    results = train_model()
    
    # 导出模型 (可选)
    # best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    # export_model(best_model_path)
