from __future__ import annotations

import os
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 环境变量设置
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
matplotlib.use("Agg")  # 无图形界面下避免报错

import torch
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry

def pick_device() -> torch.device:
    """自动选择最优计算设备 (CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if os.environ.get("SAM_USE_MPS", "0") == "1" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def show_masks(masks: list[np.ndarray], ax: plt.Axes) -> None:
    """将 boolean 类型的掩码数组绘制到图像上，每个实例分配随机颜色"""
    if len(masks) == 0:
        return
    
    # 按照面积从大到小排序，防止大掩码遮挡小掩码
    masks = sorted(masks, key=lambda x: x.sum(), reverse=True)
    
    rng = np.random.default_rng(42)
    for m in masks:
        # m 是 (H, W) 的 boolean 数组
        color_mask = np.concatenate([rng.random(3), [0.65]]).astype(np.float32)
        h, w = m.shape
        img = np.zeros((h, w, 4), dtype=np.float32)
        img[m] = color_mask  # 将对应像素点上色
        ax.imshow(img)

def main() -> None:
    # 1. 路径与配置
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "outputs"
    
    yolo_model_path = base_dir / "best.pt"  # YOLO 权重
    sam_checkpoint = base_dir / "sam_vit_b_01ec64.pth"  # SAM 权重
    
    output_dir.mkdir(parents=True, exist_ok=True)

    if not yolo_model_path.exists():
        raise FileNotFoundError(f"找不到 YOLO 模型: {yolo_model_path}")
    if not sam_checkpoint.exists():
        raise FileNotFoundError(f"找不到 SAM 模型: {sam_checkpoint}")

    device = pick_device()
    print(f"当前使用的计算设备: {device}")

    # 2. 加载 YOLO 模型
    print("正在加载 YOLOv8 检测模型...")
    yolo_model = YOLO(str(yolo_model_path))

    # 3. 加载 SAM 模型，并使用 SamPredictor (提示词模式)
    print("正在加载 SAM 分割模型...")
    sam = sam_model_registry["vit_b"](checkpoint=str(sam_checkpoint))
    sam.to(device=device)
    predictor = SamPredictor(sam) # 注意：这里改成了 Predictor

    # 4. 获取图片列表
    image_files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])

    # 5. 处理流程：YOLO 找框 -> SAM 根据框抠图
    for img_name in image_files:
        print(f"\n正在处理图片: {img_name}...")
        img_path = data_dir / img_name
        
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # --- 步骤 A：YOLO 目标检测 ---
        # conf=0.25 过滤低置信度背景，可根据你的模型效果微调
        results = yolo_model.predict(
            source=str(img_path), 
            conf=0.15,       # 放宽及格线，宁可错杀一千不可放过一个
            iou=0.65,        # 允许框之间有较大的重叠（拯救图5）
            imgsz=1024,      # 使用更高的分辨率推理，防止小螺丝特征丢失
            augment=True,    # 开启测试时增强 (Test-Time Augmentation)
            verbose=False
        )
        # results = yolo_model.predict(source=str(img_path), conf=0.25, verbose=False)
        
        # 提取边界框坐标[x_min, y_min, x_max, y_max]
        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            print(f"-> {img_name} 中未检测到任何螺丝。")
            continue
            
        boxes_np = results[0].boxes.xyxy.cpu().numpy()
        
        # --- 步骤 B：SAM 提示词分割 ---
        # 预计算当前图像的图像嵌入 (Image Embedding)，只需计算一次
        predictor.set_image(image_rgb)
        
        final_masks =[]
        for box in boxes_np:
            masks, scores, logits = predictor.predict(
                box=box,
                multimask_output=False 
            )
            mask = masks[0] 
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # --- 亮度过滤 (保留) ---
            mean_brightness = cv2.mean(gray_image, mask=mask_uint8)[0]
            if mean_brightness > 230:
                continue

            # --- 核心优化：坚实度 (Solidity) 过滤 ---
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            largest_cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_cnt)
            if area == 0: continue # 避免除零错误
            
            hull = cv2.convexHull(largest_cnt)
            hull_area = cv2.contourArea(hull)
            
            solidity = float(area) / hull_area
            
            # 田字格很饱满，坚实度很高。螺丝是实心的，坚实度较低。
            if solidity > 0.92:
                print(f" -> 剔除中空物体 (田字框), 坚实度: {solidity:.2f}")
                continue
            
            final_masks.append(mask)
            
        print(f"-> 在 {img_name} 中成功分割了 {len(final_masks)} 个实例。")

        # --- 步骤 C：可视化并保存 ---
        plt.figure(figsize=(10, 10))
        plt.imshow(image_rgb)
        
        # 将 YOLO 的边界框也画出来（可选，为了让报告更丰富，默认注释掉，仅展示 Mask）
        # for box in boxes_np:
        #     rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], fill=False, edgecolor='red', linewidth=1.5)
        #     plt.gca().add_patch(rect)
            
        show_masks(final_masks, plt.gca())
        plt.axis("off")

        output_name = f"result_{img_name}"
        output_path = output_dir / output_name
        plt.savefig(str(output_path), bbox_inches="tight", pad_inches=0)
        plt.close()

if __name__ == "__main__":
    main()