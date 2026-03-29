import os
import glob
import shutil
from sklearn.model_selection import KFold
from ultralytics import YOLO

def main():
    # 1. 配置路径
    data_dir = "datasets/all_screws" # 你的总数据目录（里面有 images 和 labels）
    images = np.array(sorted(glob.glob(f"{data_dir}/images/*.png") + glob.glob(f"{data_dir}/images/*.jpg")))
    
    # 2. 初始化 3 折交叉验证
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(images)):
        print(f"\n{'='*20} 正在训练 Fold {fold + 1}/3 {'='*20}")
        
        # 创建当前 Fold 的数据集目录
        fold_dir = f"datasets/fold_{fold}"
        os.makedirs(f"{fold_dir}/images/train", exist_ok=True)
        os.makedirs(f"{fold_dir}/images/val", exist_ok=True)
        os.makedirs(f"{fold_dir}/labels/train", exist_ok=True)
        os.makedirs(f"{fold_dir}/labels/val", exist_ok=True)
        
        # 分发文件
        for i, img_path in enumerate(images):
            split = "train" if i in train_idx else "val"
            img_name = os.path.basename(img_path)
            label_name = img_name.rsplit('.', 1)[0] + '.txt'
            label_path = f"{data_dir}/labels/{label_name}"
            
            shutil.copy(img_path, f"{fold_dir}/images/{split}/{img_name}")
            if os.path.exists(label_path):
                shutil.copy(label_path, f"{fold_dir}/labels/{split}/{label_name}")
                
        # 3. 创建当前 fold 的 yaml 文件
        yaml_path = f"{fold_dir}/dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(f"path: {os.path.abspath(fold_dir)}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write("names:\n  0: Type_1\n  1: Type_2\n  2: Type_3\n  3: Type_4\n  4: Type_5\n")
            
        # 4. 训练当前 Fold 模型（核心参数优化）
        model = YOLO('yolov8n.pt') 
        model.train(
            data=yaml_path,
            epochs=150,
            imgsz=1024,          
            device='mps',        # Mac M芯片。如果是Intel填 'cpu'
            batch=4,
            project="screw_cv",
            name=f"fold_{fold}",
        
            degrees=90.0,        # 保留旋转，因为螺丝角度不同
            flipud=0.5,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4
        )

if __name__ == "__main__":
    import numpy as np # 放到最上面
    main()