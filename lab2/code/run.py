import argparse
import os
import time
import glob
import numpy as np
from ultralytics import YOLO

def build_argparser():
    parser = argparse.ArgumentParser(description="Lab2: Screw Counting")
    parser.add_argument("--data_dir", type=str, required=True, help="测试图像文件夹路径")
    parser.add_argument("--output_path", type=str, required=True, help="输出的 npy 文件路径")
    parser.add_argument("--output_time_path", type=str, required=True, help="输出的时间 txt 文件路径")
    return parser

def main():
    args = build_argparser().parse_args()
    
    # 1. 记录总时间开始
    start_time = time.time()
    
    # 2. 加载训练好的模型
    # 确保 best.pt 与 run.py 在同一目录下，或修改为相对路径
    model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型权重文件: {model_path}")
    
    model = YOLO(model_path)
    
    # 3. 获取所有待测图片
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(os.path.join(args.data_dir, f"*{ext}")))
    
    out_dict = {}
    
    # 4. 遍历图片进行推理
    for img_path in image_paths:
        # 获取不带后缀的文件名
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # 记录 5 种螺丝的数量，初始化为 [0, 0, 0, 0, 0]
        # index 0->Type1, 1->Type2, 2->Type3, 3->Type4, 4->Type5
        counts = [0, 0, 0, 0, 0]
        
        # 使用 YOLO 进行预测
        # conf=0.25 意味着只统计置信度大于 25% 的检测框，你可以根据验证集调整
        results = model.predict(source=img_path, conf=0.25, verbose=False)
        
        # 5. 统计各类别的数量
        if len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                classes = boxes.cls.cpu().numpy().astype(int)
                for cls_id in classes:
                    if 0 <= cls_id < 5:
                        counts[cls_id] += 1
                        
        out_dict[base_name] = counts
        print(f"Processed {base_name}: {counts}")

    # 6. 保存字典到 .npy
    np.save(args.output_path, out_dict)
    
    # 7. 记录总时间结束并保存
    end_time = time.time()
    total_time = end_time - start_time
    
    with open(args.output_time_path, 'w', encoding='utf-8') as f:
        f.write(str(total_time))
        
    print("-" * 40)
    print(f"处理完成！共处理 {len(image_paths)} 张图片。")
    print(f"总耗时: {total_time:.4f} 秒")
    print(f"已保存结果至: {args.output_path}")
    print(f"已保存时间至: {args.output_time_path}")

if __name__ == '__main__':
    main()