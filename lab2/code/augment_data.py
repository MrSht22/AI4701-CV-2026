import os
import cv2
import glob

def rotate_yolo_bbox(bbox, angle):
    """旋转YOLO格式的边界框 (cx, cy, w, h)"""
    cx, cy, w, h = bbox
    if angle == 90:
        return [cy, 1.0 - cx, h, w]
    elif angle == 180:
        return [1.0 - cx, 1.0 - cy, w, h]
    elif angle == 270:
        return [1.0 - cy, cx, h, w]
    return bbox

def augment_dataset(data_dir):
    img_dir = os.path.join(data_dir, 'images')
    lbl_dir = os.path.join(data_dir, 'labels')
    
    # 支持的旋转角度
    angles = [90, 180, 270]
    
    # 找到所有图片
    image_paths = glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg"))
    
    print(f"找到 {len(image_paths)} 张原始图片，开始进行几何扩充...")
    
    for img_path in image_paths:
        if '_rot' in img_path: continue # 避免重复扩充
        
        img = cv2.imread(img_path)
        base_name = os.path.basename(img_path)
        name, ext = os.path.splitext(base_name)
        txt_path = os.path.join(lbl_dir, f"{name}.txt")
        
        # 读取原始标签
        labels = []
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = parts[0]
                        bbox = [float(x) for x in parts[1:]]
                        labels.append((cls_id, bbox))
        
        # 旋转 90, 180, 270 度
        for angle in angles:
            # 旋转图片
            if angle == 90:
                rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated_img = cv2.rotate(img, cv2.ROTATE_180)
            elif angle == 270:
                rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
            new_img_name = f"{name}_rot{angle}{ext}"
            new_txt_name = f"{name}_rot{angle}.txt"
            
            # 保存旋转后的图片
            cv2.imwrite(os.path.join(img_dir, new_img_name), rotated_img)
            
            # 计算并保存旋转后的标签
            with open(os.path.join(lbl_dir, new_txt_name), 'w') as f:
                for cls_id, bbox in labels:
                    new_bbox = rotate_yolo_bbox(bbox, angle)
                    # 确保坐标在 0-1 之间，防止精度问题越界
                    new_bbox = [max(0.0001, min(0.9999, x)) for x in new_bbox]
                    f.write(f"{cls_id} {new_bbox[0]:.6f} {new_bbox[1]:.6f} {new_bbox[2]:.6f} {new_bbox[3]:.6f}\n")
                    
    print("扩充完成！现在你拥有了 4 倍的数据量。")

if __name__ == "__main__":
    # 替换为你实际存放原始 9张图+txt 的路径
    augment_dataset("datasets/all_screws")