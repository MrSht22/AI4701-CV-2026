# Homework 1: Homography Restore

本项目用于将透视畸变后的图像恢复到模板的俯视视角（top view），核心方法是：
1. 基于颜色阈值生成掩膜，聚焦可用于匹配的彩色区域。
2. 提取局部特征点并进行描述子匹配。
3. 使用 RANSAC 估计单应矩阵（Homography）。
4. 通过 `warpPerspective` 将输入图像配准到模板坐标系。

## 目录说明

- `restore_homography.py`：主程序。
- `requirements.txt`：Python 依赖列表。
- `../HW1_data/`：输入数据目录（包含模板和待恢复图像）。
- `../restored_images/`：输出目录（恢复结果和统计文件）。

## 环境要求

- Python 3.9 及以上。
- 安装依赖：

```bash
pip install -r requirements.txt
```

## 运行方式

在 `code/` 目录下执行：

```bash
python restore_homography.py \
  --data-dir ../HW1_data \
  --template ../HW1_data/template.png \
  --out-dir ../restored_images \
  --method auto \
  --ransac-thresh 5.0
```

## 参数说明

- `--data-dir`：输入图像目录，默认 `../HW1_data`。
- `--template`：模板图像路径，默认 `../HW1_data/template.png`。
- `--out-dir`：输出目录，默认 `../restored_images`。
- `--method`：特征方法，`auto|orb|akaze|sift`，默认 `auto`。
- `--ransac-thresh`：RANSAC 重投影阈值，默认 `5.0`。
- `--debug`：开启调试输出，会保存掩膜和匹配可视化到 `out-dir/_debug/`。

## 自动方法选择逻辑

当 `--method auto` 时，程序会依次尝试：

1. `sift`
2. `akaze`
3. `orb`

并以 RANSAC 内点数（inliers）作为评分，选择本张图像最优结果。

## 输出内容

在 `--out-dir` 下会生成：

- 恢复后的图像：文件名与输入一致。
- `stats.csv`：每张图的统计信息，字段如下：
  - `file`
  - `detector`
  - `inliers`
  - `total`
  - `ratio`（`inliers / total`）
- 控制台统计汇总：包括成功/失败数量与 detector 使用分布。
- 若开启 `--debug`，额外生成 `_debug/`：
  - `mask_template.png`
  - `mask_<image_name>`
  - `match_<detector>_<image_name>`

## 关键实现细节

- 颜色掩膜在 HSV 空间中阈值为：
  - 下界 `(0, 120, 50)`
  - 上界 `(180, 255, 255)`
- 特征匹配采用 KNN + ratio test（阈值 0.75）。
- 单应矩阵估计采用 `cv2.findHomography(..., cv2.RANSAC, ransac_thresh)`。

## 常见问题

1. 输出全是失败（`FAILED`）
   - 检查输入路径和模板路径是否正确。
   - 使用 `--debug` 查看掩膜是否覆盖到有效区域。
   - 尝试调整 `--ransac-thresh`（如 3.0 到 8.0）。

2. `sift` 不可用
   - 你的 OpenCV 版本可能未包含 SIFT，程序会自动跳过并尝试其他方法。

3. 配准结果仍有偏差
   - 检查模板与输入图像内容是否一致（遮挡、模糊、曝光变化会影响匹配）。
   - 可只使用某一方法对比：`--method orb` / `--method akaze` / `--method sift`。