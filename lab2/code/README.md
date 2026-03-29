# Homework 2: Screw Counting

本项目用于自动检测并统计工业场景俯视图中的 5 种目标螺丝数量，核心方法是：
1. 采用 YOLOv8n 深度学习目标检测模型进行识别。
2. 通过 Albumentations 结合自建的旋转扩充脚本进行几何数据增强以提升旋转不变性。
3. 利用 3-Fold 交叉验证策略训练模型，以应对小样本和类别不平衡问题，并从中选取测试表现最优的模型进行推理。
4. 基于给定的测试路径自动进行批量推断，并记录处理时间与识别数量。

## 目录说明

- `run.py`：主程序执行接口，满足自动化阅卷标准。
- `train_kfold.py`：使用 3-Fold 算法进行 YOLOv8 训练的脚本。
- `augment_data.py`：将原始训练图像做 90/180/270 度旋转以扩充数据集的脚本。
- `best.pt`：最终选定并用于推理的模型权重。
- `yolov8n.pt`：YOLOv8 原始预训练权重（可选）。
- `requirements.txt`：Python 依赖包列表。

## 环境要求

- Python 3.9 及以上。
- 安装依赖库：

```bash
pip install -r requirements.txt
```

*如果你需要在 Mac (M 系列芯片) 上自行训练，还需要确保 PyTorch 对 MPS 能够正常支持。如果要在 Linux 或 Windows 的 Nvidia GPU 训练或推理，请安装带 CUDA 支持的 PyTorch。*

## 运行方式 (自动化阅卷)

为了统一阅卷，只需在 `code/` 目录下执行如下命令行格式即可（权重文件已经通过相对路径安全注册）：

```bash
python run.py \
  --data_dir /path/to/test_images \
  --output_path ./result.npy \
  --output_time_path ./time.txt
```

## 核心参数说明

- `--data_dir`：**（必填）** 包含多张未知螺丝图像的测试文件夹路径。支持如 `.jpg`, `.jpeg`, `.png`, `.bmp` 等常见格式。
- `--output_path`：**（必填）** 保存计数结果的 `.npy` 字典文件路径。
- `--output_time_path`：**（必填）** 记录脚本总耗时的文本文件存放路径（以秒进行记录）。

## 输出内容

成功运行后会生成两个核心文件：

1. `result.npy`
   通过 `numpy.load(..., allow_pickle=True).item()` 加载后，获得一个字典：
   - **Key**：不含后缀的图像名。
   - **Value**：一个长度为 5 的列表，代表 `[Type_1, Type_2, Type_3, Type_4, Type_5]` 螺丝各自的数量。（如果未检测到将填 0）。

2. `time.txt`
   仅包含一行字符串内容，代表整个推理过程耗费的总时间（浮点数秒）。

控制台也会同步格式化打印所有处理细节和单张图像的列表字典解析情况。

## 关键实现与调优细节

1. **预处理与标注**：利用 Lab1 矫正图像通过 Label Studio 标注了初步的 YOLO 格式数据集。
2. **数据增强**：原图数量极少，为了克服旋转引起的目标检测混淆影响，执行脚本将其批量扩展为原来的 4 倍量。
3. **环境兼容与推断筛选**：推断仅提取 `conf=0.25` 以上的框，并将模型推理与硬件剥离，依赖 `ultralytics` 内部自主检测最优推断后端。

## 常见问题

1. `ModuleNotFoundError: No module named 'ultralytics'`
   - 你可能忘记了安装 `requirements.txt`。可以通过 `pip install ultralytics` 快速修复。
2. `找不到模型权重文件`
   - `run.py` 严格依赖同级目录下的 `best.pt`，请确保提交时打包了模型权重并未变更文件名。
3. 字典 Key 带上了扩展名？
   - 代码内逻辑已经切出 `os.path.splitext` 防止此类问题出现，确保遵守格式说明。
