import os
import glob
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import Dataset
import os
import glob
import random

import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Normalize 类保持不变，因为它设计得很好，可以复用
class Normalize:
    """
    一个普通的归一化转换类：
    1. 将可能存在的nan值替换为0。
    2. 将整个图像的像素值统一归一化到 [0, 1] 区间。
    """

    def __call__(self, sample):
        image_key = 'input_image'
        image_numpy = sample[image_key]
        cleaned_image = np.nan_to_num(image_numpy, nan=0.0)
        min_val = np.min(cleaned_image)
        max_val = np.max(cleaned_image)
        if max_val - min_val > 1e-6:
            normalized_image = (cleaned_image - min_val) / (max_val - min_val)
        else:
            normalized_image = np.zeros_like(cleaned_image, dtype=np.float32)
        sample[image_key] = normalized_image
        return sample


# ======================================================================
# 【优化版】Dataloader，用于加载成对的时间点数据
# ======================================================================
class PairedTimePointLoader(Dataset):
    """
    为时序模型设计的Dataloader。
    它会为每个被试加载一对按文件名排序的时间点影像（t1 和 t2）。
    """

    def __init__(self, roots, file_extension='.nii', transform=None):
        super().__init__()
        self.image_pair_list = []
        self.transform = transform

        # 遍历每个类别路径 (例如 AD, NC, MCI)
        for class_path, label in roots.items():
            if not os.path.exists(class_path):
                print(f"警告：目录不存在，已跳过: {class_path}")
                continue

            # 遍历每个被试的文件夹
            for sample_name in os.listdir(class_path):
                sample_dir = os.path.join(class_path, sample_name)
                if not os.path.isdir(sample_dir):
                    continue

                # 搜索被试文件夹内所有的 .nii 文件并排序
                search_pattern = os.path.join(sample_dir, '*' + file_extension)
                image_files = sorted(glob.glob(search_pattern))

                # 【核心优化】确保该被试至少有两个时间点的数据
                if len(image_files) >= 2:
                    # 将排序后的第一个和第二个文件作为 t1 和 t2
                    path_t1 = image_files[0]
                    path_t2 = image_files[1]
                    self.image_pair_list.append((path_t1, path_t2, label))
                else:
                    print(f"信息：被试 {sample_name} 的影像文件不足2个，已跳过。")

        if not self.image_pair_list:
            raise RuntimeError("错误：未找到任何拥有至少两个时间点的有效样本。")

        print(f"成功加载 {len(self.image_pair_list)} 个被试的影像对。")

    def __len__(self):
        return len(self.image_pair_list)

    def _load_nifti_image(self, path):
        """加载并预处理单个NIFTI图像"""
        nifti_img = nib.load(path)
        # 将数据转换为 float32 并增加通道维度
        img_data = nifti_img.get_fdata().astype(np.float32)
        return np.expand_dims(img_data, axis=0)

    def __getitem__(self, idx):
        path_t1, path_t2, label = self.image_pair_list[idx]

        # --- 处理 t1 时刻的图像 ---
        image_t1_numpy = self._load_nifti_image(path_t1)
        sample_t1 = {'input_image': image_t1_numpy}
        if self.transform:
            sample_t1 = self.transform(sample_t1)
        final_image_t1_numpy = sample_t1['input_image']

        # --- 处理 t2 时刻的图像 ---
        image_t2_numpy = self._load_nifti_image(path_t2)
        sample_t2 = {'input_image': image_t2_numpy}
        if self.transform:
            sample_t2 = self.transform(sample_t2)
        final_image_t2_numpy = sample_t2['input_image']

        # 返回一个包含t1和t2图像对的字典
        return {
            'image_t1': torch.from_numpy(final_image_t1_numpy),
            'image_t2': torch.from_numpy(final_image_t2_numpy),
            'label': torch.tensor(label, dtype=torch.long)
        }


# ======================================================================
# SEED = 42
# BASE_DATA_PATH = "/root/autodl-tmp/mci_change_code/wm/"
# OUTPUT_DIR = "./densenet_checkpoints"
# BATCH_SIZE = 4
# NUM_WORKERS = 4  # 根据您的机器性能调整
#
# # 创建输出目录
# os.makedirs(OUTPUT_DIR, exist_ok=True)
#
# CLASS_MAP = {0: 'pmci', 1: 'smci'}
#
# def set_seed(seed):
#     """固定随机种子以保证实验可复现"""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False


# if __name__ == "__main__":
#     # --- 初始化环境 ---
#     #set_seed(SEED)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"正在使用的设备: {device}")
#
#     # --- 定义数据转换 ---
#     composed_transform = transforms.Compose([
#         Normalize(),
#     ])
#
#     # --- 定义数据源路径 ---
#     # 路径格式: {类别文件夹路径: 标签}
#     # train_roots = {
#     #     os.path.join(BASE_DATA_PATH, "pmciwm"): 0,  # pmci 标签为 0
#     #     os.path.join(BASE_DATA_PATH, "smciwm"): 1  # smci 标签为 1
#     # }
#
#     # --- 初始化 Dataset 和 DataLoader ---
#     print("\n--- 正在初始化 PairedTimePointLoader ---")
#     # 注意：这里我们使用 PairedTimePointLoader 来匹配时序模型的需求
#     try:
#         train_dataset = PairedTimePointLoader(
#             roots=train_roots,
#             transform=composed_transform,
#             file_extension=".nii"  # 根据您的文件后缀修改，常见为 .nii 或 .nii.gz
#         )
#
#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=BATCH_SIZE,
#             shuffle=True,
#             num_workers=NUM_WORKERS,
#             pin_memory=True
#         )
#         print(f"\n训练集初始化成功！共找到 {len(train_dataset)} 个样本。")
#         print(f"DataLoader 初始化成功！将分为 {len(train_loader)} 个批次。")
#
#         # --- 提取并验证一个批次的数据 ---
#         print("\n--- 正在提取并验证第一个批次的数据 ---")
#         first_batch = next(iter(train_loader))
#
#         images_t1 = first_batch['image_t1']
#         images_t2 = first_batch['image_t2']
#         labels = first_batch['label']
#
#         print(f"批次中 t1 影像的维度: {images_t1.shape}")
#         print(f"批次中 t2 影像的维度: {images_t2.shape}")
#         print(f"批次中标签的维度:    {labels.shape}")
#
#         # 打印标签，并使用 CLASS_MAP 转换回类名
#         label_names = [CLASS_MAP[label.item()] for label in labels]
#         print(f"批次中的标签: {labels.tolist()} -> {label_names}")
#
#         # 验证数据归一化是否生效
#         print("\n--- 验证数据预处理 ---")
#         # 检查一个样本
#         sample_image_t1 = images_t1[0]
#         print(f"第一个 t1 样本的最小值 (应接近 0.0): {sample_image_t1.min().item():.4f}")
#         print(f"第一个 t1 样本的最大值 (应接近 1.0): {sample_image_t1.max().item():.4f}")
#         print(f"第一个 t1 样本的数据类型: {sample_image_t1.dtype}")
#
#         print("\n数据提取和验证成功！可以开始进行模型训练。")
#
#     except RuntimeError as e:
#         print(f"\n初始化过程中发生错误: {e}")
#         print("请检查您的 BASE_DATA_PATH 是否正确，以及该路径下是否存在符合格式的被试文件夹和影像文件。")