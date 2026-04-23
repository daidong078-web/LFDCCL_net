import torch
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc
from torch.utils.data import DataLoader
from torchvision import transforms
import random

# 假设 model_x.py 和 dataloder_x.py 在同一个目录下
from model_x import TemporalDisentanglementModel_V2
from dataloder_x import PairedTimePointLoader, Normalize


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def worker_init_fn(worker_id):
    set_seed(42 + worker_id)


set_seed(42)


def calculate_metrics(all_labels, all_preds, all_probs):
    """
    根据真实标签、预测标签和预测概率计算混淆矩阵和各项指标。
    这是一个从训练脚本中复制过来的辅助函数。
    """
    cm = confusion_matrix(all_labels, all_preds)
    metrics = {}

    # 从混淆矩阵计算 ACC, SEN, SPE
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['acc'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        metrics['sen'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity (Recall)
        metrics['spe'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
    else:  # 处理只有一个类别被预测的特殊情况
        metrics['acc'] = np.mean(np.array(all_labels) == np.array(all_preds))
        metrics['sen'] = float('nan')
        metrics['spe'] = float('nan')
        print("警告: 混淆矩阵不是 2x2，SEN/SPE 可能不准确。")

    # 计算 AUC
    if len(np.unique(all_labels)) > 1:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        metrics['auc'] = auc(fpr, tpr)
    else:
        metrics['auc'] = float('nan')
        print("警告: 测试集中只存在一个类别，无法计算 AUC。")

    return metrics, cm


def evaluate_test_set():
    """
    主函数：加载每个折的最佳模型并在独立测试集上进行评估。
    同时统计：
      - E(|cos(fs, fc)|) (t1/t2 平均)：越低分离越强
      - E(cos(fs_t1, fs_t2))：越高跨时刻一致性越强
    """
    # --- 1. 配置 ---
    BATCH_SIZE = 4
    K_FOLDS = 5  # 与训练时保持一致
    BASE_DATA_PATH = "/root/autodl-tmp/mci_change_code/wm/"
    #CHECKPOINT_DIR = "/root/autodl-tmp/mci_change_code/R1_hope/checkpoints_5fold_cl_0.0.5/"
    #CHECKPOINT_DIR = "/root/autodl-tmp/mci_change_code/hope/loss_beifen/densenet_checkpoints_5fold_4/"
    #CHECKPOINT_DIR = "/root/autodl-tmp/mci_change_code/R1_hope/ablution_checkpoints/t_0.01/"
    CHECKPOINT_DIR = "/root/autodl-tmp/mci_change_code/R1_hope/ablution_checkpoints/batch_10/"
    CLASS_MAP = {0: 'smci', 1: 'pmci'}

    # --- 2. 设备与数据加载 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用的设备: {device}")

    composed_transform = transforms.Compose([
        Normalize(),
    ])

    test_roots = {
        os.path.join(BASE_DATA_PATH, "testwms"): 0,
        os.path.join(BASE_DATA_PATH, "testwmp"): 1
    }
    test_dataset = PairedTimePointLoader(
        roots=test_roots,
        transform=composed_transform,
        file_extension=".nii"
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    print(f"独立测试集加载完成，共 {len(test_dataset)} 个样本。")

    # 用于存储每个模型在测试集上的性能指标
    all_models_test_results = []

    # --- 3. 循环加载并评估每个折的最佳模型 ---
    for fold in range(1, K_FOLDS + 1):
        print(f"\n{'=' * 25}")
        print(f"正在评估第 {fold} 折的最佳模型...")
        print(f"{'=' * 25}")

        model_path = os.path.join(CHECKPOINT_DIR, f'fold_{fold}_best_ce_loss_model.pth')
        if not os.path.exists(model_path):
            print(f"警告: 找不到模型文件 {model_path}，跳过此折。")
            continue

        # 初始化模型架构 (必须与训练时完全一致)
        in_channels = 1
        num_classes = 2
        latent_dim = 64
        input_shape = (169, 205, 169)
        model = TemporalDisentanglementModel_V2(
            in_channels=in_channels,
            num_classes=num_classes,
            latent_dim=latent_dim,
            input_shape=input_shape
        )

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # --- 4. 在测试集上进行推理 ---
        all_labels = []
        all_preds = []
        all_probs = []

        # --- 新增：分离/一致性统计累加器 ---
        sum_abs_cos_sc_t1 = 0.0
        sum_abs_cos_sc_t2 = 0.0
        sum_cos_ss = 0.0
        n_feat = 0

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc=f"Fold {fold} 推理中"):
                images_t1 = batch['image_t1'].to(device)
                images_t2 = batch['image_t2'].to(device)
                labels = batch['label'].to(device)

                outputs = model(images_t1, images_t2)
                logits = outputs['logits']

                preds = torch.argmax(logits, 1)
                probs = torch.nn.functional.softmax(logits, dim=1)[:, 1]

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                # ===== 新增统计：E(|cos(fs, fc)|) & E(cos(fs1, fs2)) =====
                # 使用 normalize + 点积实现 cosine（更快更稳）
                fs1 = F.normalize(outputs["f_s_t1"], dim=1)
                fs2 = F.normalize(outputs["f_s_t2"], dim=1)
                fc1 = F.normalize(outputs["f_c_t1"], dim=1)
                fc2 = F.normalize(outputs["f_c_t2"], dim=1)

                abs_cos_sc_t1 = torch.abs(torch.sum(fs1 * fc1, dim=1))  # [b]
                abs_cos_sc_t2 = torch.abs(torch.sum(fs2 * fc2, dim=1))  # [b]
                cos_ss = torch.sum(fs1 * fs2, dim=1)                    # [b]

                bsz = fs1.size(0)
                sum_abs_cos_sc_t1 += abs_cos_sc_t1.sum().item()
                sum_abs_cos_sc_t2 += abs_cos_sc_t2.sum().item()
                sum_cos_ss += cos_ss.sum().item()
                n_feat += bsz
                # ==========================================================

        # --- 5. 计算并显示当前模型的性能 ---
        test_metrics, cm_test = calculate_metrics(all_labels, all_preds, all_probs)

        # --- 新增：统计均值 ---
        mean_abs_cos_sc = (sum_abs_cos_sc_t1 + sum_abs_cos_sc_t2) / (2.0 * n_feat + 1e-8)
        mean_cos_ss = sum_cos_ss / (n_feat + 1e-8)

        test_metrics["mean_abs_cos_sc"] = mean_abs_cos_sc
        test_metrics["mean_cos_ss"] = mean_cos_ss

        all_models_test_results.append(test_metrics)

        print(f"\n--- 第 {fold} 折模型在测试集上的性能 ---")
        print(f"ACC: {test_metrics['acc']:.4f}")
        print(f"SEN: {test_metrics['sen']:.4f}")
        print(f"SPE: {test_metrics['spe']:.4f}")
        print(f"AUC: {test_metrics.get('auc', float('nan')):.4f}")

        print(f"\n--- 表征统计 (Disentanglement / Consistency) ---")
        print(f"E(|cos(fs, fc)|): {mean_abs_cos_sc:.4f}  (lower = better separation)")
        print(f"E(cos(fs_t1, fs_t2)): {mean_cos_ss:.4f} (higher = more temporal consistency)")

        print("\n--- 测试集混淆矩阵 ---")
        print(f"         Predicted\n         {CLASS_MAP[0]:<8} {CLASS_MAP[1]:<8}")
        if cm_test.shape == (2, 2):
            print(f"Actual {CLASS_MAP[0]:<8} [[{cm_test[0, 0]:<8} {cm_test[0, 1]:<8}]]")
            print(f"       {CLASS_MAP[1]:<8} [[{cm_test[1, 0]:<8} {cm_test[1, 1]:<8}]]")
        else:
            print(cm_test)

    # --- 6. 计算并打印所有模型的平均性能 ---
    if not all_models_test_results:
        print("没有可评估的模型，程序退出。")
        return

    print(f"\n\n{'=' * 30}")
    print("所有模型的最终测试结果总结")
    print(f"{'=' * 30}")

    acc_list = [res['acc'] for res in all_models_test_results]
    sen_list = [res['sen'] for res in all_models_test_results]
    spe_list = [res['spe'] for res in all_models_test_results]
    auc_list = [res.get('auc', float('nan')) for res in all_models_test_results]

    abs_cos_sc_list = [res.get("mean_abs_cos_sc", float("nan")) for res in all_models_test_results]
    cos_ss_list = [res.get("mean_cos_ss", float("nan")) for res in all_models_test_results]

    avg_acc = np.nanmean(acc_list)
    std_acc = np.nanstd(acc_list)
    avg_sen = np.nanmean(sen_list)
    std_sen = np.nanstd(sen_list)
    avg_spe = np.nanmean(spe_list)
    std_spe = np.nanstd(spe_list)
    avg_auc = np.nanmean(auc_list)
    std_auc = np.nanstd(auc_list)

    avg_abs_cos_sc = np.nanmean(abs_cos_sc_list)
    std_abs_cos_sc = np.nanstd(abs_cos_sc_list)
    avg_cos_ss = np.nanmean(cos_ss_list)
    std_cos_ss = np.nanstd(cos_ss_list)

    print("在独立测试集上的平均性能 (Mean ± STD):")
    print(f"  - 平均 ACC: {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"  - 平均 SEN: {avg_sen:.4f} ± {std_sen:.4f}")
    print(f"  - 平均 SPE: {avg_spe:.4f} ± {std_spe:.4f}")
    print(f"  - 平均 AUC: {avg_auc:.4f} ± {std_auc:.4f}")
    print(f"  - E(|cos(fs, fc)|): {avg_abs_cos_sc:.4f} ± {std_abs_cos_sc:.4f}")
    print(f"  - E(cos(fs_t1, fs_t2)): {avg_cos_ss:.4f} ± {std_cos_ss:.4f}")


if __name__ == '__main__':
    evaluate_test_set()
