import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from model_x import TemporalDisentanglementModel_V2, compute_losses_v2
from sympy import false
from tqdm import tqdm
import random
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc,f1_score
from sklearn.model_selection import KFold  
from dataloder_x import PairedTimePointLoader, Normalize
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import transforms
import matplotlib.pyplot as plt



def set_seed(seed):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def calculate_metrics(all_labels, all_preds, all_probs):
    """
    """
    cm = confusion_matrix(all_labels, all_preds)
    metrics = {}

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['acc'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        metrics['sen'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['spe'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        metrics['acc'] = np.mean(np.array(all_labels) == np.array(all_preds))
        metrics['sen'] = 0.0
        metrics['spe'] = 0.0
  

    if len(np.unique(all_labels)) > 1:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        metrics['auc'] = auc(fpr, tpr)
    else:
        metrics['auc'] = float('nan')


    return metrics, cm


def train_classifier():
    SEED = 42
    NUM_EPOCHS = 100
    LR = 0.0001
    BATCH_SIZE = 2
    K_FOLDS = 5  

    BASE_DATA_PATH = "/root/code/wm/"

    OUTPUT_DIR = "./checkpoints/"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    CLASS_MAP = {0: 'smci', 1: 'pmci'}
    #CLASS_MAP = {0: 'NC', 1: 'AD'}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"check {num_gpus} GPUs, it will use DataParallel for training.")
    else:
        print("just 1 or no GPU detected, it will use single device training.")
    set_seed(SEED)
    print(f" {SEED}")


    composed_transform = transforms.Compose([
        Normalize(),
    ])

    all_data_roots = {
        os.path.join(BASE_DATA_PATH, "smciwm"): 0,
        os.path.join(BASE_DATA_PATH, "pmciwm"): 1
    }

    full_dataset = PairedTimePointLoader(
        roots=all_data_roots,
        transform=composed_transform,
        file_extension=".nii"
    )
    print(f"cross validation dataset initialized successfully! Found a total of {len(full_dataset)} samples.")

    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)


    all_folds_results = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        print(f"\n{'=' * 25}")
        print(f"Starting Fold {fold + 1} / {K_FOLDS}")
        print(f"{'=' * 25}")


        train_sub_dataset = Subset(full_dataset, train_ids)
        val_sub_dataset = Subset(full_dataset, val_ids)

        train_dataloader = DataLoader(
            dataset=train_sub_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        val_dataloader = DataLoader(
            dataset=val_sub_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,  
            num_workers=8,
            pin_memory=True
        )
        print(f"Fold {fold + 1} - Train samples: {len(train_sub_dataset)}, Validation samples: {len(val_sub_dataset)}")

   
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
        if num_gpus > 1:
            model = nn.DataParallel(model)
        model.to(device)

        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.05)
        print(f"Fold {fold + 1} - Model, optimizer, and scheduler have been reinitialized.")


        history = {
            'train_loss': [], 'train_acc': [], 'train_auc': [], 'train_ce_loss': [],
            'val_loss': [], 'val_acc': [], 'val_auc': [], 'val_ce_loss': []
        }

    
        best_val_acc = {'epoch': -1, 'value': 0.0}
        best_val_auc = {'epoch': -1, 'value': 0.0}
        best_val_total_loss = {'epoch': -1, 'value': float('inf')}
        best_val_ce_loss = {'epoch': -1, 'value': float('inf')}

     
        for epoch in range(NUM_EPOCHS):
            print(f"\n--- Fold {fold + 1}, Epoch {epoch + 1}/{NUM_EPOCHS} ---")

            # --- training phase ---
            model.train()
            running_loss = 0.0
            running_ce_loss = 0.0
            all_train_labels, all_train_preds, all_train_probs = [], [], []

            progress_bar = tqdm(train_dataloader, desc=f"Fold {fold + 1} training")
            for batch in progress_bar:
                images_t1 = batch['image_t1'].to(device)
                images_t2 = batch['image_t2'].to(device)
                labels = batch['label'].to(device)

                optimizer.zero_grad()
                outputs = model(images_t1, images_t2)
                loss = compute_losses_v2(model, outputs, labels)
                loss['total_loss'].backward()
                optimizer.step()
                running_loss += loss['total_loss'].item() * images_t1.size(0)
                logits = outputs['logits']
                preds = torch.argmax(logits, 1)
                probs = torch.nn.functional.softmax(logits, dim=1)[:, 1]

                all_train_labels.extend(labels.cpu().numpy())
                all_train_preds.extend(preds.cpu().numpy())
                all_train_probs.extend(probs.detach().cpu().numpy())

                current_acc = np.mean(np.array(all_train_labels) == np.array(all_train_preds))
                progress_bar.set_postfix(loss=f"{loss['total_loss'].item():.4f}", acc=f"{current_acc:.4f}")

            epoch_loss = running_loss / len(train_sub_dataset)
            train_metrics, _ = calculate_metrics(all_train_labels, all_train_preds, all_train_probs)

            print(f"Training phase - Total Loss: {epoch_loss:.4f}, CE Loss: {epoch_ce_loss:.4f}, "
                  f"ACC: {train_metrics['acc']:.4f}, AUC: {train_metrics.get('auc', float('nan')):.4f}")

            # --- Validation phase ---
            model.eval()
            running_val_loss, running_val_ce_loss = 0.0, 0.0
            all_val_labels, all_val_preds, all_val_probs = [], [], []

            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc=f"Fold {fold + 1} validating"):
                    images_t1 = batch['image_t1'].to(device)
                    images_t2 = batch['image_t2'].to(device)
                    labels = batch['label'].to(device)
                    outputs = model(images_t1, images_t2)
                    loss = compute_losses_v2(model, outputs, labels)

                    running_val_loss += loss['total_loss'].item() * images_t1.size(0)
                    logits = outputs['logits']
                    preds = torch.argmax(logits, 1)
                    probs = torch.nn.functional.softmax(logits, dim=1)[:, 1]

                    all_val_labels.extend(labels.cpu().numpy())
                    all_val_preds.extend(preds.cpu().numpy())
                    all_val_probs.extend(probs.cpu().numpy())

            epoch_val_loss = running_val_loss / len(val_sub_dataset)
            val_metrics, cm_val = calculate_metrics(all_val_labels, all_val_preds, all_val_probs)

            print(f"Validation phase - Total Loss: {epoch_val_loss:.4f}, CE Loss: {epoch_val_ce_loss:.4f}, "
                  f"ACC: {val_metrics['acc']:.4f}, SEN: {val_metrics['sen']:.4f}, "
                  f"SPE: {val_metrics['spe']:.4f}, AUC: {val_metrics.get('auc', float('nan')):.4f}")

            print("--- Current epoch validation confusion matrix ---")
            print(f"         Predicted\n         {CLASS_MAP[0]:<8} {CLASS_MAP[1]:<8}")
            if cm_val.shape == (2, 2):
                print(f"Actual {CLASS_MAP[0]:<8} [[{cm_val[0, 0]:<8} {cm_val[0, 1]:<8}]]")
                print(f"       {CLASS_MAP[1]:<8} [[{cm_val[1, 0]:<8} {cm_val[1, 1]:<8}]]")
            else:
                print(cm_val)

    
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(train_metrics['acc'])
            history['train_auc'].append(train_metrics.get('auc', float('nan')))
            history['val_loss'].append(epoch_val_loss)
            history['val_acc'].append(val_metrics['acc'])
            history['val_auc'].append(val_metrics.get('auc', float('nan')))


            model_to_save = model.module if isinstance(model, nn.DataParallel) else model

            if epoch_val_loss < best_val_total_loss['value']:
                best_val_total_loss['value'] = epoch_val_loss
                best_val_total_loss['epoch'] = epoch + 1
                save_path = os.path.join(OUTPUT_DIR, f'fold_{fold + 1}_best_total_loss_model.pth')
                torch.save(model_to_save.state_dict(), save_path)
                print(f"Fold {fold + 1} new best Total Loss model saved: {epoch_val_loss:.4f} at epoch {epoch + 1}")

            scheduler.step()


        print(f"\nthe {fold + 1} fold training completed!")
        print(f"Lowest Total Loss for this fold: {best_val_total_loss['value']:.4f} (at epoch {best_val_total_loss['epoch']})")


        plt.figure(figsize=(8, 6))
        plt.suptitle(f'Fold {fold + 1} Training and Validation Metrics', fontsize=16)
        epochs_range = range(NUM_EPOCHS)

        plt.plot(epochs_range, history['train_loss'], label='Training Total Loss')
        plt.plot(epochs_range, history['val_loss'], label='Validation Total Loss')
        plt.legend(loc='upper right')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = os.path.join(OUTPUT_DIR, f'training_curves_fold_{fold + 1}.png')
        plt.savefig(plot_path)
        plt.close() 
        print(f"Fold {fold + 1} training curves saved to: {plot_path}")


    print(f"\n\n{'=' * 30}")
    print("All folds cross-validation training completed!")
    print(f"{'=' * 30}")


    avg_acc = np.mean([res['best_acc'] for res in all_folds_results])
    std_acc = np.std([res['best_acc'] for res in all_folds_results])
    avg_auc = np.mean([res['best_auc'] for res in all_folds_results])
    std_auc = np.std([res['best_auc'] for res in all_folds_results])

    print("\n--- the 5 fold cross-validation final results summary ---")
    for res in all_folds_results:
        print(f"Fold {res['fold']}: Best ACC={res['best_acc']:.4f}, Best AUC={res['best_auc']:.4f}")

    print("\n--- average performance ---")
    print(f"average ACC: {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"average AUC: {avg_auc:.4f} ± {std_auc:.4f}")




if __name__ == '__main__':
    train_classifier()