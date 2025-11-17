import torch
from sklearn.metrics import classification_report, accuracy_score
import os
import csv
from Embedding.model_mscnn import TeacherStudentMSCNN
from Relation.model_cross_attention import CrossAttentionGateFusion
from data_loader import get_traindataloader, get_testdataloader
import numpy as np
from utils import adjust_learning_rate, save_best_model
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.metrics import confusion_matrix


def teacher_evaluate(model_fusion, model_ts, dataloader, device):
    model_fusion.eval()
    model_ts.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_raw, labels in dataloader:
            x_raw = x_raw.to(device)
            labels = labels.to(device).long()

            x_early = x_raw[:, :, :30].permute(0, 2, 1)
            x_severe = x_raw[:, :, 30:60].permute(0, 2, 1)

            early_groups = [x_early[:, i * 6:(i + 1) * 6, :] for i in range(5)]
            severe_groups = [x_severe[:, i * 6:(i + 1) * 6, :] for i in range(5)]

            teacher_feats = []
            for i in range(5):
                t_feat, _ = model_ts.forward_teacher(early_groups[i], severe_groups[i])
                teacher_feats.append(t_feat)

            query_feat = teacher_feats[0].squeeze(1)
            support_feats = [f.squeeze(1) for f in teacher_feats[1:]]

            distances = model_fusion(query_feat, *support_feats)
            logits = -distances
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    return acc, report, cm

def train_teacher(model_fusion, model_ts, train_loader, test_loader, device, epochs=30, lr=1e-4, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    labels_np = np.load(path1)[:, 0, -1].astype(np.int64) - 1
    class_sample_count = np.bincount(labels_np)
    num_classes = len(class_sample_count)

    class_weights = 1.0 / torch.tensor(class_sample_count, dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = class_weights.to(device)

    model_fusion.to(device)
    model_ts.to(device)
    model_fusion.train()
    model_ts.train()

    criterion_cls = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        list(model_fusion.parameters()) + list(model_ts.parameters()),
        lr=lr
    )

    history = []
    best_acc = 0.0

    for epoch in range(epochs):
        model_fusion.train()
        model_ts.train()
        current_lr = adjust_learning_rate(optimizer, epoch, lr, step_epoch=10, factor=0.9)
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for x_raw, labels in train_loader:
            x_raw = x_raw.to(device)
            labels = labels.to(device).long()

            x_early = x_raw[:, :, :30].permute(0, 2, 1)
            x_severe = x_raw[:, :, 30:60].permute(0, 2, 1)

            early_groups = [x_early[:, i * 6:(i + 1) * 6, :] for i in range(5)]
            severe_groups = [x_severe[:, i * 6:(i + 1) * 6, :] for i in range(5)]

            optimizer.zero_grad()

            teacher_feats = []

            for i in range(5):
                t_feat, t_res = model_ts.forward_teacher(early_groups[i], severe_groups[i])
                teacher_feats.append(t_feat)

            query_feat = teacher_feats[0].squeeze(1)
            support_feats = [f.squeeze(1) for f in teacher_feats[1:]]

            distances = model_fusion(query_feat, *support_feats)
            logits = -distances

            loss_cls = criterion_cls(logits, labels)

            total_loss = loss_cls

            total_loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            running_loss += total_loss.item()

        avg_train_loss = running_loss / len(train_loader)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        train_acc = accuracy_score(all_labels, all_preds)

        val_acc, val_report, val_cm = teacher_evaluate(model_fusion, model_ts, test_loader, device)

        avg_acc = (val_acc + train_acc) / 2
        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save({
                'epoch': epoch + 1,
                'model_fusion_state_dict': model_fusion.state_dict(),
                'model_ts_state_dict': model_ts.state_dict(),
            }, os.path.join(save_dir, 'teacher_best_model.pth'))

        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        })

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        print("Validation Classification Report:")
        print(classification_report(all_labels, all_preds, digits=4, zero_division=0))
        print("Validation Confusion Matrix:")
        print(val_cm)

    with open(os.path.join(save_dir, 'teacher_training_log.csv'), mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'train_acc', 'val_acc'])
        writer.writeheader()
        writer.writerows(history)

    print("训练完成，最优验证准确率：{:.4f}".format(best_acc))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path1 = r"SDTW/incipient_data.npy"
    path2 = r"SDTW/severe_data.npy"
    batch_size = 32
    epochs = 20
    lr = 1e-4

    train_loader = get_traindataloader(path1, path2, batch_size=batch_size)
    test_loader = get_testdataloader(path1, path2, batch_size=batch_size)

    model_ts = TeacherStudentMSCNN(in_channels=6, base_channels=64, embed_dim=512)
    model_fusion = CrossAttentionGateFusion(embed_dim=512, num_heads=4)

    train_teacher(model_fusion, model_ts, train_loader, test_loader, device, epochs, lr)

