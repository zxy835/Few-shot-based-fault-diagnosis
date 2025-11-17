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
import torch.nn.functional as F

def compute_residual_distillation_loss(student_residuals, teacher_residuals):
    loss = 0.0
    for s, t in zip(student_residuals, teacher_residuals):
        s = torch.nn.functional.adaptive_avg_pool1d(s, t.size(-1))
        loss += torch.nn.functional.mse_loss(s, t.detach())
    return loss / len(student_residuals)

def student_evaluate(model_fusion, model_ts, dataloader, device):
    model_fusion.eval()
    model_ts.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_raw, labels in dataloader:
            x_raw = x_raw.to(device)
            labels = labels.to(device).long()

            x_early = x_raw[:, :, :30]
            x_early = x_early.permute(0, 2, 1)

            early_groups = [x_early[:, i * 6:(i + 1) * 6, :] for i in range(5)]

            student_feats = []
            for i in range(5):
                s_feat,_ = model_ts.forward_student(early_groups[i])
                student_feats.append(s_feat)

            query_feat = student_feats[0].squeeze(1)
            support_feats = [f.squeeze(1) for f in student_feats[1:]]

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


def train_student(model_fusion, model_ts, model_ts1, train_loader, test_loader, device, epochs=100, lr=1e-4, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)

    labels_np = np.load(path1)[:, 0, -1].astype(np.int64) - 1
    class_sample_count = np.bincount(labels_np)
    num_classes = len(class_sample_count)

    class_weights = 1.0 / torch.tensor(class_sample_count, dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = class_weights.to(device)

    model_fusion.to(device)
    model_ts.to(device)

    criterion_cls = torch.nn.CrossEntropyLoss(weight=class_weights)
    criterion_kd_mse = torch.nn.MSELoss()
    criterion_kd_kl = torch.nn.KLDivLoss(reduction="batchmean")

    optimizer = torch.optim.Adam([
        {'params': model_ts.parameters(), 'lr': lr},
        {'params': model_fusion.parameters(), 'lr': lr}
    ])

    best_acc = 0.0
    history = []

    for epoch in range(epochs):
        model_ts.train()
        model_fusion.train()

        if epoch < 300:
            for param in model_fusion.parameters():
                param.requires_grad = False
        else:
            for param in model_fusion.parameters():
                param.requires_grad = True

        current_lr = adjust_learning_rate(optimizer, epoch, lr, step_epoch=10, factor=0.95)

        total_loss_epoch = 0.0
        all_preds, all_labels = [], []

        for x_raw, labels in train_loader:
            x_raw = x_raw.to(device)
            labels = labels.to(device).long()

            x_early = x_raw[:, :, :30].permute(0, 2, 1)
            x_severe = x_raw[:, :, 30:60].permute(0, 2, 1)

            early_groups = [x_early[:, i * 6:(i + 1) * 6, :] for i in range(5)]
            severe_groups = [x_severe[:, i * 6:(i + 1) * 6, :] for i in range(5)]

            optimizer.zero_grad()

            teacher_feats = []
            student_feats = []
            student_residuals = []
            teacher_residuals = []

            for i in range(5):
                with torch.no_grad():
                    t_feat, t_res = model_ts1.forward_teacher(early_groups[i], severe_groups[i])
                    teacher_feats.append(t_feat)
                s_feat, s_res = model_ts.forward_student(early_groups[i])
                student_feats.append(s_feat)
                teacher_residuals.extend(t_res)
                student_residuals.extend(s_res)

            query_feat = student_feats[0].squeeze(1)
            support_feats = [f.squeeze(1) for f in student_feats[1:]]

            query_feat1 = teacher_feats[0].squeeze(1)
            support_feats1 = [f.squeeze(1) for f in teacher_feats[1:]]

            distances1 = model_fusion(query_feat, *support_feats)
            distances2 = model_fusion(query_feat1, *support_feats1)

            logits = -distances1
            logits_teacher = -distances2
            preds = torch.argmax(logits, dim=1)

            loss_cls = criterion_cls(logits, labels)

            student_feats_tensor = torch.stack([f.squeeze(1) for f in student_feats], dim=0).permute(1, 0, 2)
            teacher_feats_tensor = torch.stack([f.squeeze(1) for f in teacher_feats], dim=0).permute(1, 0, 2)
            loss_kd_mse = criterion_kd_mse(student_feats_tensor, teacher_feats_tensor)

            T = 2.0
            log_s = F.log_softmax(logits / T, dim=1)
            soft_t = F.softmax(logits_teacher / T, dim=1)
            loss_kd_kl_all = criterion_kd_kl(log_s, soft_t) * (T * T)

            # ✅ 残差loss计算（对齐每组特征）
            loss_res = compute_residual_distillation_loss(student_residuals, teacher_residuals)

            # ✅ 总损失函数整合
            total_loss = loss_cls + 1.0 * loss_kd_mse + 2.0 * loss_kd_kl_all + 1.5 * loss_res

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model_ts.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss_epoch += total_loss.item()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        avg_train_loss = total_loss_epoch / len(train_loader)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        train_acc = accuracy_score(all_labels, all_preds)

        val_acc, val_report, val_cm = student_evaluate(model_fusion, model_ts, test_loader, device)
        avg_acc = (val_acc + train_acc) / 2

        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save({
                'model_fusion_state_dict': model_fusion.state_dict(),
                'model_ts_state_dict': model_ts.state_dict(),
            }, os.path.join(save_dir, 'student_best_model.pth'))

        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        })

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    print("训练完成，最优验证准确率：{:.4f}".format(best_acc))

    with open(os.path.join(save_dir, 'student_training_log.csv'), mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'train_acc', 'val_acc'])
        writer.writeheader()
        writer.writerows(history)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path1 = r"SDTW/incipient_data.npy"
    path2 = r"SDTW/severe_data.npy"
    path3 = r"checkpoints/teacher_best_model.pth"

    batch_size = 32
    epochs = 300
    lr = 0.0002

    train_loader = get_traindataloader(path1, path2, batch_size=batch_size)
    test_loader = get_testdataloader(path1, path2, batch_size=batch_size)

    model_ts = TeacherStudentMSCNN(in_channels=6, base_channels=64, embed_dim=512)
    model_ts1 = TeacherStudentMSCNN(in_channels=6, base_channels=64, embed_dim=512)
    model_fusion = CrossAttentionGateFusion(embed_dim=512, num_heads=4)

    checkpoint = torch.load(path3, map_location=device)
    model_ts1.load_state_dict(checkpoint['model_ts_state_dict'])
    model_fusion.load_state_dict(checkpoint['model_fusion_state_dict'])

    model_ts1.to(device)
    model_fusion.to(device)

    model_ts1.eval()
    for param in model_ts1.parameters():
        param.requires_grad = False  # 仍然冻结教师

    model_fusion.train()

    train_student(model_fusion, model_ts, model_ts1, train_loader, test_loader, device, epochs, lr)


