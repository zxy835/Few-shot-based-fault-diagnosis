import torch
import os

def adjust_learning_rate(optimizer, epoch, initial_lr, step_epoch=20, factor=0.5):
    """
    训练过程中每step_epoch个epoch后，学习率乘以factor（默认0.5）
    """
    lr = initial_lr * (factor ** (epoch // step_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_best_model(train_acc, best_acc, model_fusion, model_ts, optimizer, epoch, save_dir):
    """
    如果训练准确率比best_acc更好，则保存模型
    返回最新的best_acc
    """
    if train_acc > best_acc:
        best_acc = train_acc
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_fusion_state_dict': model_fusion.state_dict(),
            'model_ts_state_dict': model_ts.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': train_acc
        }, os.path.join(save_dir, 'best_model_train_acc.pt'))
        print(f"模型已保存，训练准确率提升至: {train_acc:.4f}")
    return best_acc
