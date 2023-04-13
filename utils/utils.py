import torch
import shutil
import os.path as osp

def adjust_learning_rate(opt, optimizer, epoch):
    if opt.lr_mode == 'step':
        lr = opt.lr * (0.1 ** (epoch // opt.step))
    elif opt.lr_mode == 'poly':
        lr = opt.lr * (1 - epoch / opt.num_epochs) ** 0.9
    elif opt.lr_mode == 'normal':
        lr = opt.lr
    else:
        raise ValueError('Unknown lr mode {}'.format(opt.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
def save_checkpoint(state,best_pred,best_pred_Test, epoch,is_best,checkpoint_path,stage="val",filename='./checkpoint/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        print("Model Saving................")
        shutil.copyfile(filename, osp.join(checkpoint_path,'model_{}_{:03d}_{:.6f}_{:.6f}.pth.tar'.format(
            stage,(epoch + 1),best_pred,best_pred_Test)))

def save_checkpoint_epoch(state,pred_Auc,pred_ACC,test_Auc,test_ACC,epoch,is_best,checkpoint_path,stage="val",filename='./checkpoint/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        print("Model Saving................")
        shutil.copyfile(filename, osp.join(checkpoint_path,'model_{}_{:03d}_Val_{:.6f}_{:.6f}_Test_{:.6f}_{:.6f}.pth.tar'.format(
            stage,(epoch + 1),pred_Auc,pred_ACC,test_Auc,test_ACC)))


