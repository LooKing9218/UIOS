import torch
import os
import tqdm
import numpy as np
import torch.nn as nn
from sklearn import metrics
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import utils.utils as u
from utils.config import DefaultConfig
from models.net_builder import net_builder
from dataprepare.dataloader import DatasetCFP
from torch.nn import functional as F

# loss function
def KL(alpha, c, device):
    beta = torch.ones((1, c)).to(device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step, device):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c, device)
    return (A + B)




def val(val_dataloader, model, epoch, args, mode, device):

    print('\n')
    print('====== Start {} ======!'.format(mode))
    model.eval()
    labels = []
    outputs = []

    predictions = []
    gts = []
    correct = 0.0
    num_total = 0
    tbar = tqdm.tqdm(val_dataloader, desc='\r')

    with torch.no_grad():
        for i, img_data_list in enumerate(tbar):
            Fundus_img = img_data_list[0].to(device)
            cls_label = img_data_list[1].long().to(device)
            pred = model.forward(Fundus_img)
            evidences = [F.softplus(pred)]

            alpha = dict()
            alpha[0] = evidences[0] + 1

            S = torch.sum(alpha[0], dim=1, keepdim=True)
            E = alpha[0] - 1
            b = E / (S.expand(E.shape))

            pred = torch.softmax(b,dim=1)

            data_bach = pred.size(0)
            num_total += data_bach
            one_hot = torch.zeros(data_bach, args.num_classes).to(device).scatter_(1, cls_label.unsqueeze(1), 1)
            pred_decision = pred.argmax(dim=-1)
            for idx in range(data_bach):
                outputs.append(pred.cpu().detach().float().numpy()[idx])
                labels.append(one_hot.cpu().detach().float().numpy()[idx])
                predictions.append(pred_decision.cpu().detach().float().numpy()[idx])
                gts.append(cls_label.cpu().detach().float().numpy()[idx])
    epoch_auc = metrics.roc_auc_score(labels, outputs)
    Acc = metrics.accuracy_score(gts, predictions)
    if not os.path.exists(os.path.join(args.save_model_path, "{}".format(args.net_work))):
        os.makedirs(os.path.join(args.save_model_path, "{}".format(args.net_work)))

    with open(os.path.join(args.save_model_path,"{}/{}_Metric.txt".format(args.net_work,args.net_work)),'a+') as Txt:
        Txt.write("Epoch {}: {} == Acc: {}, AUC: {}\n".format(
            epoch,mode, round(Acc,6),round(epoch_auc,6)
        ))
    print("Epoch {}: {} == Acc: {}, AUC: {}\n".format(
            epoch,mode,round(Acc,6),round(epoch_auc,6)
        ))
    torch.cuda.empty_cache()
    return epoch_auc,Acc
def train(train_loader, val_loader, test_loader, model, optimizer, criterion,writer,args,device):

    step = 0
    best_auc = 0.0
    best_auc_Test = 0.0
    for epoch in range(0,args.num_epochs+1):
        model.train()
        labels = []
        outputs = []
        tq = tqdm.tqdm(total=len(train_loader) * args.batch_size)
        tq.set_description('Epoch %d, lr %f' % (epoch, args.lr))
        loss_record = []
        train_loss = 0.0
        for i, img_data_list in enumerate(train_loader):
            Fundus_img = img_data_list[0].to(device)
            cls_label = img_data_list[1].long().to(device)
            optimizer.zero_grad()
            pretict = model(Fundus_img)
            evidences = [F.softplus(pretict)]
            loss_un = 0
            alpha = dict()
            alpha[0] = evidences[0] + 1

            S = torch.sum(alpha[0], dim=1, keepdim=True)
            E = alpha[0] - 1
            b = E / (S.expand(E.shape))

            Tem_Coef = epoch*(0.99/args.num_epochs)+0.01

            loss_CE = criterion(b/Tem_Coef, cls_label)


            loss_un += ce_loss(cls_label, alpha[0], args.num_classes, epoch, args.num_epochs, device)
            loss_ACE = torch.mean(loss_un)
            loss = loss_CE+loss_ACE
            loss.backward()
            optimizer.step()
            tq.update(args.batch_size)
            train_loss += loss.item()
            tq.set_postfix(loss='%.6f' % (train_loss / (i + 1)))
            step += 1
            one_hot = torch.zeros(pretict.size(0), args.num_classes).to(device).scatter_(1, cls_label.unsqueeze(1), 1)
            pretict = torch.softmax(pretict, dim=1)
            for idx_data in range(pretict.size(0)):
                outputs.append(pretict.cpu().detach().float().numpy()[idx_data])
                labels.append(one_hot.cpu().detach().float().numpy()[idx_data])

            if step%10==0:
                writer.add_scalar('Train/loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        torch.cuda.empty_cache()
        loss_train_mean = np.mean(loss_record)
        epoch_train_auc = metrics.roc_auc_score(labels, outputs)

        del labels,outputs

        writer.add_scalar('Train/loss_epoch', float(loss_train_mean),
                          epoch)
        writer.add_scalar('Train/train_auc', float(epoch_train_auc),
                          epoch)

        print('loss for train : {}, {}'.format(loss_train_mean,round(epoch_train_auc,6)))
        if epoch % args.validation_step == 0:
            if not os.path.exists(os.path.join(args.save_model_path, "{}".format(args.net_work))):
                os.makedirs(os.path.join(args.save_model_path, "{}".format(args.net_work)))
            with open(os.path.join(args.save_model_path, "{}/{}_Metric.txt".format(
                    args.net_work,args.net_work)), 'a+') as f:
                f.write('EPOCH:' + str(epoch) + ',')
            mean_AUC, mean_ACC = val(val_loader, model, epoch,args,mode="val",device=device)
            writer.add_scalar('Valid/Mean_val_AUC', mean_AUC, epoch)
            best_auc = max(best_auc, mean_AUC)
            checkpoint_dir = os.path.join(args.save_model_path)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            mean_AUC_Test, mean_ACC_Test = val(test_loader, model, epoch, args, mode="Test",device=device)
            writer.add_scalar('Test/Mean_Test_AUC', mean_AUC_Test, epoch)
            print('===> Saving models...')


            u.save_checkpoint_epoch({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'mean_AUC': mean_AUC,
                'mean_ACC': mean_ACC,
                'mean_AUC_Test': mean_AUC_Test,
                'mean_ACC_Test': mean_ACC_Test,
            }, mean_AUC, mean_ACC, mean_AUC_Test, mean_ACC_Test, epoch, True, checkpoint_dir, stage="Test",
                filename=os.path.join(checkpoint_dir,"checkpoint.pth.tar"))


def main(args=None,writer=None):
    train_loader = DataLoader(DatasetCFP(
        root=args.root,
        mode='train',
        data_file=args.train_file,
    ),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(DatasetCFP(
        root=args.root,
        mode='val',
        data_file=args.val_file,
    ),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(DatasetCFP(
        root=args.root,
        mode='test',
        data_file=args.test_file,
    ),
        batch_size=args.batch_size, shuffle=False, pin_memory=True)

    device = torch.device('cuda:{}'.format(args.cuda))

    model = net_builder(args.net_work, args.num_classes).to(device)
    print('Model have been loaded!,you chose the ' + args.net_work + '!')
    if args.trained_model_path:
        print("=> loading trained model '{}'".format(args.trained_model_path))
        checkpoint = torch.load(
            args.trained_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        print('Done!')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(device)
    train(train_loader, val_loader, test_loader, model, optimizer, criterion,writer,args,device)

if __name__ == '__main__':
    args = DefaultConfig()
    log_dir = args.log_dirs
    writer = SummaryWriter(log_dir=log_dir)
    args.save_model_path = args.save_model_path

    main(args=args, writer=writer)
