import tqdm
import torch
from torch.utils.data import DataLoader
from utils.config import DefaultConfig
from models.net_builder import net_builder
from dataprepare.dataloader import DatasetCFP
from torch.nn import functional as F
from sklearn import metrics



def val(val_dataloader, model, args, mode, device):

    print('\n')
    print('====== Start {} ======!'.format(mode))
    model.eval()


    u_list = []
    u_label_list = []

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

            u = args.num_classes / S

            un_gt = 1 - torch.eq(b.argmax(dim=-1), cls_label).float()


            data_bach = pred.size(0)
            for idx in range(data_bach):
                u_list.append(u.cpu()[idx].numpy())
                u_label_list.append(un_gt.cpu()[idx].numpy())

    return u_list, u_label_list


def main(args=None):
    args.net_work = "ResUnNet50"
    args.trained_model_path = './Trained/UIOS.pth.tar'
    # bulid model
    device = torch.device('cuda:{}'.format(args.cuda))
    args.device = device

    model = net_builder(args.net_work, args.num_classes).to(device)

    print('Model have been loaded!,you chose the ' + args.net_work + '!')
    # load trained model for test
    print("=> loading trained model '{}'".format(args.trained_model_path))
    checkpoint = torch.load(
        args.trained_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print('Done!')


    test_loader = DataLoader(DatasetCFP(
        root=args.root,
        mode='test',
        data_file=args.val_file,
    ),
        batch_size=args.batch_size, shuffle=False, pin_memory=True)

    u_list, u_label_list = val(test_loader, model, args, mode="Validation", device=device) 


    fpr_Pri, tpr_Pri, thresh = metrics.roc_curve(u_label_list, u_list)
    max_j = max(zip(fpr_Pri, tpr_Pri), key=lambda x: 2*x[1] - x[0])
    pred_thresh = thresh[list(zip(fpr_Pri, tpr_Pri)).index(max_j)]
    print("opt_pred ===== {}".format(pred_thresh))




if __name__ == '__main__':
    args = DefaultConfig()

    main(args=args)