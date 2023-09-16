import torch
import tqdm
from torch.utils.data import DataLoader
from utils.config import DefaultConfig
from models.net_builder import net_builder
from dataprepare.dataloader import DatasetCFP
from torch.nn import functional as F

import csv

def val(val_dataloader, model, args, Normal_average=0.0,device=None):

    print('\n')
    model.eval()
    tbar = tqdm.tqdm(val_dataloader, desc='\r')
    All_Infor = []

    with torch.no_grad():
        for i, img_data_list in enumerate(tbar):
            Fundus_img = img_data_list[0].to(device)
            image_files = img_data_list[2]
            pred = model.forward(Fundus_img)
            evidences = [F.softplus(pred)]

            alpha = dict()
            alpha[0] = evidences[0] + 1

            S = torch.sum(alpha[0], dim=1, keepdim=True)
            E = alpha[0] - 1
            b = E / (S.expand(E.shape))

            u = args.num_classes / S

            batch = b.shape[0]

            pred_con = pred.argmax(dim=-1)



            for idx_bs_u in range(batch):

                if u[idx_bs_u] >= Normal_average:
                    print("")

                    All_Infor.append([
                        '/'.join(image_files[idx_bs_u].split('/')[-2:]),
                        pred_con.cpu().detach().float().numpy()[idx_bs_u],
                        "Unreliable"
                    ]
                    )
                else:
                    All_Infor.append([
                        '/'.join(image_files[idx_bs_u].split('/')[-2:]),
                        pred_con.cpu().detach().float().numpy()[idx_bs_u],
                        "Reliable"
                    ]
                    )


    return All_Infor


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


    Thres = 0.1158
    Results_Heads = ["Imagefiles", 'Prediction results','Reliability']



    args.root = "./Datasets/OOD_Test"
    csv_file = "./Datasets/Pred_test.csv"
    test_loader = DataLoader(DatasetCFP(
        root=args.root,
        mode='test',
        data_file="Datasets/{}.csv".format(csv_file),
    ),
        batch_size=args.batch_size, shuffle=False, pin_memory=True)

    Results_Contents = val(test_loader, model, args,
        Normal_average=Thres, device=device)
    with open(
            "PredictionResults/Results.csv", 'w',
            newline='') as f:
        writer = csv.writer(f)
        writer.writerow(Results_Heads)
        writer.writerows(
            Results_Contents
        )


if __name__ == '__main__':
    args = DefaultConfig()

    main(args=args)