# --coding:utf-8--
from torch.utils.data import Dataset
from torchvision import transforms as T 
from PIL import Image
import os
from itertools import islice

class DatasetCFP(Dataset):
    def __init__(self,root,data_file,mode = 'train'):
        self.data_list = self.get_files(root,data_file=data_file)
        if mode == 'train':
            self.transforms= T.Compose([
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean = [0.485,0.456,0.406],
                            std = [0.299,0.224,0.225])
            ])
        else:

            self.transforms = T.Compose([
                T.ToTensor(),
                T.Normalize(mean = [0.485,0.456,0.406],
                            std = [0.299,0.224,0.225])
            ])

    def get_files(self,root, data_file):
        import csv
        csv_reader = csv.reader(open(data_file))
        img_list = []
        for line in islice(csv_reader, 1, None):
            img_list.append(
                [
                   os.path.join(root,line[0]),
                    int(line[1])
                ]
            )
        return img_list

    def __getitem__(self,index):
        image_file,label = self.data_list[index]
        img = Image.open(image_file).convert("RGB")
        img_tensor = self.transforms(img)

        return img_tensor, label

    def __len__(self):
        return len(self.data_list)

