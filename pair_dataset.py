from torch.utils.data import Dataset, DataLoader
import csv
from PIL import Image
from torchvision import transforms
# from numpy import random
import random


class ImagePair_dataset(Dataset):
    def __init__(self, csv_path, input_size):
        self.csv_data = []
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for path_pair in reader:
                self.csv_data.append(path_pair)
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.csv_data)

    def random_select(self, path, item):
        rand=random.randint(1,3)
        num0= (int(item/rand))%4
        num1=(num0+rand)%4
        # print(f"{num0}|{num1}")
        # print(path0,path1)
        return path[num0],path[num1]

    def __getitem__(self, item):
        path = self.csv_data[item]
        path0, path1 = self.random_select(path, item)
        # path0 ,path1 =path[0:2]
        path0 = path0.replace("/nlp_group/wuxing/suzhenpeng/mae/webvid_imgs_new/frames_data", "/home/wuxing/suzhenpeng/mae/webvid_imgs/frames_data")
        path1 = path1.replace("/nlp_group/wuxing/suzhenpeng/mae/webvid_imgs_new/frames_data", "/home/wuxing/suzhenpeng/mae/webvid_imgs/frames_data")
        image0, image1 = Image.open(path0).convert("RGB"), Image.open(path1).convert("RGB")
        del path0,path1
        image0, image1 = self.transform_train(image0), self.transform_train(image1)
        return image0,image1