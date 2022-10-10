# -*- coding: utf-8 -*-
'''
@Version : 0.1
@Author : Charles
@Time : 2022/10/7 15:35 
@File : test.py 
@Desc : 
'''
import torch
from PIL import Image
from torchvision import transforms

from vgg import vgg16


label_names = ['cat', 'dog']


@torch.no_grad()
def predict(img_path, model_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.4875, 0.4544, 0.4164], [0.2521, 0.2453, 0.2481])
    ])
    # 模型
    model = vgg16(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'])
    model.to(device)

    image = Image.open(img_path)
    image = test_transform(image)
    image = torch.unsqueeze(image, 0)
    image = image.to(device)

    logits, probs = model(image)
    _, preds = torch.max(probs, 1)
    pred_name = label_names[int(preds[0])]
    print(pred_name)
    return pred_name


if __name__ == '__main__':
    img_path = ''
    model_path = ''
    predict(img_path, model_path)