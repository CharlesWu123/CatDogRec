# -*- coding: utf-8 -*-
'''
@Version : 0.1
@Author : Charles
@Time : 2022/10/7 15:35 
@File : test.py 
@Desc : 
'''
import os

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from vgg import vgg16


label_names = ['Cat', 'Dog']


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
    cls = int(preds[0])
    pred_name = label_names[cls]
    return pred_name, round(float(probs[0][cls]), 2)


def plot_one_box_PIL(img, color=None, label=None, line_thickness=None):
    w, h = img.size
    draw = ImageDraw.Draw(img)
    line_thickness = line_thickness or max(int(min(img.size) / 200), 2)
    fontsize = max(round(max(img.size) / 20), 12)
    font = ImageFont.truetype("./utils/simfang.ttf", fontsize)
    txt_width, txt_height = font.getsize(label)
    if 4 - txt_height < 0:
        draw.rectangle([0, 0, txt_width, txt_height - 4], fill=tuple(color))
        draw.text((0, 0 - 3), label, fill=(255, 255, 255), font=font)
    else:
        draw.rectangle([0, 4-txt_height, txt_width, 0], fill=tuple(color))
        draw.text((0, 1 - txt_height), label, fill=(255, 255, 255), font=font)
    return img


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    img_path = './data/cat.jpg'
    model_path = './output/2022-10-10 14:24:58/model/best.pth'
    pred_name, prob = predict(img_path, model_path)
    print(pred_name, prob)
    # 画出结果
    draw_img = plot_one_box_PIL(Image.open(img_path), color=(0, 0, 255), label=f'{pred_name} {prob}')
    draw_img.save('./data/res.jpg')
