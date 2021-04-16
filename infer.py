# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import time
from io import BytesIO
import base64
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000000000
from tqdm import tqdm
import glob
import os
from scipy.io import loadmat
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
# from utils import colorEncode
import torch.nn as nn
from torch.cuda.amp import autocast
from utils.deeplearning import seg_model
from dataset import val_transform

def visualize_result(img_dir, pred):
    #
    img=cv2.imread(img_dir)
    colors = loadmat('demo/color150.mat')['colors']
    names = {
            1: "耕地",
            2: "林地",
            3: "草地",
            4: "道路",
            5: "城镇建设用地",
            6: "农村建设用地",
            7: "工业用地",
            8: "构筑物",
            9: "水域",
            10: "裸地"
        }
    # print predictions in descending order
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    #
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    #print(pred_color.shape)
    #pred_color=cv2.resize(pred_color,(256,256))
    im_vis = np.concatenate((img, pred_color), axis=1)

    #
    #img_name=image_demo_dir.split('/')[-1]
    save_dir,name=os.path.split(img_dir)
    Image.fromarray(im_vis).save('demo/256x256.png')


def inference(img_dir):
    transform=val_transform

    image = cv2.imread(img_dir, cv2.IMREAD_UNCHANGED)
    img = transform(image=image)['image']
    img=img.unsqueeze(0)
    #print(img.shape)
    with torch.no_grad():
        img=img.cuda()
        output = model(img)
    
    pred = output.squeeze().cpu().data.numpy()
    pred = np.argmax(pred,axis=0)
    return pred+1

if __name__=="__main__":
    model=seg_model().cuda()
    model= torch.nn.DataParallel(model)
    checkpoints=torch.load('../../user_data/model_data/SWA_checkpoint-best.pth')
    model.load_state_dict(checkpoints['state_dict'])
    model.eval()
    use_demo=0
    assert_list=[1,2]
    if use_demo:
        img_dir='../data_tmp/suichang_round1_test_partA_210120/000001.tif'
        pred=inference(img_dir)
        infer_start_time = time.time()
        visualize_result(img_dir, pred)
        #
    else:
        out_dir='../../prediction_result'
        if not os.path.exists(out_dir):os.makedirs(out_dir)
        test_paths=glob.glob('../../tcdata/suichang_round1_test_partB_210120/*')
        for per_path in tqdm(test_paths):
            result=inference(per_path)
            img=Image.fromarray(np.uint8(result))
            img=img.convert('L')
            #print(out_path)
            out_path=os.path.join(out_dir,per_path.split('/')[-1][:-4]+'.png')
            img.save(out_path)