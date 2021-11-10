# -*- coding: utf-8 -*-
'''
* @Author       : jiangtao
* @Date         : 2021-10-28 16:40:03
* @Email        : jiangtaoo2333@163.com
* @LastEditTime : 2021-10-28 16:56:12
'''
import argparse

# Warning, if use this as a package, add sys.path.append('../common_utils/Facealign')

import cv2
import numpy as np
import torch
from collections import OrderedDict

gpu_id = 0
device = torch.device('cuda:{}'.format(gpu_id))

def enlargeRoi(box,w,h,scale=0.10):

    minx,miny,maxx,maxy = box
    roiw = maxx - minx
    roih = maxy - miny

    minx -= roiw*scale
    miny -= roih*scale

    maxx += roiw*scale
    maxy += roih*scale

    minx = max(0,minx)
    miny = max(0,miny)

    maxx = min(w-1,maxx)
    maxy = min(h-1,maxy)

    return [int(minx),int(miny),int(maxx),int(maxy)]

def get_args():
    parser = argparse.ArgumentParser("faceAlignment of 19 points")

    parser.add_argument("--image_size", type=int, default=128, help="The common width and height for all images")

    parser.add_argument("--modelPath", type=str, default="./models/multiScale_7_20210716.pkl")
    # parser.add_argument("--modelPath", type=str, default="./models/multiScale_all_20210508.pkl")

    parser.add_argument("--imgPath", type=str, default="D:/CCF-Gaze/CCF_Gaze/data/training_data/")

    parser.add_argument("--aim", type=str, default="all", choices=["eye", 'mouth', 'all'])

    parser.add_argument("--saveImage", type=bool, default=False)

    parser.add_argument("--savePts", type=bool, default=False)

    parser.add_argument("--calNme", type=bool, default=True)

    parser.add_argument("--modelName", type=str, default='multi_out_7_0911')

    args = parser.parse_args()

    return args

getted_args = get_args()

class faceAlignment():
    """ This is a custom engine for this training cycle """

    def __init__(self, modelPath, device='cuda', image_size=128):
        self.image_size = image_size
        self.device = device
        model = torch.load(modelPath)
        model.cuda()
        self.network = model
        self.network.eval()

    def __call__(self,image):

        if type(image) != np.ndarray:
            return ('the img is wrong, not right type')

        height, width = image.shape[:2]

        image = cv2.resize(image, (self.image_size, self.image_size))
        if(len(image.shape)==2):
            image = image[:,:,np.newaxis]
        image = image.transpose(2,0,1)
        image = image * 0.0039216
        image = image[np.newaxis]

        image = torch.from_numpy(image)
        image = image.type(torch.FloatTensor)
        data = image.to(self.device)
        with torch.no_grad():
            out = self.network(data)

        # print('a',out[0].cpu().detach().numpy())
        # print('b',out[1].cpu().detach().numpy())
        eyePoints = out[0].cpu().detach().numpy().reshape((11,2))
        mouthPoints = out[1].cpu().detach().numpy().reshape((8,2))

        # print(eyePoints)
        # print(mouthPoints)
        
        eyePoints[:,0] *= width
        eyePoints[:,1] *= height

        mouthPoints[:,0] *= width
        mouthPoints[:,1] *= height

        # print(eyePoints)
        # print(mouthPoints)
        
        return eyePoints,mouthPoints

    def get_input_face(self, image, rect):
        sx, sy, ex, ey = rect
        image_shape = image.shape
        if len(image_shape) == 3:
            h, w, c = image.shape
        else:
            h, w = image.shape
        faceh = ey - sy
        facew = ex - sx

        shortsize = min(faceh, facew)
        expendw =  facew - shortsize
        expendh =  faceh - shortsize

        if expendw == 0:
            sy += expendh
        else:
            sx = sx + (expendw / 2)
            ex = ex - (expendw / 2)
        # sx = sx - (expendw / 2)
        # ex = ex + (expendw / 2)
        # sy = sy - (expendh / 2)
        # ey = ey + (expendh / 2)

        sx = int(max(0, sx))
        sy = int(max(0, sy))
        ex = int(min(w - 1, ex))
        ey = int(min(h - 1, ey))

        if len(image_shape) == 3:
            return image[sy:ey, sx:ex, :], sx, sy, ex, ey
        else:
            return image[sy:ey, sx:ex], sx, sy, ex, ey

    def forward_with_rect(self, image, rect):

        # get face image
        face_image,sx,sy,ex,ey = self.get_input_face(image, rect)

        self.det_area = (sx,sy,ex,ey)

        # get points
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        eyePoints,mouthPoints = self.__call__(face_image)

        # relocation
        eyePoints[:,0] += sx
        eyePoints[:,1] += sy

        mouthPoints[:,0] += sx
        mouthPoints[:,1] += sy

        return eyePoints,mouthPoints

if __name__ == '__main__':

    face_Alignment = faceAlignment("./models/multiScale_7_20210716.pkl")

    # 读取图片以及坐标框
    imgFile = './images/test/a.jpg'
    box = [351,518,620,779]

    #image = cv2.imread(imgFile,0)
    image_bgr = cv2.imread(imgFile)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image = image[:,:,np.newaxis]
    h,w = image.shape[0:2]
    sx, sy, ex, ey = box
    box = enlargeRoi(box,w,h)

    minx,miny,maxx,maxy = box
    img = image[int(miny):int(maxy),int(minx):int(maxx)]

    # 将处理好的图片输入网络
    eye,mouth = face_Alignment(img)

    # 还原到原图
    eye += [minx,miny]
    mouth += [minx,miny]

    # 两个点进行合并
    pts_pre = np.concatenate((eye,mouth),axis=0)

    # 原图画出
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(image_bgr,(int(minx),int(miny)),(int(maxx),int(maxy)),(0,125,0))
    cv2.rectangle(image_bgr, (int(sx), int(sy)), (int(ex), int(ey)), (0, 0, 255))
    for i in range (pts_pre.shape[0]):
        cv2.circle(image_bgr,(int(pts_pre[i][0]),int(pts_pre[i][1])),3,(255,0,0),-1)
        cv2.putText(image_bgr,str(i),(int(pts_pre[i][0]),int(pts_pre[i][1])), font, 0.4, (255, 255, 255), 1)

    # 展示
    cv2.imshow('a',image_bgr)
    cv2.waitKey(0)