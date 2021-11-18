import os
import random
import socket
import sys
import os.path as osp
from os.path import dirname as dirpath
sys.path.append(dirpath(dirpath(dirpath(os.path.abspath(__file__)))))

import cv2
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
# from src.network import RecognitionBone
from tqdm import tqdm
import shutil
from src.dataprocess.transform import randomAug_Onet_pts

def movedirbyext(dir,ext,dir2):

    fileList = getlist(dir,ext)

    for file in tqdm(fileList):

        try:
            shutil.move(file,dir2)
        except:
            print('already have')

def setUpTrainingBrachOnet(model,args):

    # 只固定基干网络，其他不管
    # if(isinstance(model,torch.nn.parallel.distributed.DistributedDataParallel)):

    #     model.module.train()
    #     model.module.baseBone.eval()
    #     for k,v in model.module.named_parameters():
    #         v.requires_grad = True
    #     for k,v in model.module.baseBone.named_parameters():
    #         v.requires_grad = False
    # else:

    #     model.train()
    #     model.baseBone.eval()
    #     for k,v in model.named_parameters():
    #         v.requires_grad = True
    #     for k,v in model.baseBone.named_parameters():
    #         v.requires_grad = False

    # for k,v in model.named_parameters():
    #     if v.requires_grad:
    #         print(k,' required grad')

    # 先统一设置成固定参数
    for k,v in model.named_parameters():
        v.requires_grad = False

    model.baseBone.eval()
    model.alignBone.eval()
    model.binaryFaceBone.eval()
    try:
        model.gazeBone.eval()
    except:
        pass

    for keyword in args.trainingBranch:
        for k,v in model.named_parameters():
            if(keyword in k):
                v.requires_grad = True

    if('base' in args.trainingBranch):
        print('model.baseBone.train()\n')
        model.baseBone.train()

    if('align' in args.trainingBranch):
        print('model.alignBone.train()\n')
        model.alignBone.train()

    if('binaryFace' in args.trainingBranch):
        print('model.binaryFaceBone.train()\n')
        model.binaryFaceBone.train()

    try:
        if('gaze' in args.trainingBranch):
            print('model.gazeBone.train()\n')
            model.gazeBone.train()
    except:
        pass

def setUpTrainingBrach(model,args):

    # 先统一设置成固定参数
    for k,v in model.named_parameters():
        v.requires_grad = False

    model.baseBone.eval()
    model.eyeBone.eval()
    model.mouthBone.eval()
    model.faceBone.eval()
    model.detectBone.eval()
    model.emotionBone.eval()
    model.FaceAreaBone.eval()

    try:
        model.binaryFaceBone.eval()
    except:
        pass

    try:
        model.gazeBone.eval()
    except:
        pass

    try:
        model.eyeBone_right.eval()
    except:
        pass

    try:
        model.mouthBone_right.eval()
    except:
        pass

    try:
        model.eyeBone_left.eval()
    except:
        pass

    try:
        model.mouthBone_left.eval()
    except:
        pass

    try:
        model.angleRegBone.eval()
    except:
        pass

    try:
        model.alignQualityBone.eval()
    except:
        pass

    try:
        model.WrinkleBone.eval()
    except:
        pass

    try:
        model.genderBone.eval()
    except:
        pass

    try:
        model.ageBone.eval()
    except:
        pass

    # 再单独设置各个需要训练的参数
    for keyword in args.trainingBranch:
        for k,v in model.named_parameters():
            if(keyword in k):
                v.requires_grad = True

    for k,v in model.named_parameters():
        if v.requires_grad:
            print(k+'required grad')

    if('base' in args.trainingBranch):
        print('model.baseBone.train()\n')
        model.baseBone.train()
    if('eye' in args.trainingBranch):
        print('model.eye.train()\n')
        model.eyeBone.train()
    if('mouth' in args.trainingBranch):
        print('model.mouth.train()\n')
        model.mouthBone.train()
    if('face' in args.trainingBranch):
        print('model.face.train()\n')
        model.faceBone.train()
    if('detect' in args.trainingBranch):
        print('model.detect.train()\n')
        model.detectBone.train()
    if('emotion' in args.trainingBranch):
        print('model.emotion.train()\n')
        model.emotionBone.train()
    if('FaceArea' in args.trainingBranch):
        print('model.FaceArea.train()\n')
        model.FaceAreaBone.train()


    try:
        if('binaryFace' in args.trainingBranch):
            print('model.binaryFaceBone.train()\n')
            model.binaryFaceBone.train()
    except:
        pass

    try:
        if('gaze' in args.trainingBranch):
            print('model.gazeBone.train()\n')
            model.gazeBone.train()
    except:
        pass

    try:
        if('eyeBone_right' in args.trainingBranch):
            print('model.eyeBone_right.train()\n')
            model.eyeBone_right.train()
    except:
        pass

    try:
        if('mouthBone_right' in args.trainingBranch):
            print('model.mouthBone_right.train()\n')
            model.mouthBone_right.train()
    except:
        pass

    try:
        if('eyeBone_left' in args.trainingBranch):
            print('model.eyeBone_left.train()\n')
            model.eyeBone_left.train()
    except:
        pass

    try:
        if('mouthBone_left' in args.trainingBranch):
            print('model.mouthBone_left.train()\n')
            model.mouthBone_left.train()
    except:
        pass

    try:
        if('angleReg' in args.trainingBranch):
            print('model.angleRegBone.train()\n')
            model.angleRegBone.train()
    except:
        pass

    try:
        if('quality' in args.trainingBranch):
            print('model.alignQualityBone.train()\n')
            model.alignQualityBone.train()
    except:
        pass

    try:
        if('WrinkleBone' in args.trainingBranch):
            print('model.WrinkleBone.train()\n')
            model.WrinkleBone.train()
    except:
        pass

    try:
        if('genderBone' in args.trainingBranch):
            print('model.genderBone.train()\n')
            model.genderBone.train()
    except:
        pass

    try:
        if('ageBone' in args.trainingBranch):
            print('model.ageBone.train()\n')
            model.ageBone.train()
    except:
        pass

def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum

    softmax = softmax.sum(axis=-2)
    return softmax

def getlist(dir,extension,Random=False):
    list = []
    for root, dirs, files in os.walk(dir, topdown=False):
        for name in dirs:
            print(os.path.join(root, name))
        for name in files:
            filename,ext = os.path.splitext(name)
            if extension == ext:
                list.append(os.path.join(root,name))
                list[-1]  = list[-1].replace('\\','/')
    if Random:
        random.shuffle(list)
    return list

def reproductive():
    seed=0

    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Remove randomness (may be slower on Tesla GPUs) 
    # https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print('seed is settled')

def isInuse(ipList, port):
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  flag=True
  for ip in ipList:
    try:
      s.connect((ip, int(port)))
      s.shutdown(2)
      flag=True
      break
    except:
      flag=False
  return flag

def getLocalIp():
  localIP = socket.gethostbyname(socket.gethostname())
  return localIP

def checkNinePort(startPort):
  flag = True
  ipList = ("127.0.0.1","0.0.0.0",getLocalIp())
  for i in range(1, 10):
    if (isInuse(ipList, startPort)):
      flag = False
      break
    else:
      startPort = startPort + 1
  return flag, startPort

def findPort(startPort):
  while True:
    flag, endPort = checkNinePort(startPort)
    if (flag == True): #ninePort is ok
      break
    else:
      startPort = endPort + 1
  return startPort

def getExtFile(imgFile,ext):

    filename = os.path.splitext(imgFile)[0]
    return filename + ext

def getPts(ptsFile):

    with open(ptsFile,'r') as f:
        lines = f.readlines()


    assert int(len(lines) - 1) % 24 == 0

    nFace = int(len(lines) - 1) / 24

    res = []
    for i in range(int(nFace)):
        begin = 24*i + 3
        end = 24*i + 24

        pts = lines[begin:end]
        pts = [a.strip().split(' ') for a in pts]
        newpts = []
        for singlepoint in pts:
            singleres = []
            for axis in singlepoint:
                axis = int(float(axis))
                singleres.append(axis)
            newpts.append(singleres)

        pts = np.array(pts)
        newpts = np.array(newpts)
        # print(pts)
        res.append(newpts)
    return res

def deletByExt(dir,extension):

    list = getlist(dir,extension)
    for file in tqdm(list):
        os.remove(file)

def pts21to19(pts):

    assert pts.shape[0] == 21

    ptsNew = np.zeros((19,2))

    for i in range(5):
        ptsNew[i] = pts[i]
    ptsNew[5] = pts[6]

    for i in range(4):
        ptsNew[i+6] = pts[i+7]

    ptsNew[10] = pts[12]
    for i in range(8):
        ptsNew[i+11] = pts[i+13]

    return ptsNew


'''
        2               7
0     1    3    5    6     8    10   
        4               9

                 12

                 14
                 17
            13        15 
                 18
                 16

                 11
0：左太阳穴
10：右太阳穴

1-4：左眼
6-9：右眼
5：眉心

11：下巴
12：鼻尖
13-16：嘴巴四周
17-18：内嘴唇
'''
if __name__ =='__main__':

    # model = RecognitionBone()
    # x = torch.randn(5, 128, 16, 16)
    # targets = torch.randint(10,(5,))
    # y = model(x,targets)
    # print(y[0].shape)
    # print(y[1].shape)
    # print(y[2].shape)

    # deletByExt('/jiangtao2/code_with_git/MultiTaskOnFaceRebuild/images/Onet/','.jpg')
    # deletByExt('/jiangtao2/code_with_git/MultiTaskOnFaceRebuild/images/OnetV3/','.jpg')

    imgList = getlist('/jiangtao2/dataset/train/alignment/unmask/img_DSM/','.jpg')
    for imgFile in tqdm(imgList[0:50]):

        img = cv2.imread(imgFile,0)
        ptsFile = getExtFile(imgFile,'.pts')
        pts = getPts(ptsFile)[0]

        if(21 == pts.shape[0]):
            pts = pts21to19(pts)

        if(0):
            pts = pts[0:11]
            imgResize,ptsResize = randomAug_Onet_pts(img,pts)
            cv2.imwrite('/jiangtao2/code_with_git/MultiTaskOnFaceRebuild/images/Onet/{}'.format(os.path.basename(imgFile)),imgResize)
        if(1):

            minx,miny = pts.min(axis=0)
            maxx,maxy = pts.max(axis=0)
            box = [[minx,miny,maxx,maxy]]

            height, width = img.shape[0:2]
            minX,minY,maxX,maxY = box[0][0],box[0][1],box[0][2],box[0][3]
            label = 1

            img[int(minY):int(maxY),int(minX):int(maxX)] = img.mean()
 

            x1 = random.randint(0,int(width-100))
            y1 = random.randint(0,int(height-100))
            x2 = random.randint(x1+100,width)
            y2 = random.randint(y1+100,height)

            cropped_im = img[int(y1):int(y2), int(x1):int(x2)]
            cv2.imwrite('/jiangtao2/code_with_git/MultiTaskOnFaceRebuild/images/Onet/0_{}'.format(os.path.basename(imgFile)),img)
            cv2.imwrite('/jiangtao2/code_with_git/MultiTaskOnFaceRebuild/images/Onet/1_{}'.format(os.path.basename(imgFile)),cropped_im)

