from __future__ import print_function
from tqdm import tqdm
import torch
from torch.autograd import Variable
torch.backends.cudnn.benchmark = True
import os
import cv2
import numpy as np

from matlab_cp2tform import get_similarity_transform_for_cv2
import backbones.net_sphere as net_sphere

def alignment(src_img,src_pts):
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5,2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img


def KFold(n=6000, n_folds=10, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[i*n/n_folds:(i+1)*n/n_folds]
        train = list(set(base)-set(test))
        folds.append([train,test])
    return folds

def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0*np.count_nonzero(y_true==y_predict)/len(y_true)
    return accuracy

def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold


net = getattr(net_sphere,'sphere20a')()
net.load_state_dict(torch.load("./models/sphere20a_20171020.pth"))
net.cuda()
net.eval()
net.feature = True

races = ['African','Asian','Caucasian','Indian']
genders = ['Man','Woman']

African_wom = os.listdir('../images/test/data/African/Woman')
Caucasian_wom = os.listdir('../images/test/data/Caucasian/Woman')
Asian_wom = os.listdir('../images/test/data/Asian/Woman')
Indian_wom = os.listdir('../images/test/data/Indian/Woman')
Asian_man = os.listdir('../images/test/data/Asian/Man')
African_man = os.listdir('../images/test/data/African/Man')
Caucasian_man = os.listdir('../images/test/data/Caucasian/Man')
Indian_man = os.listdir('../images/test/data/Indian/Man')

for race in races:

    landmark = {}
    with open(f'../images/test/txts/{race}/{race}_lmk.txt') as f:
        landmark_lines = f.readlines()
    for line in landmark_lines:
        l = line.strip().split()
        landmark[l[0]] = [float(k) for k in l[2:]]

    with open(f"../images/test/txts/{race}/{race}_pairs.txt", 'r') as f:
        pairs_lines = f.readlines()

    for i in tqdm(range(6000)):
        try:
            p = pairs_lines[i].replace('\n','').split('\t')
            race2 = None
            gen2 = None
            if p[0] in eval(f'{race}_wom'):
                        gender = "Woman"
            else:
                        gender = "Man"
            sim = 0
            path1 = f"../images/test/data/Just_Images/{race}/{gender}/{p[0]}_000{p[1]}.jpg"
            if len(p) == 3:
                    path2 = f"../images/test/data/Just_Images/{race}/{gender}/{p[0]}_000{p[2]}.jpg"
                    sim = 1
                    name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
                    name2 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[2]))
            else:
                    if p[2] in African_wom or p[2] in Caucasian_wom or p[2] in Asian_wom or p[2] in Indian_wom:
                        gen2 = "Woman"
                        if p[2] in African_wom:
                            race2 = "African"
                        elif p[2] in Caucasian_wom:
                            race2 = "Caucasian"
                        elif p[2] in Asian_wom:
                            race2 = "Asian"
                        else:
                            race2 = "Indian"
                    else:
                        gen2 = "Man"
                        if p[2] in African_man:
                            race2 = "African"
                        elif p[2] in Caucasian_man:
                            race2 = "Caucasian"
                        elif p[2] in Asian_man:
                            race2 = "Asian"
                        else:
                            race2 = "Indian"
                    path2 = f"../images/test/data/Just_Images/{race2}/{gen2}/{p[2]}_000{p[3]}.jpg"
                    name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
                    name2 = p[2]+'/'+p[2]+'_'+'{:04}.jpg'.format(int(p[3]))
            
            img1 = alignment(cv2.imread(path1),landmark[f"/test/data/{race}/{name1}"])
            img2 = alignment(cv2.imread(path2),landmark[f"/test/data/{race}/{name2}"])

            imglist = [img1,cv2.flip(img1,1),img2,cv2.flip(img2,1)]
            for i in range(len(imglist)):
                imglist[i] = imglist[i].transpose(2, 0, 1).reshape((1,3,112,96))
                imglist[i] = (imglist[i]-127.5)/128.0

            img = np.vstack(imglist)
            with torch.no_grad():
                img = Variable(torch.from_numpy(img).float()).cuda()
            output = net(img)
            f = output.data
            f1,f2 = f[0],f[2]
            cosdistance = f1.dot(f2)/(f1.norm()*f2.norm()+1e-5)
            with open(f"./sims/{race}_{gender}_sims.csv", 'a') as f:
                f.write(f"{path1},{path2},{cosdistance},{sim}\n")
        except Exception as e:
            print(e)
            continue