from __future__ import print_function
from tqdm import tqdm
import torch
torch.backends.cudnn.benchmark = True
import os
import cv2
import numpy as np
from matlab_cp2tform import get_similarity_transform_for_cv2
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import tensorflow as tf

model = tf.keras.models.load_model('./models/GN_W1.3_S2_ArcFace_epoch48.h5',compile=False)

def alignment(src_img,src_pts):
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    crop_size = (112, 112)
    src_pts = np.array(src_pts).reshape(5,2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img

races = ['African','Asian','Caucasian','Indian']

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

                i1 = img1.transpose(2, 0, 1).reshape((1,3,112,112))
                i1 = (i1-127.5)/128.0
                i1 = np.transpose(i1,(0,2,3,1))
                i2 = img2.transpose(2, 0, 1).reshape((1,3,112,112))
                i2 = (i2-127.5)/128.0
                i2 = np.transpose(i2,(0,2,3,1))

                f1 = model.predict(i1)
                f2 = model.predict(i2)
                f1 = torch.Tensor(f1).to('cuda').view(-1)
                f2 = torch.Tensor(f2).to('cuda').view(-1)
                
                with torch.no_grad():
                    cosdistance = torch.dot(f1,f2)/(f1.norm()*f2.norm())
                cosdistance = cosdistance.cpu().numpy().item()

                with open(f"./sims/{race}_sims.csv", 'a') as f:
                    f.write(f"{path1},{path2},{cosdistance},{sim}\n")
            except Exception as e:
                print(e)
                continue
            
            
