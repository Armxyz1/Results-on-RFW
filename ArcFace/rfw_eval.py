from __future__ import print_function
from tqdm import tqdm
import torch
torch.backends.cudnn.benchmark = True
import os
import cv2
import numpy as np

from matlab_cp2tform import get_similarity_transform_for_cv2
from backbones.iresnet import iresnet100

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

device ='cuda'

model = iresnet100()
model_dict = model.state_dict()
pretrained_dict = torch.load("./models/backbone.pth")
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model.to(torch.device(device))

model.eval()

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
                i1 = img1.transpose(2, 0, 1).reshape((1,3,112,112))
                i1 = (i1-127.5)/128.0
                i2 = img2.transpose(2, 0, 1).reshape((1,3,112,112))
                i2 = (i2-127.5)/128.0
                with torch.no_grad():
                    i1 = torch.Tensor(torch.from_numpy(i1).float()).to('cuda')
                    i2 = torch.Tensor(torch.from_numpy(i2).float()).to('cuda')
                    f1 = model(i1)[0]
                    f2 = model(i2)[0]
                cosdistance = torch.matmul(f1,f2)/(f1.norm()*f2.norm()+1e-5)
                cosdistance = cosdistance.cpu().numpy().item()
                with open(f"./sims/{race}_{gender}_sims.csv", 'a') as f:
                    f.write(f"{path1},{path2},{cosdistance},{sim}\n")
            except Exception as e:
                print(e)
                continue
            
