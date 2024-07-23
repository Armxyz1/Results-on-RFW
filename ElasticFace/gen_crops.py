import net
import torch
import os
from face_alignment import align
import numpy as np
from tqdm import tqdm
import cv2
adaface_models = {
    'ir_18':"adaface_ir18_casia.ckpt",
}

def load_pretrained_model(architecture='ir_18'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img/ 255.) - 0.5) / 0.5
    tensor = torch.tensor(np.array([brg_img.transpose(2,0,1)])).float()
    return tensor

if __name__ == '__main__':

    model = load_pretrained_model('ir_18')
    #feature, norm = model(torch.randn(2,3,112,112))
    
    races = ['African','Asian','Caucasian','Indian']
    genders = ['Man','Woman']
    for race in races:
        for gender in genders:
            test_image_path = f'../images/test/data/Just_Images/{race}/{gender}'
            with open(f'./cropped/{race}_{gender}_cropped.npy','wb') as f:
                for fname in tqdm(sorted(os.listdir(test_image_path))):
                    aligned_rgb_img = align.get_aligned_face(f"{test_image_path}/{fname}")
                    bgr_tensor_input = to_input(aligned_rgb_img)
                    arr = np.array(bgr_tensor_input.detach())
                    np.save(f,arr)
