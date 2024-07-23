# Results-on-RFW
These are the results obtained on RFW (test) dataset on various models, @ DRDO Young Scientist AI Lab, Bengaluru.

# Data Preparation
To run these, I segregated the RFW classes into two sub classes of Man and Woman, using DeepFace model https://github.com/serengil/deepface.  
To obtain the data, download RFW dataset (http://www.whdeng.cn/RFW/index.html) place it in the repository in a directory `images`. Next, copy the `./data_prep` directory into `images` and run `./images/data_prep/data_prep.ipynb`. At the end, your `images` directory structure should look like `./sample_images`.

The dataset has been taken (and adapted from) from: http://www.whdeng.cn/RFW/index.html

# Code
The code as been adapted from:   
AdaFace: https://github.com/mk-minchul/AdaFace   
ArcFace: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch   
ElasticFace: https://github.com/fdbtrs/ElasticFace/tree/main   
GhostFace: https://github.com/HamadYA/GhostFaceNets/tree/main   
SphereFace: https://github.com/clcarwin/sphereface_pytorch   

The model backbones used for each are:  
AdaFace: r100 WebFace 12M https://drive.google.com/file/d/1dswnavflETcnAuplZj1IOKKP0eM8ITgT/view  
ArcFace: r100 MS1MV3 https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&id=4A83B6B633B029CC%215585&cid=4A83B6B633B029CC  
ElasticFace: cos+ MS1MV2 https://drive.google.com/drive/folders/19LXrjVNt60JBZP7JqsvOSWMwGLGrcJl5  
GhostFace: GN_W1.3_S2_ArcFace_epoch48 MS1MV3 https://github.com/HamadYA/GhostFaceNets/releases/download/v1.3/GhostFaceNet_W1.3_S2_ArcFace.h5  
SphereFace: SphereFace20a Casia https://github.com/clcarwin/sphereface_pytorch/tree/master/model  

All the models can be found at:  
https://drive.google.com/file/d/1ARA1Lbb4tStk80Dnb3oP-1cX8WYJiCJd/view?usp=sharing  
https://drive.google.com/drive/folders/1ty44wEqaNGhL-TlyDOoWvotaET7AZ-tp?usp=sharing  
https://drive.google.com/file/d/1hWZs7z3TZ7DX4naB1K59oM0y9WcaE6FU/view?usp=sharing  

Please download the models, unzip and place the weights file in the respective `models` folder.  

For each folder (except for SphereFace), there are three eval files:  
`pairs_eval`: Uses AdaFace to generate cropped faces.    
`pairs_eval_lmks`: Uses RFW landmarks to generate cropped faces.  
`pairs_eval_lmks_comb`: Uses RFW landmarks on the default buckets, i.e., no gender split.  
The results have been calculated using RFW landmarks.  

For each model, I have provided my results in folders `lmks`,`results`, and `roc`.
# Citations:  
[1] Mei Wang, Weihong Deng, Jiani Hu, Xunqiang Tao, Yaohai Huang. Racial Faces in the Wild: Reducing Racial Bias by Information Maximization Adaptation Network. ICCV2019.  
[2] Mei Wang, Yaobin Zhang, Weihong Deng. Meta Balanced Network for Fair Face Recognition. TPAMI 2021.  
[3] Mei Wang, Weihong Deng. Mitigating Bias in Face Recognition using Skewness-Aware Reinforcement Learning. CVPR2020.  
[4] Mei Wang, Weihong Deng. Deep face recognition: A Survey. Neurocomputing.  
