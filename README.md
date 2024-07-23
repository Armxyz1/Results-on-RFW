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
