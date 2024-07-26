# Results-on-RFW
These are the results obtained on RFW (test) dataset on various models, @ DRDO Young Scientist AI Lab, Bengaluru.

# Data Preparation
To run these, I segregated the RFW classes into two sub classes of Man and Woman, using DeepFace model https://github.com/serengil/deepface.  
To obtain the data, download RFW dataset (http://www.whdeng.cn/RFW/index.html) place it in the repository in a directory `images`. Next, copy the `./data_prep` directory into `images` and run `./images/data_prep/data_prep.ipynb`. At the end, your `images` directory structure should look like `./sample_images`.

The dataset has been taken (and adapted from) from: http://www.whdeng.cn/RFW/index.html

# Code
The code as been adapted from:   
**AdaFace**: https://github.com/mk-minchul/AdaFace   
**ArcFace**: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch   
**ElasticFace**: https://github.com/fdbtrs/ElasticFace/tree/main   
**GhostFace**: https://github.com/HamadYA/GhostFaceNets/tree/main , https://github.com/Hazqeel09/ellzaf_ml , https://www.kaggle.com/datasets/ipythonx/ghostnetpretrained/data and https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnet_pytorch    
**SphereFace**: https://github.com/clcarwin/sphereface_pytorch   

The model backbones used for each are:  
**AdaFace**: ***r100 WebFace 12M*** https://drive.google.com/file/d/1dswnavflETcnAuplZj1IOKKP0eM8ITgT/view  
**ArcFace**: ***r100 MS1MV3*** https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&id=4A83B6B633B029CC%215585&cid=4A83B6B633B029CC  
**ElasticFace**: ***cos+ MS1MV2*** https://drive.google.com/drive/folders/19LXrjVNt60JBZP7JqsvOSWMwGLGrcJl5  
**GhostFace**: ***GN_W1.3_S2_ArcFace_epoch48 MS1MV3*** https://github.com/HamadYA/GhostFaceNets/releases/download/v1.3/GhostFaceNet_W1.3_S2_ArcFace.h5 , https://www.kaggle.com/datasets/ipythonx/ghostnetpretrained/data , and https://www.kaggle.com/datasets/tempusme/ghostfacenet  
**SphereFace**: ***SphereFace20a Casia*** https://github.com/clcarwin/sphereface_pytorch/tree/master/model  

All the models can be found at:  
https://drive.google.com/file/d/1YWIrkFIHw-Q6KUAyX7x7GZXAt4Bw5p1q/view?usp=sharing  
https://drive.google.com/file/d/1WxQ_1BYRx1g-4zIEKTywDW8-GWNCQ6ea/view?usp=sharing

Please download the models, unzip and place the weights file in the respective `models` folder.  

For each folder (except for SphereFace), there are 2 eval files:  
`rfw_eval`: Uses RFW landmarks to generate cropped faces.  
`rfw_eval_comb`: Uses RFW landmarks on the default buckets, i.e., no gender split.  

For each model, I have provided my results in folders `sims`,`results`, and `roc`.

# Results:

**TPRs at FPR $=\mathbf{10^{-3}}$**
![TPRs at FPR= 10^(-3)](./tpr@E-03.png "TPRs at FPR= 10^(-3)")

**Accuracies:**
![Accuracies](./acc.png "Accuracies")

# Citations:  
[1] Mei Wang, Weihong Deng, Jiani Hu, Xunqiang Tao, Yaohai Huang. Racial Faces in the Wild: Reducing Racial Bias by Information Maximization Adaptation Network. ICCV2019.  
[2] Mei Wang, Yaobin Zhang, Weihong Deng. Meta Balanced Network for Fair Face Recognition. TPAMI 2021.  
[3] Mei Wang, Weihong Deng. Mitigating Bias in Face Recognition using Skewness-Aware Reinforcement Learning. CVPR2020.  
[4] Mei Wang, Weihong Deng. Deep face recognition: A Survey. Neurocomputing.  
[5] https://github.com/mk-minchul/AdaFace  
[6] https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch  
[7] https://github.com/fdbtrs/ElasticFace/tree/main  
[8] https://github.com/HamadYA/GhostFaceNets/tree/main  
[9] https://github.com/Hazqeel09/ellzaf_ml  
[10] https://www.kaggle.com/datasets/ipythonx/ghostnetpretrained/data  
[11] https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnet_pytorch  
[12] https://github.com/clcarwin/sphereface_pytorch   
