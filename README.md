# Development_of_deep_learning_systems

Paper link: https://paperswithcode.com/paper/attention-guided-low-light-image-enhancement

Original repository: https://github.com/yu-li/AGLLNet

## Description

The authors of the paper present a deep learning based method for low-light image enhancement. This
problem is challenging due to the difficulty in handling various factors simultaneously
including brightness, contrast, artifacts and noise. To address this task, we propose the
multi-branch low-light enhancement network (MBLLEN). The key idea is to extract rich
features up to different levels, so that we can apply enhancement via multiple subnets
and finally produce the output image via multi-branch fusion. In this manner, image
quality is improved from different aspects. Through extensive experiments, our proposed
MBLLEN is found to outperform the state-of-art techniques by a large margin. We additionally show that our method can be directly extended to handle low-light videos.

## Instructions:
1. Clone repository:
```shell script
git clone https://github.com/yasshma/AGLLNet.git
```
2. Go to the required folder:
```shell script
cd AGLLNet
```
3. Build docker:
```shell script
docker build -t agllnet .
```
4. Run docker:
```shell script
docker run -v ./:/app --shm-size 20G agllnet
```

The result is here: 
```shell script
./output/
```
