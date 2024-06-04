# Development_of_deep_learning_systems

Paper link: https://paperswithcode.com/paper/attention-guided-low-light-image-enhancement

Original repository: https://github.com/yu-li/AGLLNet

## Description

Low-light image enhancement is challenging in that it needs to consider not only brightness recovery but also complex issues like color distortion and noise, which usually hide in the dark. Simply adjusting the brightness of a low-light image will inevitably amplify those artifacts. To address this difficult problem, this paper proposes a novel end-to-end attention-guided method based on multi-branch convolutional neural network. To this end, we first construct a synthetic dataset with carefully designed low-light simulation strategies. The dataset is much larger and more diverse than existing ones. With the new dataset for training, our method learns two attention maps to guide the brightness enhancement and denoising tasks respectively. The first attention map distinguishes underexposed regions from well lit regions, and the second attention map distinguishes noises from real textures. With their guidance, the proposed multi-branch decomposition-and-fusion enhancement network works in an input adaptive way. Moreover, a reinforcement-net further enhances color and contrast of the output image. Extensive experiments on multiple datasets demonstrate that our method can produce high fidelity enhancement results for low-light images and outperforms the current state-of-the-art methods by a large margin both quantitatively and visually.

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
