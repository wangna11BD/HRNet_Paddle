# [HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation (CVPR 2020)](https://arxiv.org/abs/1908.10357)

## 1 Introduction
This is the paddle code of [HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation](https://arxiv.org/abs/1908.10357).  
Bottom-up human pose estimation methods have difficulties in predicting the correct pose for small persons due to challenges in scale variation. In this paper, we present **HigherHRNet**: a novel bottom-up human pose estimation method for learning scale-aware representations using high-resolution feature pyramids. Equipped with multi-resolution supervision for training and multi-resolution aggregation  for inference, the proposed approach is able to solve the scale variation challenge in *bottom-up multi-person* pose estimation and localize keypoints more precisely, especially for small person. The feature pyramid in HigherHRNet consists of feature map outputs from HRNet and upsampled higher-resolution outputs through a transposed convolution. HigherHRNet outperforms the previous best bottom-up method by 2.5% AP for medium person on COCO test-dev, showing its effectiveness in handling scale variation. Furthermore, HigherHRNet achieves new state-of-the-art result on COCO test-dev (70.5% AP) without using refinement or other post-processing techniques, surpassing all existing bottom-up methods. HigherHRNet even surpasses all top-down methods on CrowdPose test (67.6% AP), suggesting its robustness in crowded scene. 

![Illustrating the architecture of the proposed Higher-HRNet](/figures/arch_v2.png)

## 2 How to use

### 2.1 Environment

### Requirements:
- PaddlePaddle 2.2
- OS 64 bit
- Python 3(3.5.1+/3.6/3.7/3.8/3.9)，64 bit
- pip/pip3(9.0.1+), 64 bit
- CUDA >= 10.1
- cuDNN >= 7.6

### Installation
#### 1. Install PaddlePaddle
```
# CUDA10.1
python -m pip install paddlepaddle-gpu==2.2.0.post101 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

- For more CUDA version or environment to quick install, please refer to the [PaddlePaddle Quick Installation document](https://www.paddlepaddle.org.cn/install/quick)
- For more installation methods such as conda or compile with source code, please refer to the [installation document](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html)

Please make sure that your PaddlePaddle is installed successfully and the version is not lower than the required version. Use the following command to verify.

```
# check
>>> import paddle
>>> paddle.utils.run_check()

# confirm the paddle's version
python -c "import paddle; print(paddle.__version__)"
```

**Note**

1.  If you want to use PaddleDetection on multi-GPU, please install NCCL at first.

#### 2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
#### 3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
#### 4. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── config
   ├── dataset
   ├── figures
   ├── lib
   ├── log
   ├── output
   ├── tools 
   ├── README.md
   └── requirements.txt
   ```

### 2.2 Data preparation
#### COCO Data Download
- The coco dataset is downloaded automatically through the code. The dataset is large and takes a long time to download

    ```
    # automatically download coco datasets by executing code
    python dataset/download_coco.py
    ```

    after code execution, the organization structure of coco dataset file is：
    ```
    >>cd dataset
    >>tree
    ├── annotations
    │   ├── instances_train2017.json
    │   ├── instances_val2017.json
    │   |   ...
    ├── train2017
    │   ├── 000000000009.jpg
    │   ├── 000000580008.jpg
    │   |   ...
    ├── val2017
    │   ├── 000000000139.jpg
    │   ├── 000000000285.jpg
    │   |   ...
    |   ...
    ```
- If the coco dataset has been downloaded  
    The files can be organized according to the above data file organization structure.

### 2.3 Training & Evaluation & Inference

We provides scripts for training, evalution and inference with various features according to different configure.

```bash
# training on single-GPU
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/dark_hrnet_w32_256x192.yml

# training on multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/dark_hrnet_w32_256x192.yml

# GPU evaluation
export CUDA_VISIBLE_DEVICES=0
python tools/eval.py -c configs/dark_hrnet_w32_256x192.yml -o weights=https://paddledet.bj.bcebos.com/models/keypoint/hrnet_w32_256x192.pdparams

# Inference
python tools/infer.py -c configs/dark_hrnet_w32_256x192.yml --infer_img=dataset/test_image/000000397133.jpg -o weights=https://paddledet.bj.bcebos.com/models/keypoint/hrnet_w32_256x192.pdparams
```

## 3 Result
COCO Dataset
| Model              | Input Size | AP(coco val) |                           Model Download                           | Config File                                                    |
| :---------------- | -------- | :----------: | :----------------------------------------------------------: | ----------------------------------------------------------- |
| HRNet-w32             | 256x192  |     77.8     | [hrnet_w32_256x192.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/hrnet_w32_256x192.pdparams) | [config](./hrnet/hrnet_w32_256x192.yml)                     |

## Citation
````
@inproceedings{cheng2020bottom,
  title={HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation},
  author={Bowen Cheng and Bin Xiao and Jingdong Wang and Honghui Shi and Thomas S. Huang and Lei Zhang},
  booktitle={CVPR},
  year={2020}
}
````

