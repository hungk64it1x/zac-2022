# Zalo AI Challenge 2022 - Liveness Detection - 1st place solution


## 1. INSTALLATION
### 1.1 Phần cứng
- 2 x NVIDIA TESLA A100 40G GPU
- Ubuntu 18.04.5 LTS
- CUDA 11.4
- Python 3.6.9

### 1.2 Cài đặt môi trường:
```
conda env create -f environment.yml
conda activate zac2022
```

## 2. DATASET

### 2.1 Chuẩn bị dữ liệu huấn luyện

```
mkdir raw
mkdir train_video&&cd train_video
wget https://dl-challenge.zalo.ai/liveness-detection/train.zip
```
```
unzip train.zip
rm -rf train.zip
mv ./train/videos/*.mp4 .
mv ./train/label.csv ../raw
rm -rf train
```
### 2.2 Chuẩn bị dữ liệu test

```
cd ..
mkdir data&&cd data
wget https://dl-challenge.zalo.ai/liveness-detection/private_test.zip
```
```
unzip private_test.zip
rm -rf private_test.zip
mv ./private_test/videos/*.mp4 .
rm -rf private_test
```

### 2.3 Extract frames từ dữ liệu huấn luyện
```
cd ..
bash prepare_data.sh
```
### 2.4 Tạo sample test
```
python create_sample_sub.py
```
## 3. SOLUTION SUMMARY
![image](https://user-images.githubusercontent.com/80585483/208281892-79a4c918-9856-441c-bfbf-73071d2d201a.png)
## 4. TRAINING

### 4.1 Tải pretrained ConvNeXt
```
wget https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_224_ema.pth -P weights
wget https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_224.pth -P weights
```

### 4.2 Huấn luyện 

```
bash scripts/train_convnext_xlarge_fold0.sh
bash scripts/train_convnext_xlarge_fold1.sh
bash scripts/train_convnext_xlarge_fold2.sh
bash scripts/train_convnext_xlarge_fold3.sh
bash scripts/train_convnext_xlarge_fold4.sh


bash scripts/train_convnext_large_fold0.sh
bash scripts/train_convnext_large_fold1.sh
bash scripts/train_convnext_large_fold2.sh
```

## 5. INFERENCE

```
bash predict.sh
```
Có thể sử dụng checkpoints nhóm mình đã train bằng cách download từ google drive:
```
bash download_weights.sh
bash predict.sh
```
Output là file **submission.csv** sẽ được lưu vào folder result. Sử dụng file này để nộp kết quả lên hệ thống.

## 6. RESULTS

### 6.1 Comparision of ConvNeXt with different models
![image](https://user-images.githubusercontent.com/80585483/208281942-03abf0a5-cfca-452a-be1c-589866775385.png)
![image](https://user-images.githubusercontent.com/80585483/208281954-ece9b374-4033-4a04-9c17-a6d30fcf256f.png)

### 6.2 Ablation study
![image](https://user-images.githubusercontent.com/80585483/208281963-19083127-d9a9-4b81-8aa0-087dc558ff1f.png)
![image](https://user-images.githubusercontent.com/80585483/208281969-4ad3d607-86d1-4620-be54-5fdf4ed36a94.png)

### 6.3 Final result
![image](https://user-images.githubusercontent.com/80585483/208289064-e4644590-ca7d-402b-b58a-1b992c31cbba.png)

## 7. REFERENCES
- [Zalo AI Challenge 2022](https://challenge.zalo.ai/portal/liveness-detection)

- [timm](https://github.com/rwightman/pytorch-image-models)

- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
