# Masked AutoEncoder

## 1. Pretrain
We have kindly provided the bash script `train_pretrain.sh` file for pretraining. You can modify some hyperparameters in the script file according to your own needs.

```Shell
cd Vision-Pretraining-Tutorial/masked_image_modeling/
python main_pretrain.py --cuda \
                        --dataset cifar10 \
                        --model vit_t \
                        --batch_size 256 \
                        --optimizer adamw \
                        --base_lr 1e-3 \
                        --min_lr 1e-6
```

## 2. Finetune
We have kindly provided the bash script `train_finetune.sh` file for finetuning. You can modify some hyperparameters in the script file according to your own needs.

```Shell
cd Vision-Pretraining-Tutorial/masked_image_modeling/
python main_finetune.py --cuda \
                        --dataset cifar10 \
                        --model vit_t \
                        --batch_size 256 \
                        --optimizer adamw \
                        --base_lr 1e-3 \
                        --min_lr 1e-6
```
## 3. Evaluate 
- Evaluate the `top1 & top5` accuracy of `ViT-Tiny` on CIFAR10 dataset:
```Shell
python train_finetune.py --dataset cifar10 -m vit_tiny --batch_size 256 --img_size 32 --patch_size 2 --eval --resume path/to/vit_tiny_cifar10.pth
```

- Evaluate the `top1 & top5` accuracy of `ViT-Tiny` on ImageNet-1K dataset:
```Shell
python train_finetune.py --dataset cifar10 -m vit_tiny --batch_size 256 --img_size 224 --patch_size 16 --eval --resume path/to/vit_tiny_cifar10.pth
```


## 4. Visualize Image Reconstruction
- Evaluate `MAE-ViT-Tiny` on CIFAR10 dataset:
```Shell
python train_pretrain.py --dataset cifar10 -m mae_vit_tiny --resume path/to/mae_vit_tiny_cifar10.pth --img_size 32 --patch_size 2 --eval --batch_size 1
```

- Evaluate `MAE-ViT-Tiny` on ImageNet-1K dataset:
```Shell
python train_pretrain.py --dataset cifar10 -m mae_vit_tiny --resume path/to/mae_vit_tiny_cifar10.pth --img_size 224 --patch_size 16 --eval --batch_size 1
```


## 5. Experiments
- On CIFAR10

| Method |  Model  | Epoch | Top 1    | Weight |  MAE weight  |
|  :---: |  :---:  | :---: | :---:    | :---:  |    :---:     |
|  MAE   |  ViT-T  | 100   |   91.2   | [ckpt](https://github.com/yjh0410/MAE/releases/download/checkpoints/ViT-T_Cifar10.pth) | [ckpt](https://github.com/yjh0410/MAE/releases/download/checkpoints/MAE_ViT-T_Cifar10.pth) |


## 6. Acknowledgment
Thank you to **Kaiming He** for his inspiring work on [MAE](http://openaccess.thecvf.com/content/CVPR2022/papers/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf). His research effectively elucidates the semantic distinctions between vision and language, offering valuable insights for subsequent vision-related studies. I would also like to express my gratitude for the official source code of [MAE](https://github.com/facebookresearch/mae). Additionally, I appreciate the efforts of [**IcarusWizard**](https://github.com/IcarusWizard) for reproducing the [MAE](https://github.com/IcarusWizard/MAE) implementation.
