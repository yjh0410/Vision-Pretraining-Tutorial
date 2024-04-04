# General Image Classification Laboratory


## Train a CNN
We have kindly provided a bash script `train_weak_aug.sh` for training small and medium models and a bash script `train_strong_aug.sh` for training large models. You can modify some hyperparameters in the script file according to your own needs.

For example, we are going to use 8 GPUs to train `RTCNet-N` designed in this repo, so we can use the following command:

```Shell
bash train_weak_aug.sh rtcnet_n 128 imagenet_1k path/to/imagnet_1k 8 1699 None
```

## Evaluate a CNN
- Evaluate the `top1 & top5` accuracy of `ViT-Tiny` on ImageNet-1K dataset:
```Shell
python main.py --cuda -dataset imagenet_1k --root path/to/imagnet_1k -m rtcnet_n --batch_size 256 --img_size 224 --eval --resume path/to/rtcnet_n.pth
```


## Experimental results
Tips:
- **Weak augmentation:** includes `random hflip` and `random crop resize`.
- **Strong augmentation:** includes `mixup`, `cutmix`, `rand aug`, `random erase` and so on.
- The `AdamW` with `weight decay = 0.05` and `base lr = 4e-3 (for bs of 4096)` is deployed as the optimzier, and the `CosineAnnealingLR` is deployed as the lr scheduler, where the `min lr` is set to 1e-6.
- The `EMA` is used for the model with `decay=0.9999` and `tau=2000`.

### ImageNet-1K
* Modified DarkNet (Yolov3's DarkNet with width and depth scaling factors)

|    Model      | Augment | Batch | Epoch | size | acc@1 | GFLOPs | Params |  Weight |
|---------------|---------|-------|-------|------|-------|--------|--------|---------|
| DarkNet-S     |   weak  |  4096 |  100  | 224  |       |        |        |  |
| DarkNet-M     |   weak  |  4096 |  100  | 224  |       |        |        |  |
| DarkNet-L     |   weak  |  4096 |  100  | 224  |       |        |        |  |
| DarkNet-X     |   weak  |  4096 |  100  | 224  |       |        |        |  |

* Modified CSPDarkNet (Yolov5's DarkNet with width and depth scaling factors)

|    Model      | Augment | Batch | Epoch | size | acc@1 | GFLOPs | Params |  Weight |
|---------------|---------|-------|-------|------|-------|--------|--------|---------|
| CSPDarkNet-S  |   weak  |  4096 |  100  | 224  |       |        |        |  |
| CSPDarkNet-M  |   weak  |  4096 |  100  | 224  |       |        |        |  |
| CSPDarkNet-L  |   weak  |  4096 |  100  | 224  |       |        |        |  |
| CSPDarkNet-X  |   weak  |  4096 |  100  | 224  |       |        |        |  |

* RTCNet (Yolov8's backbone)

|    Model      | Augment | Batch | Epoch | size | acc@1 | GFLOPs | Params  |  Weight |
|---------------|---------|-------|-------|------|-------|--------|---------|---------|
| RTCNet-N      |   weak  |  4096 |  100  | 224  |  62.1 |  0.38  | 1.36 M  | [ckpt](https://github.com/yjh0410/ICLab/releases/download/in1k_pretrained/rtcnet_n_in1k_62.1.pth) |
| RTCNet-S      |   weak  |  4096 |  100  | 224  |  71.3 |  1.48  | 4.94 M  | [ckpt](https://github.com/yjh0410/ICLab/releases/download/in1k_pretrained/rtcnet_s_in1k_71.3.pth) |
| RTCNet-M      |   weak  |  4096 |  100  | 224  |       |  4.67  | 11.60 M |  |
| RTCNet-L      |   weak  |  4096 |  100  | 224  |       |  10.47 | 19.66 M |  |
| RTCNet-X      |   weak  |  4096 |  100  | 224  |       |  20.56 | 37.86 M |  |

* MiniNet

|    Model      | Augment | Batch | Epoch | size | acc@1 | GFLOPs | Params |  Weight |
|---------------|---------|-------|-------|------|-------|--------|--------|---------|
| MiniNet-S     |   weak  |  4096 |  300  | 224  |       |        |        |  |
| MiniNet-M     |   weak  |  4096 |  300  | 224  |       |        |        |  |
| MiniNet-L     |   weak  |  4096 |  300  | 224  |       |        |        |  |
