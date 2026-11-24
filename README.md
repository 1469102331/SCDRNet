# Learning Spectral Consistency and Detail Refinement for Hyperspectral Image Fusion

The code for 《 Learning Spectral Consistency and Detail Refinement for Hyperspectral Image Fusion 》

> **Status**: accepted for publication; the online/camera-ready version will be linked here once available.


![Language](https://img.shields.io/badge/language-python-brightgreen)

Our model (SCDRNet) was trained on an NVIDIA A800-SXM4-80GB GPU.

<div align="center">
    <img src="SCDRNet.png" alt="framework" width="900"/>
</div>

## 👉 Data

NOTE: the training/testing scripts expect the following directory layout:

```
Data/
├── CAVE/
│   ├── Train/HSI/
│   └── Test/HSI/
└── Harvard/
    ├── Train/HSI/
    └── Test/HSI/
```

Datasets:

* [CAVE](https://www.cs.columbia.edu/CAVE/databases/multispectral/)
* [Harvard](http://vision.seas.harvard.edu/hyperspec/)
* [Pavia](https://github.com/liangjiandeng/HyperPanCollection)

## 🌿 Getting Started

### Environment Setup

    conda create -n scdrnet_env python==3.11

    conda activate scdrnet_env

    pip install -r requirements.txt

### Train

    python train_cave.py

Checkpoints and logs are written to `./checkpoint_cave/model/`.

### Test

    python Test.py

The test script loads `./checkpoint_cave/model/model_%04d.pth` (edit `test_epoch` to match your checkpoint),
saves fused results to `./test_save/model/<dataset>/`, and reports PSNR / RMSE / SAM / ERGAS / UIQI.

## 📁 Code Structure

```
SCDRNet/
├── Model/
│   └── model.py            # SCDRNet: coefficient estimation + spectral reconstruction + DWT/DGHA refinement
├── train_cave.py           # training script (CAVE / KAIST switchable)
├── Test.py                 # testing / evaluation script
├── data_load.py            # dataset classes (CAVE / Harvard / KAIST / ICVL ...)
├── Utils.py                # data preparation, Gaussian downsample, SSIM, etc.
├── calculate_metrics.py    # PSNR / RMSE / SAM / ERGAS / UIQI and AverageMeter
├── requirements.txt
├── SCDRNet.pdf             # paper
└── SCDRNet.png             # framework figure
```

## ⚠️ Notes

* `train_cave.py` imports `from loss import *` and calls `calculate_swt_frequency_loss(...)`. This loss module is **not** included in this repository yet — add your own `loss.py` before training, or replace the call with plain `L1Loss` / `MSELoss`.
* `model.py` currently returns `(fused, pseudo_hsi, spectral_abundance)` while `train_cave.py` unpacks three outputs and supervises the third with the downsampled MSI. Make sure the third output and its supervision are consistent with the version you trained.
* `read_r()` is called once **per iteration** inside the training loop (`r = read_r().to(device)`), which re-reads `./r.mat` from disk every batch, and the returned `r` is not used afterwards. Move it out of the loop (or delete the line) if you do not need it.
* `sio` used inside `read_r()` is available through `from Utils import *` (`Utils.py` imports `scipy.io as sio`), so no extra import is required there.

## Citation

If this code is useful for your research, please cite this paper.

## 🌸 Acknowledgment

The repository layout, training/testing pipeline and evaluation code follow our previous work
[SINet](https://github.com/1469102331/SINet), which in turn refers to
[DSPNet](https://github.com/syc11-25/DSPNet/tree/main),
[LRTN](https://github.com/YuanyeLiu/Low-Rank-Transformer-For-High-Resolution-Hyperspectral-Computational-Imaging),
[Hyper-DSNet](https://github.com/liangjiandeng/Hyper-DSNet) and
[HyperPanCollection](https://github.com/liangjiandeng/HyperPanCollection).
We thank all the contributors for open-sourcing.
