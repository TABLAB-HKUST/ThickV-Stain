# ThickV-Stain in PyTorch
A multi-scale virtual staining model for unprocessed thick tissues

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation
- Clone this repo:
```bash
git clone https://github.com/TABLAB-HKUST/ThickV-Stain.git
cd ThickV-Stain
```

### Train and test
- Prepare the dataset: (a folder includes ```trainA``` and ```trainB```)
- Train a model:
```
  python train.py --dataroot your_dataset --name your_modelname --model multi_scale
```
- Test the model:
```
  python test.py --dataroot your_dataset --name your_modelname --model multi_scale
```
- We provided our trained model [here](https://drive.google.com/file/d/1v0vtl6nRH0MCKKYL1MjRzXTH4ZCBxf-x/view?usp=drive_link).

## Acknowledgments
Our code follows the architecture of[pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
