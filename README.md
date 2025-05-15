# BambInk
Official Pytorch Code base for "BambInk: Attention-CycleGAN for Ancient Manuscript Bamboo Slips Ink Enhancement".

## Using the code:
The code is stable while using Python 3.8.3, PyTorch 1.13.0, CUDA =11.4.
- Clone this repository:
```bash
git clone https://github.com/Truman0o0/BambInk.git
cd BambInk
```

## DataFormat
Make sure to put the files as the following structure:
```
dataset
├── train
|   ├── A
|   │   ├── ...
|   │
|   └── B
|       ├── ...
|
└── test
    ├── A
    |   ├── ...
    |
    └── B
        ├── ...
```

## Training and Validation
### 1) Train the model.
```
python train.py --dataroot path_dataset --lr 0.0002 --n_epochs 100 --size 256 --batchSize 4
```

### 2) Test.
```
python test.py --dataroot path_dataset --size 256 --batchSize 4
```
