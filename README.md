# Ink Enhancement for Ancient Bamboo Manuscripts  Using Iterative Restoration- Degradation Adversarial Learning

## Abstract
Bamboo slips are crucial documentary evidences for investigating ancient Chinese history and civilization. However, due to the lengthy deterioration process for about 2000 years, inscriptions on bamboo slips typically suffer from ink fading problem, which seriously impacts their clarity and legibility. This paper proposes BambInk, which is among the first technical initiatives that explore AI-enabled ink enhancement for bamboo slips with faded inks. BambInk is a new self-supervised learning method that devises an iterative restoration-degradation adversarial learning mechanism to progressively enhance the faded inks on the bamboo slips, yet without requiring any annotated data. In specific, BambInk designs two generators which are ink enhancer and ink degrader, and two discriminators for evaluate the effects of the enhanced and degraded images, respectively. Moreover, in the generators, it introduces a dynamic convolution mechanism  that integrates features captured via three heterogeneous types of attentions; it also adopts a simple yet effective high-low frequency information differentiation mechanism for extracting the fine-details of the ink traces. Experiments conducted on the real-world bamboo slips dataset demonstrate the effectiveness of our method. Code, data and the enhanced results are available at: https://anonymous.4open.science/r/BambInk-46E5.

## Overall
<p align="center">
  <img src="imgs/Overall.png"/>
</p>

## Architecture
<p align="center">
  <img src="imgs/Architecture.png" width="80%"/>
</p>

## Using the code:
The code is stable while using Python 3.8.3, PyTorch 1.13.0, CUDA =11.4.

## Data Presentation and Model Parameters
- We provide a selection of results, including the original inputs and the enhanced model outputs, at the [link](https://drive.google.com/drive/folders/1-8aSNbFd5BKs0ZmNHY0dBxWj2PXGB9ea?usp=drive_link).
- Furthermore, we have provided the optimal model [parameters](https://drive.google.com/drive/folders/1GvsHvcw-RwCF_Pi9iiPidcIzCva0frw7?usp=drive_link).

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

## Comparison with other methods
<p align="center">
  <img src="imgs/Comparison.png" width="60%"/>
</p>

## Visual comparison
<p align="center">
  <img src="imgs/Visual result.png"/>
</p>
