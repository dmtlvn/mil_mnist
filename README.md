## Setup

```pip install -r requirements.txt```

## Train: 

```python train.py --batch-size 4 --train-steps 1000 --val-steps 1000 --epochs-no-gain 10 --seed 42```

## Inference Example:

```python inference.py --checkpoint <path>```

Trained model is provided at: `weights/ckpt_0_991.pth`
