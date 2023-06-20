import random
import argparse
import torch
from dataset import Dataset
from model import MILModel


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint')


if __name__ == "__main__":
    args = parser.parse_args()
    dataset = Dataset(train = False)
    
    state = torch.load(args.checkpoint)
    model = MILModel(janossy_samples = 4)
    model.load_state_dict(state)
    model.eval().requires_grad_(False).cuda()
    
    idx = random.randint(0, dataset.mnist.targets.shape[0] - 1)
    img, label = dataset.mnist[idx]
    _, _, inst_prob = model(img[None,...].cuda())
    p = float(inst_prob.squeeze().detach().cpu())
    print(f"Instance Prediction: Image #{idx} - {label}. Is 7: {inst_prob:.2f}")
    
    bag, target, _ = dataset[0]
    _, bag_prob, _ = model(bag.cuda())
    p = float(bag_prob.squeeze().detach().cpu())
    print(f"Bag Prediction: Label - {target}, Prediction - {bag_prob:.2f}")
    