import argparse

from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy

from autokeras import CnnModule
from nas.greedy import GreedySearcher
from autokeras.nn.metric import Accuracy

from nlp.data import Synthetic


def main():
    parser = argparse.ArgumentParser(description='Deep Kernel Learning with synthetic data.')
    parser.add_argument('--datapath', type=str, help='Path to data directory.')
    parser.add_argument('--batchsize', type=int, default=10, help='Batch size.')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=0.1, help='Path to data directory.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    args = parser.parse_args()

    train = Synthetic(args.datapath, 'train', download=True)
    test = Synthetic(args.datapath, 'test', download=True)

    train_loader = DataLoader(train, batch_size=16)
    test_loader = DataLoader(test, batch_size=16)

    