import argparse
import numpy as np

import torch
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
    args = parser.parse_args()

    train = Synthetic(args.datapath, 'train', target=0, download=True)
    test = Synthetic(args.datapath, 'test', target=0, download=True)

    train.data = train.data.astype(float)
    test.data = test.data.astype(float)

    x, y = train.load_data()
    input_shape = np.expand_dims(x, axis=0).shape
    num_classes = np.max(y) + 1


    trainloader = DataLoader(train, batch_size=16)
    testloader = DataLoader(test, batch_size=16)

    cnnModule = CnnModule(
        loss=cross_entropy,
        metric=Accuracy,
        searcher_args={},
        verbose=True,
        search_type=GreedySearcher
    )

    cnnModule.fit(
        n_output_node=num_classes,
        input_shape=input_shape,
        train_data=trainloader,
        test_data=testloader
    )


if __name__=='__main__':
    main()
