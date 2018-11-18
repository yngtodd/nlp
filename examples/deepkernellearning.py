import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import argparse
import numpy as np

import gpytorch
from gpytorch import settings
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import SoftmaxLikelihood

from nlp.data import Synthetic
from nlp.ml.models import ConvFeatureExtractor
from nlp.ml.models import DKLModel


def train(epoch, train_loader, optimizer, likelihood, model, device):
    model.train()
    likelihood.train()

    mll = VariationalELBO(likelihood, model.gp_layer, num_data=len(train_loader.dataset))

    train_loss = 0.
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = -mll(output, target)
        loss.backward()
        optimizer.step()
        if (idx + 1) % 25 == 0:
            current_loss = loss.item()
            print(f'Epoch: {epoch} [{idx+1}/{len(train_loader)}], Loss: {current_loss:.6f}')


def test(test_loader, likelihood, model, device):
    model.eval()
    likelihood.eval()

    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = likelihood(model(data))
            pred = output.probs.argmax(1)
            correct += pred.eq(target.view_as(pred)).cpu().sum()

    percent_correct = 100. * correct / float(len(test_loader.dataset))
    print(f'Test set: Accuracy: {correct}/{len(test_loader.dataset)} ({percent_correct}%)')


def main():
    parser = argparse.ArgumentParser(description='Deep Kernel Learning with synthetic data.')
    parser.add_argument('--datapath', type=str, help='Path to data directory.')
    parser.add_argument('--batchsize', type=int, default=10, help='Batch size.')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=0.1, help='Path to data directory.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:2" if use_cuda else "cpu")

    traindata = Synthetic(args.datapath, 'train', download=True)
    train_loader = DataLoader(traindata, batch_size=args.batchsize)
    num_classes = len(np.unique(traindata.targets))

    testdata = Synthetic(args.datapath, 'test')
    test_loader = DataLoader(testdata, batch_size=args.batchsize)

    feature_extractor = ConvFeatureExtractor().to(device)
    num_features = feature_extractor._filter_sum

    model = DKLModel(feature_extractor, num_dim=num_features).to(device)
    likelihood = SoftmaxLikelihood(num_features=model.num_dim, n_classes=num_classes).to(device)

    optimizer = SGD([
        {'params': model.feature_extractor.parameters()},
        {'params': model.gp_layer.hyperparameters(), 'lr': args.lr * 0.01},
        {'params': model.gp_layer.variational_parameters()},
        {'params': likelihood.parameters()},
    ], lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0)

    scheduler = MultiStepLR(
        optimizer,
        milestones=[0.5 * args.n_epochs, 0.75 * args.n_epochs],
        gamma=0.1
    )

    for epoch in range(1, args.n_epochs+1):
        scheduler.step()
        with settings.use_toeplitz(False), settings.max_preconditioner_size(0):
            train(epoch, train_loader, optimizer, likelihood, model, device)
            test(test_loader, likelihood, model, device)

        state_dict = model.state_dict()
        likelihood_state_dict = likelihood.state_dict()
        torch.save(
            {'model': state_dict,
            'likelihood': likelihood_state_dict},
            'dkl_synthetic_checkpoint.dat'
        )


if __name__=='__main__':
    main()

