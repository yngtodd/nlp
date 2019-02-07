from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy

from autokeras import CnnModule
from autokeras.nn.metric import Accuracy
from nas.greedy import GreedySearcher