import torch
import torch.nn as nn


class Config:
    """
    Configuration for CNN.
    """
    def __init__(self):
        self.kernel1 = 3
        self.kernel2 = 4
        self.kernel3 = 5
        self.n_filters1 = 100
        self.n_filters2 = 100
        self.n_filters3 = 100
        self.dropout1 = 0.5
        self.dropout2 = 0.5
        self.dropout3 = 0.5
        self.max_sent_len = 3000
        self.word_dim = 300
        self.vocab_size = 35095


def conv_block(n_filters, kernel_size, dropout):
    return nn.Sequential(
        nn.Conv1d(1, n_filters, kernel_size),
        nn.ReLU(),
        nn.AdaptiveMaxPool1d(1),
        nn.Dropout(p=dropout)
    )



class CNN(nn.Module):
    """CNN model for document classification.

    Parameters
    ----------
     n_classes : int
       Number of classes to predict.

     alt_model_type : str, default=None
       Alternative type of model being used.
       -Options:
           "static"
           "multichannel"
    """
    def __init__(self, config=Config(), wv_matrix=None, n_classes=10, alt_model_type=None):
        super(CNN, self).__init__()
        self.config = config
        self.wv_matrix = wv_matrix
        self.n_classes = n_classes
        self.alt_model_type = alt_model_type
        self._filter_sum = None
        self._sum_filters()

        self.embedding = nn.Embedding(self.config.vocab_size + 2, self.config.word_dim, padding_idx=0)

        if self.alt_model_type == 'static':
            self.embedding.weight.requires_grad = False
        elif self.alt_model_type == 'multichannel':
            self.embedding2 = nn.Embedding(self.config.vocab_size + 2, self.config.word_dim, padding_idx=self.config.vocab_size + 1)
            self.embedding2.weight.data.copy_(torch.from_numpy(self.wv_matrix))
            self.embedding2.weight.requires_grad = False
            self.IN_CHANNEL = 2

        self.features = nn.Sequential()
        self.features.add_module(
            'conv1',
            conv_block(
                self.config.n_filters1,
                self.config.kernel1,
                self.config.dropout1
             )
         )

        self.features.add_module(
            'conv2',
            conv_block(
                self.config.n_filters2,
                self.config.kernel2,
                self.config.dropout2
           )
        )

        self.features.add_module(
            'conv3',
            conv_block(
                self.config.n_filters3,
                self.config.kernel3,
                self.config.dropout3
            )
        )

        self.fc = nn.Linear(self._filter_sum, self.n_classes)

    def _sum_filters(self):
        """Get the total number of convolutional filters."""
        self._filter_sum = self.config.n_filters1 + self.config.n_filters2 + self.config.n_filters3

    def forward(self, x):
        x = self.embedding(x).view(-1, 1, self.config.word_dim * self.config.max_sent_len)
        if self.alt_model_type == "multichannel":
            x2 = self.embedding2(x).view(-1, 1, self.config.word_dim * self.config.max_sent_len)
            x = torch.cat((x, x2), 1)

        conv_results = []
        conv_results.append(self.features.conv1(x).view(-1, self.config.n_filters1))
        conv_results.append(self.features.conv2(x).view(-1, self.config.n_filters2))
        conv_results.append(self.features.conv3(x).view(-1, self.config.n_filters3))
        x = torch.cat(conv_results, 1)
        out = self.fc(x)
        return out
