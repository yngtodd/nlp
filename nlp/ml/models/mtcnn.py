import torch
import torch.nn as nn


class Config:

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


class MTCNN(nn.Module):
    """Multi-task CNN model for document classification.

    Parameters
    ----------
    subsite_size : int
      Class size for subsite task.

     laterality_size : int
      Class size for laterality task.

     behavior_size : int
       Class size for behavior task.

     grade_size : int
       Class size for grade task.

     alt_model_type : str, default=None
       Alternative type of model being used.
       -Options:
           "static"
           "multichannel"
    """
    def __init__(self, config=Config(), wv_matrix=None, subsite_size=34, laterality_size=4,
                 behavior_size=3, histology_size=44, grade_size=5, alt_model_type=None):
        super(MTCNN, self).__init__()
        self.wv_matrix = wv_matrix
        self.subsite_size = subsite_size
        self.laterality_size = laterality_size
        self.behavior_size = behavior_size
        self.histology_size = histology_size
        self.grade_size = grade_size
        self.alt_model_type = alt_model_type
        self._filter_sum = None
        self._sum_filters()

        self.embedding = nn.Embedding(self.vocab_size + 2, self.word_dim, padding_idx=0)

        if self.alt_model_type == 'static':
            self.embedding.weight.requires_grad = False
        elif self.alt_model_type == 'multichannel':
            self.embedding2 = nn.Embedding(self.config.vocab_size + 2, self.config.word_dim, padding_idx=self.config.vocab_size + 1)
            self.embedding2.weight.data.copy_(torch.from_numpy(self.wv_matrix))
            self.embedding2.weight.requires_grad = False
            self.IN_CHANNEL = 2

        self.convblock1 = nn.Sequential(
            nn.Conv1d(1, self.config.num_filters1, self.config.kernel1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Dropout(p=self.config.dropout1)
        )

        self.convblock2 = nn.Sequential(
            nn.Conv1d(1, self.config.num_filters2, self.config.kernel2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Dropout(p=self.config.dropout2)
        )

        self.convblock3 = nn.Sequential(
            nn.Conv1d(1, self.config.num_filters3, self.config.kernel3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Dropout(p=self.config.dropout3)
        )

        self.fc1 = nn.Linear(self._filter_sum, self.subsite_size)
        self.fc2 = nn.Linear(self._filter_sum, self.laterality_size)
        self.fc3 = nn.Linear(self._filter_sum, self.behavior_size)
        self.fc4 = nn.Linear(self._filter_sum, self.histology_size)
        self.fc5 = nn.Linear(self._filter_sum, self.grade_size)

    def _sum_filters(self):
        """Get the total number of convolutional filters."""
        self._filter_sum = self.num_filters1 + self.num_filters2 + self.num_filters3

    def forward(self, x):
        x = self.embedding(x).view(-1, 1, self.word_dim * self.max_sent_len)
        if self.alt_model_type == "multichannel":
            x2 = self.embedding2(x).view(-1, 1, self.word_dim * self.max_sent_len)
            x = torch.cat((x, x2), 1)

        conv_results = []
        conv_results.append(self.convblock1(x).view(-1, self.num_filters1))
        conv_results.append(self.convblock2(x).view(-1, self.num_filters2))
        conv_results.append(self.convblock3(x).view(-1, self.num_filters3))
        x = torch.cat(conv_results, 1)

        out_subsite = self.fc1(x)
        out_laterality = self.fc2(x)
        out_behavior = self.fc3(x)
        out_histology = self.fc4(x)
        out_grade = self.fc5(x)
        return out_subsite, out_laterality, out_behavior, out_histology, out_grade
