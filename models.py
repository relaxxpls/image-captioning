import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()

        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.embed.weight.data.normal_(0.0, 0.02)
        self.embed.bias.data.fill_(0)
        self.batch = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.batch(self.embed(features))

        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.drop_prob = 0.2
        self.vocabulary_size = vocab_size
        self.lstm = nn.LSTM(
            self.embed_size, self.hidden_size, self.num_layers, batch_first=True
        )
        self.dropout = nn.Dropout(self.drop_prob)

        self.embed = nn.Embedding(self.vocabulary_size, self.embed_size)
        self.embed.weight.data.uniform_(-0.1, 0.1)

        self.linear = nn.Linear(hidden_size, self.vocabulary_size)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        features = features.unsqueeze(1)
        embeddings = torch.cat((features, embeddings[:, :-1, :]), dim=1)
        hiddens, c = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs
