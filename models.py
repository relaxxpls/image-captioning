import torch
import torch.nn as nn
import torchvision.models as models

from vocabulary import CocoCaptionsVocabulary


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()

        # * As we are not using auxiliary loss, we set aux_logits to False.
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)

        # ? Freeze model parameters except those of the last layer.
        for name, param in self.inception.named_parameters():
            if name in ["fc.weight", "fc.bias"]:
                param.requires_grad = True
            param.requires_grad = False

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images)

        return self.dropout(self.relu(features))


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, 1)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = self.dropout(embeddings)
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)

        return outputs


class EncoderDecoderModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()

        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)

        return outputs

    @torch.no_grad()
    def get_caption(
        self, image: torch.Tensor, vocabulary: CocoCaptionsVocabulary, max_length=50
    ):
        result = []
        x = self.encoderCNN(image).unsqueeze(0)
        states = None

        for _ in range(max_length):
            hiddens, states = self.decoderRNN.lstm(x, states)
            output = self.decoderRNN.linear(hiddens.squeeze(0))
            predicted = output.argmax(1)
            x = self.decoderRNN.embed(predicted).unsqueeze(0)

            predicted_word = vocabulary.idx2word[predicted.item()]
            result.append(predicted_word)
            if predicted_word == "<EOS>":
                break

        return result
