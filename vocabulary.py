from collections import Counter

import spacy
from pycocotools.coco import COCO
from tqdm import tqdm


class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.len = 0

        self.pad = self.add_word("<PAD>")
        self.unk = self.add_word("<UNK>")
        self.sos = self.add_word("<SOS>")
        self.eos = self.add_word("<EOS>")

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.len
            self.idx2word[self.len] = word
            self.len += 1

        return self.word2idx[word]

    def __call__(self, word):
        return self.word2idx.get(word, self.unk)

    def __getitem__(self, idx):
        return self.idx2word.get(idx)

    def __len__(self):
        return self.len


class CocoCaptionsVocabulary(Vocabulary):
    def __init__(self, annotation_file, freq_threshold=5):
        super().__init__()

        self.coco = COCO(annotation_file)
        self.nlp = spacy.load("en_core_web_sm")
        self.freq_threshold = freq_threshold
        self.__build_vocab()

    def tokenize(self, text):
        return [tok.text.lower() for tok in self.nlp.tokenizer(text)]

    def __build_vocab(self):
        counter = Counter()

        for value in tqdm(self.coco.anns.values()):
            tokens = self.tokenize(value["caption"])
            counter.update(tokens)

        for word, cnt in counter.items():
            if cnt >= self.freq_threshold:
                # ? If count of word in dataset is greater than freq_threshold, add it to vocab.
                self.add_word(word)

        print(
            f"Selecting words with >= {self.freq_threshold} appearances ie "
            f"{self.len} words of {len(counter)} total."
        )

    def encode(self, text):
        encoded_text = [self(token) for token in self.tokenize(text)]
        encoded_text = [self.sos] + encoded_text + [self.eos]

        return encoded_text
