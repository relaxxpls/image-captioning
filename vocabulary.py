from collections import Counter

import spacy
from pycocotools.coco import COCO
from tqdm import tqdm


class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.len = 0

        self.add_word("<PAD>")
        self.add_word("<SOS>")
        self.add_word("<EOS>")
        self.add_word("<UNK>")

    def add_word(self, word):
        if word in self.word2idx:
            return

        self.word2idx[word] = self.len
        self.idx2word[self.len] = word
        self.len += 1

    def __call__(self, word):
        if word not in self.word2idx:
            word = "<UNK>"

        return self.word2idx[word]

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

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)

        return [
            self.word2idx[token] if token in self.word2idx else self.word2idx["<UNK>"]
            for token in tokenized_text
        ]
