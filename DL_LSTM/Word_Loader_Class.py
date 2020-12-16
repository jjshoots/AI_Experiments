import unicodedata
import string
import torch
from torch.utils.data import Dataset
import numpy as np

class WordLoader(Dataset):
    def __init__(self, text, sequence_length, DICTIONARY_PATH, build_dictionary=True):
        self.sequence_length = sequence_length
        self.all_words = []
        self.n_words = 0

        text = text.split(" ")

        if build_dictionary:
            for i in text:
                try:
                    self.all_words.index(i)
                except ValueError:
                    self.all_words.append(i)
                    self.n_words += 1

            dictionary_file = open(DICTIONARY_PATH, "w", encoding="utf-8")
            for word in self.all_words:
                dictionary_file.write(word)
                dictionary_file.write("\n")
            dictionary_file.close()
        else:
            dictionary_file = open(DICTIONARY_PATH, "r", encoding="utf-8")
            self.all_words = dictionary_file.read().splitlines()
            self.n_words = len(self.all_words)

        self.text_length = len(text)
        self.text_onehot, self.text_digits = self.textToTensors(text)

    def __len__(self):
        return self.text_length - self.sequence_length - 2

    def __getitem__(self, idx):
        data = self.text_onehot[idx:idx + self.sequence_length]
        label = self.text_digits[idx+1:idx + self.sequence_length+1]
        return data, label

    def return_all_words(self):
        return self.all_words

    def return_n_words(self):
        return self.n_words

    def wordToIndex(self, word):
        return self.all_words.index(word)

    def textToTensors(self, text):
        text_onehot = torch.zeros(self.text_length, self.n_words)
        text_digits = torch.zeros(self.text_length, dtype=torch.long)
        for i, word in enumerate(text):
            word_index = self.wordToIndex(word)
            text_onehot[i][word_index] = 1
            text_digits[i] = word_index

        return text_onehot, text_digits

    def indexToWord(self, index):
        return self.all_words[index]






