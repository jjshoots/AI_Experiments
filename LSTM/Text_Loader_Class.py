import unicodedata
import string
import torch
from torch.utils.data import Dataset
import numpy as np

class TextLoader(Dataset):
    def __init__(self, text, sequence_length):
        self.sequence_length = sequence_length
        self.all_letters = []
        self.n_letters = 0

        for i in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890' + text:
            try:
                self.all_letters.index(i)
            except ValueError:
                self.all_letters.append(i)
                self.n_letters += 1

        self.text_length = len(text)
        self.text_onehot, self.text_digits = self.textToTensors(text)

    def __len__(self):
        return self.text_length - self.sequence_length - 2

    def __getitem__(self, idx):
        data = self.text_onehot[idx:idx + self.sequence_length]
        label = self.text_digits[idx+1:idx + self.sequence_length+1]
        return data, label

    def return_all_letters(self):
        return self.all_letters

    def return_n_letters(self):
        return self.n_letters

    def letterToIndex(self, letter):
        return self.all_letters.index(letter)

    def textToTensors(self, text):
        text_onehot = torch.zeros(self.text_length, self.n_letters)
        text_digits = torch.zeros(self.text_length, dtype=torch.long)
        for i, letter in enumerate(text):
            letter_index = self.letterToIndex(letter)
            text_onehot[i][letter_index] = 1
            text_digits[i] = letter_index

        return text_onehot, text_digits

    def stringToTensors(self, text):
        text_onehot = torch.zeros(len(text), self.n_letters)
        for i, letter in enumerate(text):
            letter_index = self.letterToIndex(letter)
            text_onehot[i][letter_index] = 1

        return text_onehot




