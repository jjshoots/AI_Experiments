import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas
import numpy as np
import sys
import matplotlib.pyplot as plt
import os.path

from LSTM_Class import LSTM
from Word_Loader_Class import WordLoader as WL

HIDDEN_DIM = 5000
SEQUENCE_LENGTH = 200
BATCH_SIZE = 3
SAVE_PATH = './lstm_net_word_v1.path'
RAW_DATA_PATH = './BOOK.txt'
FIX_DATA_PATH = './BOOK_FIXED.txt'
DICTIONARY_PATH = './DICTIONARY.txt'
OUTPUT_PATH = './OUTPUT.txt'
IS_TRAINING = True

if not IS_TRAINING:
    torch.no_grad()


##########################################################
# READ THE FILE
# modify the raw data (only do once)
if 0:
    # prune the file
    raw_file = open(RAW_DATA_PATH, "r", encoding="utf-8")
    fix_file = open(FIX_DATA_PATH, "w", encoding="utf-8")

    text = raw_file.read().lower()
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace("’s", "'s")
    text = text.replace("-", "")
    text = text.replace(";", " . ")
    text = text.replace(":", " . ")
    text = text.replace("!", " . ")
    text = text.replace("?", " . ")
    text = text.replace(".", " . ")
    text = text.replace(",", " , ")
    text = text.replace("”", '')
    text = text.replace("“", '')
    text = text.replace("\\", '')

    text = text.replace("  ", " ")
    text = text.replace("  ", " ")
    text = text.replace("  ", " ")
    text = text.replace("  ", " ")

    text = text.replace("\n", "")
    text = text.replace(" . . .", "")

    fix_file.write(text)
    fix_file.close()


# start feeding the data into the dataloader
fix_file = open(FIX_DATA_PATH, "r", encoding="utf-8")
text = fix_file.read()
dataset = WL(text, SEQUENCE_LENGTH, DICTIONARY_PATH, build_dictionary=False)
print(dataset.text_length)

if IS_TRAINING:
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

if 0:
    dataiter = iter(dataloader)
    user_input = None
    while user_input != 'Y':
        user_input = input('Key in "Y" to end display, enter to continue...')
        data, label = dataiter.next()

        for index in label[0]:
            print(dataset.indexToWord(index))


##########################################################
# DEFINE NET
# Define the net
LSTM_net = LSTM(dataset.return_n_words(), HIDDEN_DIM, BATCH_SIZE)

# check if net exists
if os.path.isfile(SAVE_PATH):
    LSTM_net.load_state_dict(torch.load(SAVE_PATH))

print('Saving net parameters to ' + SAVE_PATH + ' every epoch')

# select device
device = 'cpu'
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    LSTM_net.to(device)
print('USING DEVICE', device)

# loss function and optimizer function and hidden cells
loss_function = nn.CrossEntropyLoss()
optim_function = optim.SGD(LSTM_net.parameters(), lr=0.001, momentum=0.9)
hidden_state = LSTM_net.initHiddenLayer(test=False)[0].to(device), LSTM_net.initHiddenLayer(test=False)[1].to(device)



##########################################################
# TRAIN!
if IS_TRAINING:
    for epoch in range(2000):

        running_loss = 0.

        for i, data in enumerate(dataloader):

            LSTM_net.zero_grad()

            data, labels = data[0].to(device), data[1].to(device)
            hidden = hidden_state[0], hidden_state[1]

            output, hidden = LSTM_net.forward(data, hidden)
            output.transpose_(1, 2)
            loss = loss_function(output, labels)
            loss.backward()
            optim_function.step()
            running_loss += loss

            if i % 100 == 0:
                print('Epoch ' + str(epoch) + '; Batch Number '+ str(i) + '; Running Loss '+ str(running_loss.item()))
                running_loss = 0.

                torch.save(LSTM_net.state_dict(), SAVE_PATH)

    print('Done Training')


    ##########################################################
    # SAVE THE NET
    torch.save(LSTM_net.state_dict(), SAVE_PATH)
    print("Saving net to " + SAVE_PATH)



##########################################################
# TEST

if not IS_TRAINING:

    output_file = open(OUTPUT_PATH, "a", encoding="utf-8")

    hidden = LSTM_net.initHiddenLayer(test=True)[0].to(device), LSTM_net.initHiddenLayer(test=True)[1].to(device)

    starting_line = 'hell'
    starting_tensor = dataset.stringToTensors(starting_line)

    for i, letter_tensor in enumerate(starting_tensor):
        output, hidden = LSTM_net.forward(letter_tensor.unsqueeze(0).unsqueeze(0).to(device), hidden)
        print(starting_line[i])
        output_file.write(starting_line[i])

    output_file.close()

    user_input = None
    while user_input != 'Y':
        user_input = input()
        output_file = open(OUTPUT_PATH, "a", encoding="utf-8")

        for i in range(1000):
            output, hidden = LSTM_net.forward(output, hidden)

            letter_index = torch.argmax(output.squeeze())

            print(dataset.return_all_letters()[letter_index], end='')

            output_file.write(dataset.return_all_letters()[letter_index])

        output_file.close()
