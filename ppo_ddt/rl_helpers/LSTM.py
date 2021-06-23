import torch
import torch.nn as nn
import torch.optim as optim
from torchnlp.encoders.text import StaticTokenizerEncoder, pad_tensor
import torch
# from tokenizers import BertWordPieceTokenizer
# from tokenizers.models import BPE
# from tokenizers.pre_tokenizers import Whitespace
# from tokenizers.trainers import BpeTrainer
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from nltk.corpus import stopwords
# STOPWORDS = set(stopwords.words('english'))


# LSTM 
class LSTMNet(nn.Module):
    #define all the layers used in model
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 use_embeds,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):
        # Constructor
        super(LSTMNet, self).__init__()
        # embedding layer
        self.use_embeds = use_embeds
        if not use_embeds:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # lstm layer
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)
        # dense layer
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        # activation function
        self.act = nn.Sigmoid()
        
    def forward(self, embedded):
        if not self.use_embeds:
            embedded = self.embedding(embedded)

        packed_output, (hidden, cell) = self.lstm(embedded)  # packed_embedded)
        # concat the final forward and backward hidden state??
        hidden = torch.cat((hidden[0, :, :], hidden[-1, :, :]), dim=1)
        outputs = self.fc(hidden)
        return outputs


# train model
def train(net,
          optimizer,
          criterion,
          X_train,
          y_train,
          epochs=1):
    for epoch in range(epochs):  # loop over the dataset multiple times

        # running_loss = 0.0
        for i in range(len(y_train)):
            inputs = X_train[i]
            inputs = inputs.unsqueeze(0) # adds dim to front (no batch)
            labels = y_train[i]

            if type(labels) == int:
                labels = torch.tensor([labels])  # is this correct?

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels) # ValueError
            loss.backward()
            optimizer.step()

            # print statistics
            # running_loss += loss.item()
            # if i % 2000 == 1999: 
            #     print('[%d, %5d] loss: %.3f' %
            #         (epoch + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.0
    return net


# split input data
def split_data(x_data,
               y_data,
               train_proportion):

    test_division = round(len(y_data) * train_proportion)  # amount of samples in training vs testing data

    X_train = x_data[0:test_division]
    X_test = x_data[test_division:]

    y_train = y_data[0:test_division]
    y_test = y_data[test_division:]

    return (X_train, y_train), (X_test, y_test)


def main(x_data, y_data, model, train_proportion=0.8, train_epochs=1):
    """
    train and test model
    Args:
        x_data: samples in
        y_data: labels in
        model: network to train and test
        train_proportion: proportion of data to be in training vs testing
        train_epochs: how many epochs to train?

    Returns:

    """
    ### preprocess data ###

    # get data (words)
    x_data = [x.lower() for x in x_data]
    all_words = ' '.join(x_data).split()
    (X_train, y_train), (X_test, y_test) = split_data(x_data, y_data, train_proportion=train_proportion)

    # tokenizers
    tokenizer = StaticTokenizerEncoder(x_data)
    # tokenizer = BertWordPieceTokenizer(lowercase=True, clean_text=True)
    # tokenizer.train(
    #     files='../../pretrain_texts/gridworld_commands.txt',
    #     special_tokens=["[CLS]", "[UNK]", "[SEP]", "[PAD]", "[MASK]", "[EOS]", "[SOS]"],
    #     vocab_size=len(set(all_words)))
    # labels = ["RED", "GREEN", "PINK", "BLUE"]
    # label_dict = {"RED": 0, "GREEN":1, "PINK": 2, "BLUE": 3}

    # tokenize sequences
    train_sequences = [tokenizer.encode(x) for x in X_train]
    validation_sequences = [tokenizer.encode(x) for x in X_test]
    
    # sentence padding
    train_padded = [pad_tensor(x, length=max_length) for x in train_sequences]
    validation_padded = [pad_tensor(x, length=max_length) for x in validation_sequences]

    # tokenize goals
    # training_label_seq = [label_dict[x] for x in y_train]
    # validation_label_seq = [label_dict[x] for x in y_test]
    training_label_seq = y_train
    validation_label_seq = y_test
    ### train model ### 


    # optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, eps=1e-08)

    train(model, optimizer, criterion, train_padded, training_label_seq, train_epochs)

    ### test model ###
    validation_padded = torch.stack(validation_padded)
    pred = model(validation_padded)

    # prediction decoding
    print("\nLSTM predictions")
    print("----------------")
    guess = []
    for x in pred:
        goal_guessed = torch.argmax(x).item()
        guess.append(goal_guessed) 
        print(x)
    # accuracy
    print("\nlabel \t guess")
    print("--------------")
    accuracy = 0
    for i in range(len(guess)):
        print(validation_label_seq[i], "\t", guess[i], "\t", guess[i] == validation_label_seq[i])
        if guess[i] == validation_label_seq[i]:
            accuracy += 1

    # accuracy /= len(guess)*100
    print("\naccuracy: {0}/{1}\n".format(accuracy, len(guess)))


if __name__ == "__main__":
    ### hyperparams ###
    vocab_size = 50
    max_length = 15  # padding length
    dropout = 0.2
    num_layers = 2
    embedding_dim = 64
    hidden_dim = embedding_dim
    output_dim = 4
    epochs = 25
    train_perc = 6.5 / 12  # fraction to get a 14:16 division of train and test data

    # Full data set
    # X_dataset = [
    #     # training data
    #
    #     # red
    #     "Go to red",  # .split(),
    #     "Go to the red",  # .split(),
    #     "Move to south west",  # .split(),
    #     "Go to bottom left",  # .split(),
    #
    #     # green
    #     "Go to green",  # .split(),
    #     "Go to the green",  # .split(),
    #     "Navigate to south east",  # .split(),
    #     "Go to lower right",  # .split(),
    #
    #     # pink
    #     "Go to pink",  # .split(),
    #     "Turn to upper west",  # .split(),
    #     "Go to north western corner",  # .split(),
    #
    #     # blue
    #     "Go to blue",  # .split(),
    #     "Redirect to north east",  # .split(),
    #     "Go to top eastern box",  # .split(),
    #
    #     # test
    #     "Move to south left",  # .split(),               # red
    #     "Go to lower east",  # .split(),                 # green
    #     "Turn to the pink",  # .split(),                 # pink
    #     "Go to north eastern",  # .split(),              # blue
    #
    #     "Redirect to blue",  # .split(),                 # blue
    #     "Go to the upper western corner",  # .split(),   # pink
    #     "Go to the south right",  # .split(),            # green
    #     "Move to red",  # .split(),                      # red
    #
    #     "Turn to green box",  # .split(),                # green
    #     "Go to bottom west",  # .split(),                # red
    #     "Navigate to pink",  # .split(),                 # pink
    #     "Go to the top east corner"]  # .split()]        # blue
    #
    #
    # y_dataset = [
    #     # training data
    #     "RED", "RED", "RED", "RED",
    #     "GREEN", "GREEN", "GREEN", "GREEN",
    #     "PINK", "PINK", "PINK",
    #     "BLUE", "BLUE", "BLUE",
    #
    #     # test
    #     "RED", "GREEN", "PINK", "BLUE",
    #     "BLUE", "PINK", "GREEN", "RED",
    #     "GREEN", "RED", "PINK", "BLUE"]
    x_dataset = []
    y_dataset = []
    with open('../../pretrain_texts/gridworld_commands.txt', 'r') as f:
        for line in f.readlines():
            split_line = line.split(': ')
            y_dataset.append(int(split_line[0]))
            command = split_line[1].strip('\n')
            command = command.lower()
            x_dataset.append(command)
    lstm_net = LSTMNet(vocab_size, embedding_dim, hidden_dim=embedding_dim, output_dim=output_dim, n_layers=num_layers,
                       bidirectional=True, dropout=dropout)
    main(x_dataset, y_dataset, model=lstm_net, train_proportion=train_perc, train_epochs=epochs)
