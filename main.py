import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from collections import Counter
from rnn import RNN, LSTM, GRU
import numpy as np
import argparse


class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_classes=1, bidirectional=False, padding_idx=0):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.bridge = nn.Linear(embedding_dim, embedding_dim)
        self.rnn = RNN(embedding_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        h = None
        x = x.transpose(1, 0)
        x = F.relu(self.embedding(x))
        x = self.bridge(x)
        x = self.rnn(x)
        x = torch.sigmoid(self.out(x.squeeze(0)))
        return x


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_classes=1, bidirectional=False, padding_idx=0,
                 n_layers=1, dropout=0.2):
        super(SentimentLSTM, self).__init__()
        self.bridge = nn.Linear(embedding_dim, embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.rnn = LSTM(embedding_dim,
                        hidden_dim,
                        )
        self.out = nn.Linear(hidden_dim * n_layers, n_classes)
        self.n_layers = n_layers

    def forward(self, x):
        x = x.transpose(1, 0)
        x = F.dropout(self.embedding(x))
        x = self.bridge(x)
        x = self.rnn(x)

        x = self.out(x)
        return x


class SentimentGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_classes=1, bidirectional=False, padding_idx=0,
                 n_layers=1, dropout=0.2):
        super(SentimentGRU, self).__init__()
        self.bridge = nn.Linear(embedding_dim, embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.rnn = GRU(embedding_dim,
                       hidden_dim,
                       )
        self.out = nn.Linear(hidden_dim * n_layers, n_classes)
        self.n_layers = n_layers

    def forward(self, x):
        x = x.transpose(1, 0)
        x = F.dropout(self.embedding(x))
        x = self.bridge(x)
        x = self.rnn(x)

        x = self.out(x)
        return x


class Data:
    def __init__(self, texts, labels, vocab=None, word2id=None):
        self.sentences = texts
        self.labels = labels
        self.vocab = [] if vocab is None else vocab
        self.word2id = {"PAD": 0, "UNK": 1} if word2id is None else word2id
        self.nummerized_sentences = []

    def sort(self):
        lens = [len(item) for item in self.nummerized_sentences]
        indices = np.argsort(lens)
        new_sent_list = []
        new_label = []
        for i in indices:
            new_sent_list.append(self.nummerized_sentences[i])
            new_label.append(self.labels[i])
        self.nummerized_sentences = new_sent_list
        self.labels = new_label

    def build_vocab(self, train_set=True, sentences=None):
        if train_set == True:  # build vocab
            vocab_freq = Counter()
            for sentence in self.sentences:
                # sentence = sentence.split()
                vocab_freq.update(sentence)
            for w, f in vocab_freq.items():
                if f >= 2:
                    self.vocab.append(w)
            # self.vocab = list(vocab_freq.keys())
            for i, word in enumerate(self.vocab):
                self.word2id[word] = i

        for sentence in self.sentences:
            sent = []
            for w in sentence:
                if w in self.vocab:
                    sent.append(self.word2id[w])
                else:
                    sent.append((self.word2id["UNK"]))
            self.nummerized_sentences.append(sent)
        self.sort()

    def nummerize(self, texts):
        # nummerized_sentences = []
        for sentence in texts:
            sent = []
            for w in sentence:
                if w in self.vocab:
                    sent.append(self.word2id[w])
                else:
                    sent.append((self.word2id["UNK"]))
            self.nummerized_sentences.append(sent)
        # return nummerized_sentences

    def get_vocab_size(self):
        return len(self.vocab)

    def get_batchs(self, batch_size=32):
        n_batchs = len(self.nummerized_sentences) // batch_size
        from_ = 0
        to_ = batch_size
        batches = []
        for i in range(0, n_batchs):
            batch = self.next(from_, to_)
            from_ = to_
            to_ = to_ + batch_size
            batches.append(batch)
        return batches

    def get_legnths(self, value):
        return [len(x) for x in value]

    def __padding(self, value, batch_size):
        max_len = max([len(x) for x in value])
        matrix = torch.zeros(size=(batch_size, max_len), dtype=torch.long)
        for i, x in enumerate(value):
            if len(x) == max_len:
                matrix[i] = torch.tensor(x, dtype=torch.long)
            else:
                matrix[i][:len(x)] = torch.tensor(x, dtype=torch.long)
        return matrix, self.get_legnths(value)

    def next(self, from_, to_):
        batch_size = to_ - from_
        sent_batch, lengths = self.__padding(self.nummerized_sentences[from_: to_], batch_size)
        labels = torch.tensor(self.labels[from_: to_], dtype=torch.float)
        return sent_batch, lengths, labels


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch[0]).squeeze(1)
        loss = criterion(predictions, batch[2])

        acc = binary_accuracy(predictions, batch[2])

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch[0]).squeeze(1)

            loss = criterion(predictions, batch[2])

            acc = binary_accuracy(predictions, batch[2])

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def read_file(path):
    sentences = []
    labels = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip().split()
            sent = line[1:-1]
            lb = line[-1]
            if lb == "#neg":
                labels.append(0)
            else:
                labels.append(1)
            sentences.append(sent)
    return sentences, labels


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, metavar="STR")
    parser.add_argument('--dev', type=str, metavar="STR")
    parser.add_argument('--test', type=str, metavar="STR")
    parser.add_argument('--model', type=str, metavar="STR")
    parser.add_argument('--embedding-dim', type=int, metavar="N")
    parser.add_argument('--hidden-dim', type=int, metavar="N")

    args = parser.parse_args()

    train_sentences, train_labels = read_file(args.train)
    valid_sentences, valid_labels = read_file(args.dev)
    test_sentences, test_labels = read_file(args.test)

    train_data = Data(texts=train_sentences, labels=train_labels)
    train_data.build_vocab()
    train_iterator = train_data.get_batchs(32)

    valid_data = Data(texts=valid_sentences, labels=valid_labels, vocab=train_data.vocab, word2id=train_data.word2id)
    valid_data.nummerize(valid_sentences)
    valid_iterator = valid_data.get_batchs(32)
    test_data = Data(texts=train_sentences, labels=train_labels, vocab=train_data.vocab, word2id=train_data.word2id)
    test_iterator = train_data.get_batchs(32)

    vocab_size = train_data.get_vocab_size()
    embedding_dim = args.embedding_dim
    hidden_dim = args.hidden_dim
    print("Training with " + args.model)
    if args.model == "rnn":
        model = SentimentRNN(vocab_size, embedding_dim, hidden_dim, n_classes=1, bidirectional=False, padding_idx=0)
    elif args.model == "lstm":
        model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, n_classes=1, bidirectional=False, padding_idx=0)
    else:
        model = SentimentGRU(vocab_size, embedding_dim, hidden_dim, n_classes=1, bidirectional=False, padding_idx=0)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    N_EPOCHS = 10

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model_' + args.model + '.pt')

        print(f'Epoch: {epoch + 1:02}:')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f'\tTest. Loss: {test_loss:.3f} |  Test. Acc: {test_acc * 100:.2f}%')
