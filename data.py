import os
import torch

import util.datahelper as datahelper
import util.texthelper as texthelper

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word[len(self.idx2word)] = word
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train, self.train_lang = self.tokenize(os.path.join(path, 'train.txt'), True)
        print("train:", len(self.dictionary))
        self.valid, self.valid_lang = self.tokenize(os.path.join(path, 'valid.txt'), False)
        print("valid:", len(self.dictionary))
        self.test, self.test_lang = self.tokenize(os.path.join(path, 'test.txt'), False)
        print("test:", len(self.dictionary))

        # self.train_seq_idx_matrix = self.create_seq_idx_matrix(os.path.join(path, 'train.txt'))
        # self.valid_seq_idx_matrix = self.create_seq_idx_matrix(os.path.join(path, 'valid.txt'))
        # self.test_seq_idx_matrix = self.create_seq_idx_matrix(os.path.join(path, 'test.txt'))

        # self.train_seq_word_matrix = self.create_seq_word_matrix(os.path.join(path, 'train.txt'))
        # self.valid_seq_word_matrix = self.create_seq_word_matrix(os.path.join(path, 'valid.txt'))
        # self.test_seq_word_matrix = self.create_seq_word_matrix(os.path.join(path, 'test.txt'))

        print("dictionary size:", len(self.dictionary))

    def create_seq_idx_matrix(self, path):
        assert os.path.exists(path)

        matrix = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.replace("  ", " ")
                words = line.split() + ['<eos>']
                word_tokens = torch.LongTensor(len(words))
                for i in range(len(words)):
                    word = words[i]

                    if not word in self.dictionary.word2idx:
                        word_id = self.dictionary.word2idx["<oov>"]
                    else:
                        word_id = self.dictionary.word2idx[word]
                        
                    word_tokens[i] = word_id
                matrix.append(word_tokens.unsqueeze_(1))
        return matrix

    def create_seq_word_matrix(self, path):
        assert os.path.exists(path)

        matrix = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.replace("  ", " ")
                words = line.split() + ['<eos>']
                word_tokens = []
                for word in words:
                    word_tokens.append(word)
                matrix.append(word_tokens)
        return matrix

    def tokenize(self, path, save):
        """Tokenizes a text file."""
        assert os.path.exists(path)

        # Add words to the dictionary
        self.dictionary.add_word("<oov>")

        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                line = line.strip()
                line = line.replace("  ", " ")
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    if save:
                        self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            langs = torch.LongTensor(tokens)
            token = 0
            for line in f:
                line = line.strip()
                line = line.replace("  ", " ")
                words = line.split() + ['<eos>']
                for word in words:
                    if not word in self.dictionary.word2idx:
                        ids[token] = self.dictionary.word2idx["<oov>"]
                    else:
                        ids[token] = self.dictionary.word2idx[word]

                    if texthelper.is_contain_chinese_word(word):
                        langs[token] = 1
                    else:
                        langs[token] = 0

                    token += 1

        return ids, langs
