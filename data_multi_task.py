import os
import torch

import util.datahelper as datahelper
import util.texthelper as texthelper

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.pos2idx = {}
        self.idx2pos = {}

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word[len(self.idx2word)] = word
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def add_pos(self, pos):
        if pos not in self.pos2idx:
            self.idx2pos[len(self.idx2pos)] = pos
            self.pos2idx[pos] = len(self.idx2pos) - 1
        return self.pos2idx[pos]

    def __len__(self):
        return len(self.idx2word)

    def get_postag_len(self):
        return len(self.idx2pos)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'), True)
        print("train:", len(self.dictionary))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'), False)
        print("valid:", len(self.dictionary))
        self.test = self.tokenize(os.path.join(path, 'test.txt'), False)
        print("test:", len(self.dictionary))

        self.train_lang = self.tokenize_lang(os.path.join(path, 'train.txt'))
        self.valid_lang = self.tokenize_lang(os.path.join(path, 'valid.txt'))
        self.test_lang = self.tokenize_lang(os.path.join(path, 'test.txt'))

        self.train_postag = self.tokenize_pos(os.path.join(path, 'train_pos_tag.txt'), True)
        self.valid_postag = self.tokenize_pos(os.path.join(path, 'valid_pos_tag.txt'), True)
        self.test_postag = self.tokenize_pos(os.path.join(path, 'test_pos_tag.txt'), True)

        self.train_seq_idx_matrix = self.create_seq_idx_matrix(os.path.join(path, 'train.txt'))
        self.valid_seq_idx_matrix = self.create_seq_idx_matrix(os.path.join(path, 'valid.txt'))
        self.test_seq_idx_matrix = self.create_seq_idx_matrix(os.path.join(path, 'test.txt'))

        self.train_seq_word_matrix = self.create_seq_word_matrix(os.path.join(path, 'train.txt'))
        self.valid_seq_word_matrix = self.create_seq_word_matrix(os.path.join(path, 'valid.txt'))
        self.test_seq_word_matrix = self.create_seq_word_matrix(os.path.join(path, 'test.txt'))

        print("dictionary size:", len(self.dictionary))
        self.en_mask = []
        self.zh_mask = []
        self.other_mask = []

        self.create_language_mask()
        
        self.zh_mask = torch.LongTensor(self.zh_mask)
        self.en_mask = torch.LongTensor(self.en_mask)
        self.other_mask = torch.LongTensor(self.other_mask)

    def create_language_mask(self):
        for i in range(len(self.dictionary.idx2word)):
            word = self.dictionary.idx2word[i]
            if texthelper.is_contain_chinese_word(word):
                self.zh_mask.append(1)
                self.en_mask.append(0)
                self.other_mask.append(0)
            elif word == "<eos>":
                self.zh_mask.append(0)
                self.en_mask.append(0)
                self.other_mask.append(1)
            else:
                self.zh_mask.append(0)
                self.en_mask.append(1)
                self.other_mask.append(0)

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

    def tokenize_pos(self, path, save):
        """Tokenizes a text file."""
        assert os.path.exists(path)

        # Add words to the dictionary
        self.dictionary.add_pos("<oop>")

        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                line = line.strip()
                line = line.replace("  ", " ")
                all_pos = line.split() + ['<oop>']
                tokens += len(all_pos)
                for pos in all_pos:
                    if save:
                        self.dictionary.add_pos(pos)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            langs = torch.LongTensor(tokens)
            token = 0
            for line in f:
                line = line.strip()
                line = line.replace("  ", " ")
                all_pos = line.split() + ['<oop>']
                for pos in all_pos:
                    if not pos in self.dictionary.pos2idx:
                        ids[token] = self.dictionary.pos2idx["<oop>"]
                    else:
                        ids[token] = self.dictionary.pos2idx[pos]

                    token += 1

        return ids

    def tokenize_lang(self, path):
        # 0 for ZH, 1 for EN, 2 for EOS
        assert os.path.exists(path)

        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                line = line.strip()
                line = line.replace("  ", " ")
                words = line.split() + ['<eos>']
                tokens += len(words)

        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            langs = torch.LongTensor(tokens)
            token = 0
            for line in f:
                line = line.strip()
                line = line.replace("  ", " ")

                strs = line.split()
                for i in range(len(strs)):
                    word = strs[i]
                    if texthelper.is_contain_chinese_word(word):
                        ids[token] = 0
                    else:
                        ids[token] = 1
                    token += 1

                # add EOS (2)
                ids[token] = 2
                token += 1

        return ids

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

                    token += 1

        return ids
