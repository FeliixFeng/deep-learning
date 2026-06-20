"""
    RNN Lyrics Generation
    - Train on Jay Chou lyrics
    - Generate new lyrics using RNN
"""

import torch
import jieba
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time


def build_vocab():
    """
    Build vocabulary from lyrics file:
    1. Read all lines
    2. Tokenize with jieba
    3. Build word-to-index mapping
    4. Convert all words to indices
    """
    unique_words, all_words = [], []

    # 1. Read lyrics and tokenize
    for line in open('./data/jaychou_lyrics.txt', 'r', encoding='utf8').readlines():
        words = jieba.lcut(line)
        all_words.append(words)
        # Collect unique words
        for word in words:
            if word not in unique_words:
                unique_words.append(word)

    # 2. Build vocabulary
    word_count = len(unique_words)
    word_to_index = {word: i for i, word in enumerate(unique_words)}

    # 3. Convert all words to indices
    corpus_idx = []
    for words in all_words:
        tmp = []
        for word in words:
            tmp.append(word_to_index[word])
        tmp.append(word_to_index[' '])  # Add space as separator
        corpus_idx.extend(tmp)

    return unique_words, word_to_index, word_count, corpus_idx




if __name__ == '__main__':
    # 1. Build vocabulary
    unique_words, word_to_index, word_count, corpus_idx = build_vocab()

    # 2. Print results
    print(f'Vocabulary size: {word_count}')
    print(f'First 10 words: {unique_words[:10]}')
    print(f'Corpus length: {len(corpus_idx)} indices')
