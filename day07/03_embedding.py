"""
    Word Embedding
    - Convert words to vectors
    - nn.Embedding: lookup table that maps word indices to dense vectors
"""

import torch
import jieba
import torch.nn as nn


def dm01():
    """
    Basic Embedding Example:
    1. Tokenize text with jieba
    2. Create embedding layer
    3. Convert word indices to vectors
    """
    # 1. Input text
    text = '北京冬奥的进度条已经过半,不少外国运动员在完成自己的比赛后踏上归途.'

    # 2. Tokenize with jieba (Chinese word segmentation)
    words = jieba.lcut(text)
    print(f'Tokens: {words}')
    print(f'Token count: {len(words)}')

    # 3. Create embedding layer
    # num_embeddings: vocabulary size (number of unique words)
    # embedding_dim: size of each word vector
    embedding = nn.Embedding(num_embeddings=len(words), embedding_dim=4)
    print(f'Embedding weight shape: {embedding.weight.shape}')

    # 4. Convert each word to its vector
    print('\nWord vectors:')
    for i, word in enumerate(words):
        word_vector = embedding(torch.tensor(i))
        print(f'{word}: {word_vector.detach().numpy()}')


if __name__ == '__main__':
    dm01()