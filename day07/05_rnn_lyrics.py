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

class LyricsDataset(torch.utils.data.Dataset):
    def __init__(self, corpus_idx, num_chars):
        self.corpus_idx = corpus_idx
        self.num_chars = num_chars
        self.word_count = len(self.corpus_idx)
        self.number = self.word_count // self.num_chars

    def __len__(self):
        return self.number

    def __getitem__(self, idx):
        start = min(max(idx, 0), self.word_count - self.num_chars - 1)
        end = start + self.num_chars
        x = self.corpus_idx[start: end]
        y = self.corpus_idx[start+1:end+1]
        return torch.tensor(x), torch.tensor(y)


class TextGenerator(nn.Module):
    def __init__(self, unique_word_count):
        super().__init__()
        self.ebd = nn.Embedding(unique_word_count, 128)
        self.rnn = nn.RNN(128, 256, 1)
        self.out = nn.Linear(256, unique_word_count)

    def forward(self, inputs, hidden):
        embd = self.ebd(inputs)
        output, hidden = self.rnn(embd.transpose(0, 1), hidden)
        output = self.out(output.reshape(-1, output.shape[-1]))
        return output, hidden

    def init_hidden(self, bs):
        return torch.zeros(1, bs, 256)

def train():
    unique_words, word_to_index, unique_word_count, corpus_idx = build_vocab()
    lyrics = LyricsDataset(corpus_idx, 32)
    model = TextGenerator(unique_word_count)
    lyrics_dataloader = DataLoader(lyrics, batch_size=5, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 50
    for epoch in range(epochs):
        start, iter_num, total_loss = time.time(), 0, 0.0
        for x, y in lyrics_dataloader:
            hidden = model.init_hidden(5)
            output, hidden = model(x, hidden)
            y = torch.transpose(y, 0, 1).reshape(shape=(-1,))
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            iter_num += 1

        print(f'Epoch {epoch + 1}, loss: {total_loss / iter_num:.4f}, time: {time.time() - start:.2f}s')

    torch.save(model.state_dict(), './model/lyrics.pth')


def evaluate(start_word, sentence_lenght):
    unique_words, word_to_index, unique_word_count, corpus_idx = build_vocab()
    model = TextGenerator(unique_word_count)
    model.load_state_dict(torch.load('./model/lyrics.pth'))
    hidden = model.init_hidden(1)
    word_index = word_to_index[start_word]
    generate_sentence = [word_index]

    for i in range(sentence_lenght):
        output,hidden = model(torch.tensor([[word_index]]), hidden)
        word_index = torch.argmax(output)
        generate_sentence.append(word_index)

    for idx in generate_sentence:
        print(unique_words[idx], end='')


if __name__ == '__main__':
    # 1. Build vocabulary
    unique_words, word_to_index, word_count, corpus_idx = build_vocab()


    # # dataset = LyricsDataset(corpus_idx, 5)
    # model = TextGenerator(word_count)
    #
    # for name, parameter in model.named_parameters():
    #     print(f'name: {name}, parameter: {parameter.shape}')
    train()
    evaluate("分手", 50)