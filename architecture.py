import os
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import Sequence, Lowercase
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
import tensorflow as tf


class BPETokenizer():
    # hugging face API code
    # is taken and modified from instructor's
    # https://colab.research.google.com/drive/1-TgwCXqYd8ON-58TFzLk413mEqC-7r5F?usp=sharing#scrollTo=0AetkU9nu8OD

    def __init__(self,vocab_size, alphabet):
        self.tokenizer = Tokenizer(BPE(
        ))  # byte pair encoding
        self.tokenizer.normalizer = Sequence([Lowercase()])  # normalization
        self.tokenizer.pre_tokenizer = ByteLevel()  # pre-tokenizer
        self.tokenizer.decoder = ByteLevelDecoder()  # decoder
        self.vocab_size = vocab_size
        self.alphabet = alphabet

    def bpe_train(self, paths):
        trainer = BpeTrainer(vocab_size=self.vocab_size,
                             initial_alphabet=self.alphabet,
                             )
        self.tokenizer.train(paths, trainer)

    def save_tokenizer(self, location, prefix=None):
        if not os.path.exists(location):
            os.makedirs(location)
        self.tokenizer.model.save(location, prefix)

def get_tensor_data(tok, block_size = 5, BUFFER_SIZE = 1000):
    inputs = []
    labels = []

    for i in range(0, len(tok) - block_size + 1, block_size):
        block_to_lag =tok[i:i + block_size]
        inputs.append(block_to_lag[:-1])
        labels.append(block_to_lag[1:])
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    return dataset.shuffle(BUFFER_SIZE).batch(1, drop_remainder=True)
