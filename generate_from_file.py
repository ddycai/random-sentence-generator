"""
Generate a random sentence given an input file.
"""

import nltk.data
from nltk import word_tokenize
import argparse
from sentence_generator import Generator

parser = argparse.ArgumentParser()
parser.add_argument('filename', help="Filename to use as a corpus.")
parser.add_argument('-c', '--chain_length', help="Number of words to look back \
        as context when generating new words. Default is 2.", type=int, default=2)
args = parser.parse_args()

with open(args.filename, 'r') as f:
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sents = sent_detector.tokenize(f.read().strip())
    sent_tokens = [word_tokenize(sent.replace('\n', ' ').lower()) for sent in sents]
    generator = Generator(sent_tokens, args.chain_length)
    print(generator.generate())
