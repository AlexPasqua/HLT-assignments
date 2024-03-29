"""
Derived from code by:
Sahil Chopra <schopra8@stanford.edu>
"""
import time
import os
import logging

from tqdm import tqdm
import numpy as np

from parser_state import ParserState
from corpus import Token
from model import ParserModel

UNK = '<UNK>'                # Unknown token
PAD = '<PAD>'                # Padding token
ROOT = '<ROOT>'              # Root token


class Parser():
    """
    Contains everything needed for transition-based dependency parsing except for the parser model.
    """

    def __init__(self, sentences):
        logging.info('Build dictionary for words.')
        vocab = set([tok.form for sent in sentences for tok in sent])
        self.tok2id = {w: i+3 for i,w in enumerate(vocab)} # leave index for the tokens below
        self.tok2id[ROOT] = self.ROOT = 0 # special token denoting root of parse tree
        self.tok2id[UNK] = self.UNK = 1
        self.tok2id[PAD] = self.PAD = 2
        # mapping from indices to tokens
        self.id2tok = [t for i, t in self.tok2id.items()]

        # build the vocabulary of POS tags:
        logging.info('Build dictionary for POS tags.')
        postags = set([tok.pos for sent in sentences for tok in sent])
        self.pos2id = {w: i+3 for i,w in enumerate(postags)} # leave index for the tokens below
        # Add special tokens for POS to the index
        self.pos2id[ROOT] = self.POS_ROOT = 0
        self.pos2id[UNK] = self.POS_UNK = 1
        self.pos2id[PAD] = self.POS_PAD = 2
        # mapping from indices to POS
        self.id2pos = [t for i, t in self.pos2id.items()]

        # build the vocabulary of deprels:
        deprels = set([tok.deprel for sent in sentences for tok in sent])
        self.dep2id = {w: i+3 for i,w in enumerate(deprels)} # leave index for the tokens below
        # Add special tokens for deprels to the index
        self.dep2id[ROOT] = self.DEPREL_ROOT = 0
        self.dep2id[UNK] = self.DEPREL_UNK = 1
        self.dep2id[PAD] = self.DEPREL_PAD = 2
        # mapping from indices to deprel
        self.id2dep = [t for i, t in self.dep2id.items()]

        # special root token
        self.root_token = Token(0, self.ROOT, self.POS_ROOT, 0, self.DEPREL_ROOT)

    def vectorize(self, sentences):
        """Replace strings with numeric IDs"""
        return [[Token(tok.id,
                       self.tok2id.get(tok.form, self.UNK),
                       self.pos2id.get(tok.pos, self.POS_UNK),
                       tok.head,
                       self.dep2id.get(tok.deprel, self.DEPREL_UNK))
                 for tok in sent]
                for sent in sentences]

    def create_features(self, sentences):
        """
        Build training instances.
        @return: list(features), list(action) for each state while parsing each sentence
        """
        train_x, train_y = [], []
        with tqdm(total=len(sentences)) as prog:
            for sent in sentences:
                # arcs = [(head, dependent, deprel)]
                state = ParserState([self.root_token], sent)
                while state.buffer:
                    gold_t = state.get_oracle()
                    if gold_t is None:
                        break
                    train_x.append(state.extract_features(self))
                    train_y.append(gold_t)
                    state.step(gold_t)  # perform transition
                prog.update(1)

        return train_x, train_y

    def parse(self, sentences, model):
        """
        @param sentences: a list of (list Token).
        @param model: a trained parser model.
        """
        vsentences = self.vectorize(sentences)

        UAS = LAS = all_tokens = 0.0
        for sent in vsentences:
            print('.', end='') # DEBUG
            state = ParserState([self.root_token], sent, []) # FIXME
            while state.buffer:
                feats = state.extract_features(self)
                trans = model.predict([feats])[0].argmax()
                state.step(trans)
            for arc in state.arcs:
                pred_h = arc[0].id
                gold_h = sent[arc[1].id - 1].head
                UAS += pred_h == gold_h
                # pred_l = arc[2]
                # gold_l = sent[arc[1].id - 1].deprel
                # LAS += pred_l == gold_l and pred_h == gold_h
                all_tokens += 1
        UAS /= all_tokens
        LAS /= all_tokens
        return UAS, LAS

