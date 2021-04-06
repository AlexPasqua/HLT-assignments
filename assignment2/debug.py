from parser_state import ParserState
from corpus import Token

ROOT = Token(0, 'ROOT')

def test_parse():
    """Simple tests for the PartialParse.parse function.
    Warning: these are not exhaustive.
    """
    sentence = [Token(i+1, f) for i,f in enumerate(["parse", "this", "sentence"])]
    state = ParserState(stack=[ROOT], buffer=sentence)
    dependencies = state.parse(["S", "S", "S", "LA", "RA", "RA"])
    dependencies = [(a[0].form, a[1].form) for a in sorted(dependencies)]
    expected = [('ROOT', 'parse'), ('parse', 'sentence'), ('sentence', 'this')]
    assert dependencies == expected, \
        f"parse test resulted in dependencies {dependencies}, expected {expected}"
    assert [t.form for t in sentence] == ["parse", "this", "sentence"], \
        f"parse test failed: the input sentence should not be modified"
    print(f"{transition} parse test passed!")

test_parse()

from corpus import read_conll

train_file = 'data/train.gold.conll'
dev_file = 'data/dev.gold.conll'
test_file = 'data/test.gold.conll'

max_sent = 500

train_sents = read_conll(train_file, max_sent=max_sent)
dev_sents = read_conll(dev_file, max_sent=max_sent//2)
test_sents = read_conll(test_file, max_sent=max_sent//2)

from parser import Parser

parser = Parser(train_sents)

train_vectors = parser.vectorize(train_sents)
dev_vectors = parser.vectorize(dev_sents)

train_x, train_y = parser.create_features(train_vectors)
dev_x, dev_y = parser.create_features(dev_vectors)

from tensorflow.data import Dataset

ds_train = Dataset.from_tensor_slices((train_x, train_y)).shuffle(1000).batch(32)
ds_dev = Dataset.from_tensor_slices((dev_x, dev_y)).shuffle(1000).batch(32)

# load GloVE
from glove import Glove

glove_path = '../data/glove.6B/glove.6B.50d.txt'

glove = Glove(glove_path)

# Prepate the embeddings
num_tokens = len(parser.tok2id)
embedding_dim = len(next(iter(glove.wv.values())))

# Create the model
import numpy as np

# Fill the matrix with Glove embeddings
embedding_matrix = np.random.uniform(-1, 1, (num_tokens, embedding_dim))
for word, i in parser.tok2id.items():
    embedding_vector = glove.wv.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be random.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector

from model import ParserModel

n_features = len(train_x[0][0])
n_pos = len(parser.pos2id)
n_tags = len(parser.dep2id)
tag_size = 20 # size of embeddings for POS and DEPRELs 
n_actions = 2 * n_tags + 1 # L- + R- + S
hidden_size = 200

model = ParserModel(embeddings=embedding_matrix, n_features=n_features,
                    n_pos=n_pos, n_tags=n_tags, tag_size=tag_size,
                    n_actions=n_actions, hidden_size=hidden_size)

# Compile the model
from tensorflow import keras

model.compile(
    # Optimizer
    optimizer=keras.optimizers.Adam(),
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(name='train_loss'),
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy(name='train acc')],
)

# Train the model

EPOCHS = 3
history = model.fit(ds_train, epochs=EPOCHS,
                    validation_data=ds_dev)

# Parse test

UAS, LAS = parser.parse(test_sents, model)
print('UAS', UAS, 'LAS', LAS)
