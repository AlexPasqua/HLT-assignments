import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class FeaturesEmbedding(layers.Layer):
    """
    Custom layer to create embeddings for the input data.
    
    @param embeddings (ndarray): word embeddings for input tokens.
    @param pos_size (int): number of POS tags.
    @param pos_dim (int): dimension of POS embeddings.
    @param deprel_size (int): number of deprels.
    @param deprel_dim (int): dimension of deprel embeddings.
    """

    ###
    ### Please see the following documentation for support:
    ###     Custom layer: https://www.tensorflow.org/guide/keras/custom_layers_and_models
    ###     Embeddings: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
    ###     Concatenate: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Concatenate
    ###

    def __init__(self, embeddings, pos_size, pos_dim, deprel_size, deprel_dim):
        super().__init__()
        # initialize with pretrained weights
        self.token_emb = layers.Embedding(embeddings.shape[0],
                                          embeddings.shape[1],
                                          weights=[embeddings])
        # initialize with random weights
        self.pos_emb = layers.Embedding(input_dim=pos_size, output_dim=pos_dim)
        # initialize with random weights
        self.dep_emb = layers.Embedding(input_dim=deprel_size, output_dim=deprel_dim)
        self.concat = layers.Concatenate()
        self.flatten = layers.Flatten()

    def call(self, x):
        w = self.token_emb(x[:,0])
        p = self.pos_emb(x[:,1])
        d = self.dep_emb(x[:,2])
        o = self.concat([w, p, d])
        # return single vector of all features
        return self.flatten(o)


class ParserModel(keras.Model):
    """
    Feedforward neural network with an embedding layer and two hidden layers.
    Input conists of a list of triples (list of 18 form ids, list of 18 pos, list of 12 deprel ids)
    The ParserModel will predict which transition should be applied to a
    given parser state.

    Keras Notes:
        - Note that "ParserModel" is a subclass of the "keras.Model" class.
        - The "__init__" method is where you define all the layers and parameters
            (embedding layers, linear layers, dropout layers, etc.).
        - "__init__" gets automatically called when you create a new instance of your class, e.g.
            when you write "m = ParserModel()".
        - Other methods of ParserModel can access variables that have "self." prefix. Thus,
            you should add the "self." prefix layers, values, etc. that you want to utilize
            in other ParserModel methods.
        - For further documentation on "keras.Model" please see https://www.tensorflow.org/api_docs/python/tf/keras/Model.
    """
    def __init__(self, embeddings, n_features=18,
                 n_pos=20, n_tags=30, tag_size=20, n_actions=3,
                 hidden_size=200, dropout_prob=0.5):
        """ Initialize the parser model.

        @param embeddings (ndarray): word embeddings (num_words, embedding_size)
        @param n_features (int): number of input features.
        @param n_pos (int): number of POS tags.
        @param n_tags (int): number of DEPREL tags.
        @param tag_size (int): size of embeddings for POS and DEPREL tags.
        @param n_actions (int): number of possible parser actions.
        @param hidden_size (int): number of hidden units.
        @param dropout_prob (float): dropout probability.
        """
        super().__init__()
        self.n_features = n_features
        self.n_actions = n_actions
        self.hidden_size = hidden_size

        ### YOUR CODE HERE (~4-5 Lines)
        ### TODO:
        ###     1) Use the `FeaturesEmbedding` as first layer.
        ###     2) Construct `self.dropout` layer.
        ###     3) Declare `self.hidden_to_logits_weight` and `self.hidden_to_logits_bias`
        ###        Initialize weight with the `nn.init.xavier_uniform_` function and bias
        ###        with default parameters.
        ###
        ### Please see the following docs for support:
        ###     Model: https://www.tensorflow.org/guide/keras/custom_layers_and_models
        ###     Dropout: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout
        ###
        self.features_embedding = FeaturesEmbedding(embeddings=embeddings, pos_size=n_pos, pos_dim=tag_size,
                                                    deprel_size=n_tags, deprel_dim=tag_size)
        self.dense = layers.Dense(units=self.hidden_size, activation="relu")
        self.dropout = layers.Dropout(dropout_prob)
        self.softmax = layers.Softmax()
        initializer = keras.initializers.GlorotUniform()
        self.hidden_to_logits_weight = initializer(shape=(hidden_size, n_actions))
        self.hidden_to_logits_bias = initializer(shape=(1, n_actions))
        ### END YOUR CODE


    def call(self, inputs, training=False):
        """
        Run the model forward.

        Keras Notes:
            - Every `Model` object (Keras model) has a `call` function.
            - When you apply your `Model` to an input tensor `w` this function is applied to the tensor.
                For example, if you created an instance of your ParserModel and applied it to some `w` as follows,
                the `call` function would called on `w` and the result would be stored in the `output` variable:
                    model = ParserModel()
                    output = model(w) # this calls the `call` method
            - For more details see: https://www.tensorflow.org/guide/keras/functional

        @param w (Tensor): input tensor of tokens (batch_size, n_features)

        @return logits (Tensor): tensor of predictions (output after applying the layers of the network)
                                 of size (batch_size, n_actions)
        """
        ### YOUR CODE HERE (~3-5 lines)
        ### TODO:
        ###     Complete the forward computation as described in the `Sketch` notebook.
        ###   . In addition, include a dropout layer
        ###     as declared in `__init__` after ReLU function.
        ###
        ### Please see the following docs for support:
        ###     Matrix product: https://www.tensorflow.org/api_docs/python/tf/linalg/matmul
        ###     ReLU: https://www.tensorflow.org/api_docs/python/tf/keras/activations/relu

        x = self.features_embedding(inputs)
        h = self.dense(x)
        h = self.dropout(h)
        l = tf.linalg.matmul(h, self.hidden_to_logits_weight) # + self.hidden_to_logits_bias
        y_hat = self.softmax(l)
        return y_hat

        ### END YOUR CODE

