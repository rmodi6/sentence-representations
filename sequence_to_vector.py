# std lib imports
from typing import Dict

# external libs
import tensorflow as tf
from tensorflow.keras import layers, models


class SequenceToVector(models.Model):
    """
    It is an abstract class defining SequenceToVector enocoder
    abstraction. To build you own SequenceToVector encoder, subclass
    this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    """
    def __init__(self,
                 input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``tf.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``tf.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : tf.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : tf.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """
        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError


class DanSequenceToVector(SequenceToVector):
    """
    It is a class defining Deep Averaging Network based Sequence to Vector
    encoder. You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this DAN encoder.
    dropout : `float`
        Token dropout probability as described in the paper.
    """
    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2):
        super(DanSequenceToVector, self).__init__(input_dim)
        # TODO(students): start

        self.num_layers = num_layers
        self.dropout_prob = dropout
        self.dan_layers = [layers.Dense(self._input_dim, activation='relu') for i in range(self.num_layers)]

        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start

        batch_size, max_tokens_num, embed_size = vector_sequence.shape

        sequence_mask = tf.reshape(sequence_mask, (batch_size, max_tokens_num, 1))
        num_words = tf.math.reduce_sum(sequence_mask, axis=1)
        vector_sequence = sequence_mask * vector_sequence
        if training:
            dropout_mask = tf.random.uniform(shape=(batch_size, max_tokens_num, 1)) >= self.dropout_prob
            dropout_mask = tf.cast(dropout_mask, 'float32')
            num_words = tf.math.reduce_sum(dropout_mask * sequence_mask, axis=1)
            vector_sequence = dropout_mask * vector_sequence

        combined_vector = tf.math.reduce_sum(vector_sequence, axis=1) / num_words

        layer_representations = []
        for layer in self.dan_layers:
            combined_vector = layer(combined_vector)
            layer_representations.append(combined_vector)

        layer_representations = tf.stack(layer_representations, axis=1)

        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}


class GruSequenceToVector(SequenceToVector):
    """
    It is a class defining GRU based Sequence To Vector encoder.
    You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """
    def __init__(self, input_dim: int, num_layers: int):
        super(GruSequenceToVector, self).__init__(input_dim)
        # TODO(students): start

        self.num_layers = num_layers
        self.gru_layers = [layers.GRU(self._input_dim, return_sequences=True, return_state=True) for i in range(self.num_layers)]

        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start

        batch_size, max_tokens_num, embed_size = vector_sequence.shape

        sequence_mask = tf.reshape(sequence_mask, (batch_size, max_tokens_num, 1))
        layer_representations = []
        for layer in self.gru_layers:
            vector_sequence, combined_vector = layer(vector_sequence, mask=sequence_mask)
            layer_representations.append(combined_vector)

        combined_vector = layer_representations[-1]
        layer_representations = tf.stack(layer_representations, axis=1)

        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}
