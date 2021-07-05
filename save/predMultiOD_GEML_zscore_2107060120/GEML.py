from keras.models import Model
from keras.layers import Input, Dense, Reshape, Lambda, add, LSTM, Concatenate
from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
import keras.backend as K
import tensorflow as tf
from Param import *
from Param_GEML import *

BATCHSIZE = 1


class GraphConvolution(Layer):
    def __init__(self, units, support=1,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',  # Gaussian distribution
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True

        self.support = support
        assert support >= 1

    def build(self, input_shapes):
        features_shape = input_shapes[0]
        # assert len(features_shape) == 2
        input_dim = features_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        super(GraphConvolution, self).build(input_shapes)

    def call(self, inputs, mask=None):
        features = inputs[0]
        links = inputs[1]

        result = K.batch_dot(links, features, axes=1)
        output = K.dot(result, self.kernel)
        # output = result

        if self.bias:
            output += self.bias

        return self.activation(output)

    def compute_output_shape(self, input_shapes):
        input_shape = input_shapes[0]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)


def sequence_GCN(input_seq, adj_seq, unit, act='relu', **kwargs):
    GCN = GraphConvolution(unit, activation=act, **kwargs)
    embed = []
    for n in range(input_seq.shape[1]):
        frame = Lambda(lambda x: x[:, n, :, :])(input_seq)
        adj = Lambda(lambda x: x[:, n, :, :])(adj_seq)
        embed.append(GCN([frame, adj]))
    output = Lambda(lambda x: tf.stack(x, axis=1))(embed)
    return output


class TransitionLayer(Layer):
    def __init__(self, kernel_initializer='glorot_uniform',  # Gaussian distribution
                 kernel_regularizer=None,
                 kernel_constraint=None, **kwargs):
        super(TransitionLayer, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

    def build(self, input_shape):
        sample = input_shape[1]

        self.W = self.add_weight(shape=(sample, sample),
                                 initializer=self.kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.built = True

    def call(self, input, mask=None):
        H_i = tf.unstack(input)
        H_j = tf.unstack(tf.transpose(input, (0, 2, 1)))
        transition = []
        for hi, hj in zip(H_i, H_j):
            transition.append(K.dot(hi, hj))
        transition = tf.stack(transition, axis=0)

        output = K.dot(transition, self.W)
        return output

    def compute_output_shape(self, input_shapes):
        sample = input_shapes[1]
        output_shape = list(input_shapes)
        output_shape[-1] = sample
        return tuple(output_shape)


def getModel():
    X_fea = Input(batch_shape=(BATCHSIZE, TIMESTEP_IN, HEIGHT, WIDTH))
    X_adj = Input(batch_shape=(BATCHSIZE, TIMESTEP_IN, HEIGHT, WIDTH))
    X_seman = Input(batch_shape=(BATCHSIZE, TIMESTEP_IN, HEIGHT, WIDTH))
    X_temp = Input(batch_shape=(BATCHSIZE, day_fea))

    x1_nebh = sequence_GCN(X_fea, X_adj, 128)
    x2_nebh = sequence_GCN(x1_nebh, X_adj, 128)

    x1_seman = sequence_GCN(X_fea, X_seman, 128)
    x2_seman = sequence_GCN(x1_seman, X_seman, 128)

    dens1 = Dense(units=10, activation='relu')(X_temp)
    dens2 = Dense(units=TIMESTEP_IN * HEIGHT * 128, activation='relu')(dens1)
    hmeta = Reshape((TIMESTEP_IN, HEIGHT, 128))(dens2)

    # embed_fea = add([x2_nebh, x2_seman, hmeta])
    embed_fea = Concatenate()([x2_nebh, x2_seman, hmeta])
    embed_fea = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)))(embed_fea)
    # lstm_fea = TimeDistributed(LSTM(2 * ENCODER_DIM, return_sequences=False))(embed_fea)
    embed_fea = Lambda(lambda x: K.reshape(x, (BATCHSIZE * HEIGHT, TIMESTEP_IN, 3 * 128)))(embed_fea)

    # lstm_fea = LSTM(128, return_sequences=False)(embed_fea)
    # hidden = Lambda(lambda x: K.reshape(x, (BATCHSIZE, HEIGHT, 128)))(lstm_fea)
    # out = TransitionLayer(name='trans')(hidden)
    # out = Lambda(lambda x: tf.expand_dims(x, axis=1))(out)

    lstm_fea = LSTM(WIDTH, return_sequences=False, activation='linear')(embed_fea)
    out = Lambda(lambda x: K.reshape(x, (BATCHSIZE, 1, HEIGHT, WIDTH)))(lstm_fea)

    model = Model(inputs=[X_fea, X_adj, X_seman, X_temp], outputs=out)
    return model
