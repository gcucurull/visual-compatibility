import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import os
import warnings


class GraphAttention(layers.Layer):
    def __init__(
        self,
        units,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):

        self.kernel = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel",
        )
        self.kernel_attention = self.add_weight(
            shape=(self.units * 2, 1),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_attention",
        )
        self.built = True

    def call(self, inputs):
        node_states, edges = inputs

        # Linearly transform node states
        node_states_transformed = tf.matmul(node_states, self.kernel)

        # (1) Compute pair-wise attention scores
        node_states_expanded = tf.gather(node_states_transformed, edges)
        node_states_expanded = tf.reshape(
            node_states_expanded, (tf.shape(edges)[0], -1)
        )
        attention_scores = tf.nn.leaky_relu(
            tf.matmul(node_states_expanded, self.kernel_attention)
        )
        attention_scores = tf.squeeze(attention_scores, -1)

        # (2) Normalize attention scores
        attention_scores = tf.math.exp(tf.clip_by_value(attention_scores, -2, 2))
        attention_scores_sum = tf.math.unsorted_segment_sum(
            data=attention_scores,
            segment_ids=edges[:, 0],
            num_segments=tf.reduce_max(edges[:, 0]) + 1,
        )
        attention_scores_sum = tf.repeat(
            attention_scores_sum, tf.math.bincount(tf.cast(edges[:, 0], "int32"))
        )
        attention_scores_norm = attention_scores / attention_scores_sum

        # (3) Gather node states of neighbors, apply attention scores and aggregate
        node_states_neighbors = tf.gather(node_states_transformed, edges[:, 1])
        out = tf.math.unsorted_segment_sum(
            data=node_states_neighbors * attention_scores_norm[:, tf.newaxis],
            segment_ids=edges[:, 0],
            num_segments=tf.shape(node_states)[0],
        )
        return out


class MultiHeadGraphAttention(layers.Layer):
    def __init__(self, units, num_heads=2, merge_type="concat", **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.merge_type = merge_type
        self.attention_layers = [GraphAttention(units) for _ in range(num_heads)]

    def call(self, inputs):
        atom_features, pair_indices = inputs

        # Obtain outputs from each attention head
        outputs = [
            attention_layer([atom_features, pair_indices])
            for attention_layer in self.attention_layers
        ]
        # Concatenate or average the node states from each head
        if self.merge_type == "concat":
            outputs = tf.concat(outputs, axis=-1)
        else:
            outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)
        # Activate and return node states
        return tf.nn.relu(outputs)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = tf.keras.layers.Dense(1)
        self.dropout = 1
    
    def call(self, inputs, r_indices, c_indices):

        node_inputs = tf.nn.dropout(inputs, 1 - self.dropout)

        # r corresponds to the selected rows, and c to the selected columns
        row_inputs = tf.gather(node_inputs, r_indices)
        col_inputs = tf.gather(node_inputs, c_indices)

        diff = tf.abs(row_inputs - col_inputs)

        outputs = self.params(diff)

        return outputs


class GAEAttn(tf.keras.Model):
    
    def __init__(self, output_shapes = [350, 350, 350] ,**kwargs):
        super().__init__(**kwargs)
        self.output_shapes = output_shapes
        self.attn = []
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.accuracy_func = tf.keras.metrics.binary_accuracy
        for m in self.output_shapes:
            self.attn.append(MultiHeadGraphAttention(m))
        
        self.decoder = Decoder()

    def call(self, inputs, r_indices, c_indices):
        # return self.model(inputs)
        O , A  = inputs
        for layer in self.attn:
             O = layer([O , A])
        return self.decoder(O, r_indices, c_indices)
    
    # @tf.function
    def train_step(self, labels, inputs, train_r_indices, train_c_indices):
        with tf.GradientTape() as tape:
            outputs = self(inputs, train_r_indices, train_c_indices)
            loss = self.loss_func(labels, outputs)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        acc = self.accuracy_func(labels, outputs)
        return loss , tf.reduce_mean(acc , axis = 0)