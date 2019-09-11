"""

RNNCell where states are updated based on the model used in FORCE training method.

    Every RNNCell must have state_size and output_size as property and implement __call__() method to be compatible with
    dynamic_rnn method
    input size is [batch_size, time_step, input_dim]
    state size is [batch_size, cell.state_size]
    output size is [batch_size, time_step, cell.output_dim]
    
author: Elham
    
    *******************************  UPDATED on July 30, 2019    ***********************************  
 """

import tensorflow as tf


class forceRNNCell(tf.contrib.rnn.RNNCell):

    def __init__(self, output_dim, neuron_size, fb_weights, net_weights, input_weights,
                 initial_weight, g, dt, tau):
        self.output_dim = output_dim
        self.neuron_size = neuron_size
        self.fb_weights = fb_weights
        self.net_weights = net_weights
        self.input_weights = input_weights
        self.initial_weight = initial_weight
        self.g = g
        self.dt = dt
        self.tau = tau

        self.w = tf.get_variable('wo', [self.neuron_size, self.output_dim], tf.float32,
                                 initializer=self.initial_weight)

    @property
    def output_size(self):
        return self.neuron_size

    @property
    def state_size(self):
        return self.neuron_size

    def __call__(self, inputs, state):

        r = tf.tanh(state)
        d_state = -state + self.g * tf.matmul(r, self.net_weights) + tf.matmul(tf.matmul(r, self.w), self.fb_weights)  \
                     + tf.matmul(inputs, self.input_weights)

        next_state = state + (d_state * self.dt)/self.tau
        

        return next_state, next_state



