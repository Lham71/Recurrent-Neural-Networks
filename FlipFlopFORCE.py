""" 
FlipFlop class

 **********************************************************
                       MAIN CODE
 **********************************************************

Use FORCE training and custom Continuous RNNCell to learn an n-bit flipflop
July 2019

author: Elham

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from scipy import sparse
import numpy.random as npr
from ContinousRNNCell import forceRNNCell
from matplotlib import pyplot as plt
from drawnow import *
import random

#plt.style.use('ggplot')



class ForceFlipFlop:

    def __init__(self, **hps):

        # RNN model parameters
        self.n_input = hps['n_input']
        self.n_output = hps['n_output']
        self.tau = hps['tau']
        self.N = hps['N_G']
        self.pr = hps['p_G']
        self.g = hps['g_G']
        self.dt = hps['dt']

        # learning params
        self.t_step = hps['time_step']
        self.alpha_w = hps['alpha_w']
        self.T = hps['total_T']
        self.n_batch = 1
        self.t = np.arange(0, self.T, self.dt)   # time intervals
        self.t_train = np.reshape([np.split(self.t, 3)[0], np.split(self.t, 3)[1]], (1, -1))[0]
        self.t_test = np.split(self.t, 3)[2]


        if len(self.t) % self.t_step != 0:
            raise ValueError(' T * dt (trial length * time interval) is not divisible by time_step')

        # other
        self.traj_length = hps['traj_length']
        self.p_flop = hps['p_flop']
        self.pulse_dur = hps['pulse_duration']
        self.n_bits = hps['n_bits']
        self.n_plot = hps['n_plot']
        self.seed = hps['seed']
        self.rng = npr.RandomState(self.seed)

        self.neuron_num = hps['nueron_plot']
        self.color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                      for i in range(self.neuron_num)]
        self.neuron_list = self.rng.randint(self.N, size=self.neuron_num)

        self.initialize_model()
        self.set_model()


        self.split_to_mini_batch()


    def initialize_model(self):
        """ function to initialize Pw (trainable_var) [N, N], initial values of readouts [n_output, N]
        initial_state_values [1, N] (for dynamic rnn), J [N, N], wi [n_input*N] and wf [n_output, N]
        and Session """

        self.Pw = tf.get_variable('cov_inv', [self.N, self.N], tf.float32,
                                  initializer=tf.constant_initializer(np.eye(self.N)/ self.alpha_w))
        #self.initial_w = tf.constant_initializer((2 * self.rng.rand(self.n_output, self.N) -1)/10)   # tf op
        self.initial_w = tf.constant_initializer((np.zeros((self.n_output, self.N))))  # tf op
        std = 1/(np.sqrt(self.pr * self.N))
        J_G = std * sparse.random(self.N, self.N, density=self.pr, random_state=self.seed, data_rvs=self.rng.randn).toarray()
        self.J = tf.constant(J_G/1., dtype=tf.float32)
        self.init_state_val = 0.05  * self.rng.randn(1,self.N)
        #self.init_state_val = (2 * self.rng.rand(1, self.N) - 1)/10.
        self.wi = tf.constant((2 * self.rng.rand(self.n_input, self.N) -1.)/50. , dtype=tf.float32)
        #self.wi = tf.constant(np.ones([self.n_input, self.N]), dtype=tf.float32)/10
        self.wf = tf.constant((2 * self.rng.rand(self.n_output, self.N) -1)/1. , dtype=tf.float32)
        self.sess = tf.Session()


    def set_model(self):
        # data handling
        self.inputs_bxtxd = tf.placeholder(dtype=tf.float32, shape=[1, self.t_step, self.n_input])
        self.outputs_bxtxd = tf.placeholder(dtype=tf.float32, shape=[1, self.t_step, self.n_output])
        self.init_state_1xN = tf.placeholder(dtype=tf.float32, shape=[1, self.N])

        # create RNN
        self.rnn_cell = forceRNNCell(output_dim=self.n_output, neuron_size=self.N, fb_weights=self.wf,
                                 net_weights=self.J, input_weights=self.wi, initial_weight=self.initial_w,
                                 g=self.g, dt=self.dt, tau=self.tau)

        self.w = self.rnn_cell.w
        self.hidden_states, self.final_state = tf.nn.dynamic_rnn(self.rnn_cell, self.inputs_bxtxd, initial_state=self.init_state_1xN)
        self.z_hat = tf.tensordot(tf.tanh(self.hidden_states), self.w, axes=1)


        # learning model
        self.error = self.z_hat - self.outputs_bxtxd
        self.Pw, self.w, self.update_Pw, self.update_w, self.denum = self.update_rule(rates=tf.tanh(self.final_state),
                                                                error=self.error[:,-1,:], Pw=self.Pw, w=self.w)



    def update_rule(self, rates, error, Pw, w):

        """ r is a [N, 1] vector """
        r = tf.transpose(rates)
        rT = rates

        #   Update covariance inv matrix
        Pr = tf.matmul(Pw, r)
        num_pw = tf.matmul(Pr, tf.matmul(rT, Pw))
        denum_pw = 1 + tf.matmul(rT, Pr)
        Delta_Pw = num_pw / denum_pw
        update_Pw = tf.assign_sub(Pw, Delta_Pw)

        #   Update read-out weights
        #Delta_w = tf.matmul(Pr, error)
        Delta_w = tf.multiply(tf.matmul(Pr, error), 1/denum_pw)
        update_w = tf.assign_sub(w, Delta_w)

        return Pw, w, update_Pw, update_w, denum_pw



    def generate_ff_trials(self):

        # randomly generate unsigned input pulses
        unsigned_inputs = self.rng.binomial(1, self.p_flop, [self.n_batch, len(self.t), self.n_bits ])

        unsigned_inputs[:, 0, :] = 1

        #generate random signs
        random_signs = 2 * self.rng.binomial(1, 0.5, [self.n_batch, len(self.t), self.n_bits ]) - 1

        ff_inputs = np.multiply(unsigned_inputs, random_signs)

        ff_outputs = np.zeros([self.n_batch, len(self.t), self.n_bits ])

        for trial_indx in range(self.n_batch):
            for bit_indx in range(self.n_bits):
                input_ = np.squeeze(ff_inputs[trial_indx, :, bit_indx])
                t_flop = np.where(input_ != 0)
                for flop_indx in range(np.size(t_flop)):
                    t_flop_i = t_flop[0][flop_indx]

                    ff_outputs[trial_indx, t_flop_i:, bit_indx] = ff_inputs[trial_indx, t_flop_i, bit_indx]


        return 0.5 * ff_inputs, 0.5 * ff_outputs


    def generate_ff_trials_duration(self, pulse_duration):
        inputs = np.zeros([self.n_batch, len(self.t), self.n_bits])
        #inputs[:, 0:pulse_duration, :] = [1, 1, -1]
        #inputs[:, 0:pulse_duration, :] = [1]
        pulse_list = np.arange(0, len(self.t), self.tau/self.dt, dtype=int)
        #n_pulses = int(len(pulse_list)* self.p_flop)
        n_pulses = int(np.floor(self.T * 0.01))
        t_start = self.rng.choice(pulse_list, n_pulses, replace=False)

        for j in range(self.n_batch):
            for i in range(n_pulses):
                inputs[j, t_start[i]:t_start[i] + pulse_duration, self.rng.randint(0, self.n_input)] = \
                    2 * self.rng.randint(0, 2) - 1

        outputs = np.zeros([self.n_batch, len(self.t), self.n_bits])

        for trial_indx in range(self.n_batch):
            for bit_indx in range(self.n_bits):
                input_ = np.squeeze(inputs[trial_indx, :, bit_indx])
                t_flop = np.where(input_ != 0)
                for flop_indx in range(np.size(t_flop)):
                    t_flop_i = t_flop[0][flop_indx]

                    outputs[trial_indx, t_flop_i:, bit_indx] = inputs[trial_indx, t_flop_i, bit_indx]

        return inputs, outputs


    def generate_sine_trials(self):

        freq = 1/300.
        inputs = np.zeros([self.n_batch, len(self.t), 1])
        outputs = 1. * np.reshape(np.sin(2 * np.pi * freq * self.t), [self.n_batch, len(self.t), 1])

        return inputs, outputs

    def split_to_mini_batch(self):
        """ split the whole trial of input into mini_batches ie each mini batch has the length equal to time step """

        self.in_trial, self.out_trial = self.generate_ff_trials_duration(self.pulse_dur)
        self.n_mini_batch = self.in_trial.shape[1] // self.t_step
        self.mini_batch_in = np.reshape(self.in_trial, [self.n_mini_batch, 1, self.t_step, self.n_input] )
        self.mini_batch_out = np.reshape(self.out_trial, [self.n_mini_batch, 1, self.t_step, self.n_output] )




    def train(self):
        self.z_hat_seq = np.zeros((len(self.t), self.n_output))
        self.hid_state_seq = np.zeros((len(self.t), self.N))

        #print('shape', self.z_hat_seq.shape)
        self.time_counter = np.array([])
        mb_c = 0  # minibatch counter
        self.sess.run(tf.global_variables_initializer())
        ss = 0
        self.wo_init = self.sess.run(self.w)
        #print('before update', self.sess.run(self.w))

        for i in range(len(self.t_train)):


            if i % self.t_step == 0 :

                self.Pw_, self.w_, _, _, self.z_hat_, self.final_state_, self.hidden_states_ , e_, denum = \
                    self.sess.run([self.Pw, self.w, self.update_Pw, self.update_w, self.z_hat, self.final_state, self.hidden_states,
                                   self.error, self.denum],
                              feed_dict={self.inputs_bxtxd:self.mini_batch_in[mb_c], self.init_state_1xN:self.init_state_val,
                                         self.outputs_bxtxd:self.mini_batch_out[mb_c]})


                #print('epoch =', i, 'w = ', self.w_)
                # print('iteration= ', i, ' *** e_ = ', e_)
                # next_z = np.dot(np.tanh(self.hidden_states_), self.w_)
                # print('iteration= ', i, '----Epsilon=', next_z - self.mini_batch_out[mb_c])
                # print('DENUMINATOR     ', denum)
                # print('norm readouts', np.linalg.norm(self.w_))
                # print('------------------------------------------------------------------------------------------------')


                self.init_state_val = self.final_state_
                mb_c += 1
                #print('hidden_state size', self.hidden_states_.shape)

                #print('zhatshape', np.squeeze(self.z_hat_).shape)
                self.z_hat_seq[i:i+self.t_step, :] = np.squeeze(self.z_hat_)
                #self.z_hat_seq[i:i+self.t_step, :] = (self.z_hat_)
                self.z_hat_seq_plot = self.z_hat_seq[ss:i+self.t_step, :]

                self.hid_state_seq[i:i+self.t_step, :] = np.squeeze(self.hidden_states_)
                self.hid_state_plot = self.hid_state_seq[ss:i+self.t_step, :]

                self.time_counter =  self.t[ss:i+self.t_step]
                self.in_trial_log = np.squeeze(self.in_trial[:, ss:i+self.t_step, :])
                self.out_trial_log = np.squeeze(self.out_trial[:, ss:i + self.t_step, :])
                # print('input', self.in_trial_log)
                # print('outlog', self.out_trial_log)

                # print('z_hat', self.z_hat_)
                # print('i', i)
                #
                # print('z_hat_seq_plot', self.z_hat_seq_plot)
                # # print('time', self.time_counter)
                if i % self.n_plot == 0:

                    plt.figure(num=1, figsize=(14, 8), dpi=150)
                    drawnow(self.draw_fig)
                    # plt.pause(0.5)
                    # plt.figure(num=2, figsize=(14, 8), dpi=150 )
                    # drawnow(self.draw_fig_states)
                    ss = i
                    # if i > len(self.t_train)/5:
                    #     ss = i



        self.last_mini_batch = mb_c
        self.wo_learned = self.w_
        print('-------- Training Done --------')
        print('readout norm = ', np.linalg.norm(self.w_))



    def test(self):
        ss = 0
        self.state_traj = np.zeros([self.n_batch, self.t_step, self.N])
        self.z_hat_seq = np.zeros((len(self.t), self.n_output))
        self.hid_state_seq = np.zeros((len(self.t), self.N))
        self.time_counter = np.array([])
        #self.train_steps = self.T // self.t_step
        mb_c = self.last_mini_batch  # minibatch counter

        self.init_state_val = self.final_state_

        for i in range(len(self.t_test)):
            if i % self.t_step == 0:
                z_hat_predict, f_state_predict, self.hidden_states_predict = self.sess.run([self.z_hat, self.final_state, self.hidden_states],
                                                           feed_dict={self.inputs_bxtxd:self.mini_batch_in[mb_c],
                                                                     self.init_state_1xN:self.init_state_val,
                                                                     self.outputs_bxtxd:self.mini_batch_out[mb_c]})


                self.init_state_val = f_state_predict

                mb_c += 1


                if i >= len(self.t_test) - self.traj_length:
                    self.state_traj = np.concatenate((self.state_traj, self.hidden_states_predict), axis=1)


            self.z_hat_seq[i:i + self.t_step, :] = np.squeeze(z_hat_predict)
            #self.z_hat_seq[i:i + self.t_step, :] = (z_hat_predict)
            self.z_hat_seq_plot = self.z_hat_seq[ss:i + self.t_step, :]
            self.hid_state_seq[i:i + self.t_step, :] = np.squeeze(self.hidden_states_predict)
            self.hid_state_plot = self.hid_state_seq[ss:i + self.t_step, :]

            self.time_counter = self.t_test[ss:i + self.t_step]
            off_set = len(self.t_train)
            self.in_trial_log = np.squeeze(self.in_trial[:, ss + off_set:i + self.t_step + off_set, :])
            self.out_trial_log = np.squeeze(self.out_trial[:, ss+off_set:i + self.t_step + off_set, :])

            if i %  self.n_plot == 0:
                plt.figure(num=1, figsize=(14, 8), dpi=150)
                drawnow(self.draw_fig)
                # plt.figure(num=2, figsize=(14, 8), dpi=150)
                # drawnow(self.draw_fig_states)
                ss = i


        return self.hidden_states_predict, self.state_traj[:, self.t_step:, :]


    def get_random_matrix(self):
        J, wf = self.sess.run([self.J, self.wf])
        J_eff_before = self.g * J + np.matmul(self.wo_init, wf)
        J_eff_after = self.g * J + np.matmul(self.wo_learned, wf)

        return J_eff_before, J_eff_after




    def draw_fig(self):
        #plt.xkcd()
        #vertical_spacing = 2.5
        ax1 = plt.subplot(311)
      

        ax1.step(self.time_counter, self.z_hat_seq_plot[:, 0], color='cyan')
        ax1.step(self.time_counter, self.out_trial_log[:, 0], color='purple', linestyle='--')

        ax1.fill_between(self.time_counter,  self.in_trial_log[:, 0], step='pre',color='gray')

        ax2 = plt.subplot(312)
        ax2.step(self.time_counter, self.z_hat_seq_plot[:, 1], color='cyan')
        ax2.step(self.time_counter, self.out_trial_log[:, 1], color='purple', linestyle='--')

        ax2.fill_between(self.time_counter, self.in_trial_log[:, 1], step='pre', color='gray')

        ax3 = plt.subplot(313)
        ax3.step(self.time_counter, self.z_hat_seq_plot[:, 2], color='cyan')
        ax3.step(self.time_counter, self.out_trial_log[:, 2], color='purple', linestyle='--')

        ax3.fill_between(self.time_counter, self.in_trial_log[:, 2], step='pre', color='gray')



    def draw_fig_states(self):
        #plt.xkcd()

        vertical_spacing = 2.5
        ax1 = plt.subplot(111)

        y_ticks = [vertical_spacing * neuron_indx for neuron_indx in range(self.neuron_num)]
        y_tick_labels = ['neuron %d' % n for n in self.neuron_list]
        plt.yticks(y_ticks, y_tick_labels)
        for neuron_indx in range(self.neuron_num):

            v_offset = vertical_spacing * neuron_indx


            ax1.step(self.time_counter, v_offset + self.hid_state_plot[:, self.neuron_list[neuron_indx]], color=self.color[neuron_indx])




