import tensorflow as tf
import numpy as np


class QuestionRNN:
    def __init__(self):
        self.batch_size = 200
        self.max_steps = 100
        self.input_dim = 256
        self.cell_size = 1024
        self.lr = 0.001
        self.question_input_data = tf.placeholder(tf.float32, [None, self.max_steps, self.input_dim],name="question_input_data")
        self.question_sequenceLength = tf.placeholder(tf.int32, [None], name="question_sequenceLength")
        self.cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias=1.0),output_keep_prob=0.75)
        self.weight = tf.Variable(tf.random_normal([self.input_dim,self.cell_size]),name="weight_q_in")
        self.bias = tf.Variable(tf.random_normal([self.cell_size]),name="bias_q_in")



    def buildTrainNet(self):
        question_input = self.question_input_data
        x = tf.reshape(question_input, [-1, self.input_dim])
        x_in = tf.matmul(x, self.weight) + self.bias
        x_in = tf.reshape(x_in, [-1, self.max_steps, self.cell_size])
        outputs, states = tf.nn.dynamic_rnn(self.cell, x_in, dtype=tf.float32, sequence_length=self.question_sequenceLength,
                                            time_major=False)
        outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
        question_output = outputs[-1]
        question_output_extend = tf.tile(question_output, [1, 30])  # batch, 30 X 1024
        question_output_extend = tf.reshape(question_output_extend, [-1, self.cell_size])  # batch X 30 , 1024




