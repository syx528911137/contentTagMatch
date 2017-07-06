import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10


n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)


x = tf.placeholder("float", [None, None, n_input])
y = tf.placeholder("float", [None, n_classes])
step = tf.placeholder(tf.int32)

weights = {
    'in':tf.Variable(tf.random_normal([n_input,n_hidden])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'in':tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases,step):
    #print "in RNN......."
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    #print "before unstack"
    #print x.shape
    #print "after unstack"
    #x = tf.unstack(x, n_steps, 1)
    x = tf.reshape(x,[-1,n_input])
    print "***********************************************"
    print x.shape
    x_in = tf.matmul(x,weights['in']) + biases['in']
    x_in = tf.reshape(x_in,[-1,step,n_hidden])
    #print x.shape
    # Define a lstm cell with tensorflow
    cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell,input_keep_prob=1,output_keep_prob=0.5)
    # Get lstm cell output
    #outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    outputs,states = tf.nn.dynamic_rnn(lstm_cell,x_in,dtype=tf.float32,time_major=False)
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']



pred = RNN(x, weights, biases,step)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    step = 1
    test_data = mnist.test.images.reshape((-1, n_steps, n_input))
    # test_data = tf.matmul(test_data,weights['in']) + biases['in']
    # test_data = tf.reshape(test_data,[-1,n_steps,n_hidden])
    test_label = mnist.test.labels

    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,step:28})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            testacc = sess.run(accuracy, feed_dict={x: test_data, y: test_label})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc) + ", " + "Testing Accuracy:"+ "{:.5f}".format(testacc)

        step += 1
    print "Optimization Finished!"

    # Calculate accuracy for 128 mnist test images
    #test_len = 128
    #test_data = mnist.test.images.reshape((-1, n_steps, n_input))
    #test_data = tf.matmul(test_data,weights['in']) + biases['in']
    #test_data = tf.reshape(test_data,[-1,n_steps,n_hidden])
    #test_label = mnist.test.labels
    print "Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label})
