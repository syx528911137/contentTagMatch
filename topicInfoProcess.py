#-*-coding:UTF-8-*-
import tensorflow as tf
import numpy as np
import sys
import Queue


if __name__ == '__main__':

    # queue = Queue.Queue()
    # queue.put(1)
    # queue.put(2)
    # print queue.qsize()
    # queue.put(3)
    # print queue.qsize()
    # print queue.get()
    c = [0] * 10
    print c
    a = [1,2,3]
    b = [4,5,6]
    # print a + b
    # print np.array(a) + np.array(b)
    test = [[1,2,3],[4,5,6]]
    print test
    test = np.array(test).reshape([-1]).tolist()
    print test
    m = {1:2}
    print m
    # a = tf.Variable(tf.random_normal([1,2,3]))
    # init = tf.global_variables_initializer()
    # b = tf.constant([[1.0,2.0],[2.0,2.0],[3.0,3.0]])
    # sess = tf.Session()
    # sess.run(init)
    # print sess.run(b)
    # #b = tf.tile(b, [1, 1+2])
    # b = tf.square(b)
    # #b = tf.reshape(b,[-1,2])
    # print "---------------------"
    # print sess.run(b)
    # b = tf.constant([[1,3,5,2],[5,8,4,3],[6,4,1,9]])
    # sess = tf.Session()
    # # d = tf.tile(b,[2,1])
    # #print sess.run(b)
    # a = tf.unstack(b)
    # a = sess.run(a)
    # c = [np.ndarray.tolist(a[i]) for i in range(0,len(a))]
    # #c = sorted(a)
    # c = np.array(c)
    # d = np.argsort(c,axis=1)
    # print d
    # e = np.array([d[j][0:2] for j in range(0,len(d))])
    # print e
    #
    # c = tf.constant([0.,1.,0.,0.,1.,0.,0.])
    # d = tf.constant([0.,0.,0.,0.,1.,0.,0.])
    # #e = tf.reduce_sum(tf.multiply(c,d),axis=1,keep_dims=True)
    # e = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=c,logits=d))
    # print sess.run(e)
    # a = [[[1,2,3],[4,5,6],[7,8,9],[10,11,12]],[[11,22,33],[44,55,66],[77,88,99],[100,111,122]]]
    #
    # b = [0]*5
    # b[1] = 1
    #
    # print b
