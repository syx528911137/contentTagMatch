#-*-coding:UTF-8-*-
import tensorflow as tf
import numpy as np
import loadData
import dataFeed
from tensorflow.contrib import rnn
import os
import math

learning_rate = 0.01
batch_size = 200
max_steps = 100
display_step = 10
question_in_dim = 256
topic_in_dim = 256
question_hidden_dim = 1024
topic_hidden_dim = 1024

dataSet = dataFeed.DataSetFeed(wordEmbaddingPath="/home/sunyx/local/zhihuCup/ieee_zhihu_cup/word_embedding.txt",
                               topicInfoPath="/home/sunyx/local/zhihuCup/ieee_zhihu_cup/topic_info.txt",
                               questionTrainSetPath="/home/sunyx/local/zhihuCup/ieee_zhihu_cup/question_train_set.txt",
                               questionTopicTrainSetPath="/home/sunyx/local/zhihuCup/ieee_zhihu_cup/question_topic_train_set.txt"
                               )



weight={
    'q_in':tf.Variable(tf.random_normal([question_in_dim,question_hidden_dim]),name="weight_q_in"),
    't_in':tf.Variable(tf.random_normal([topic_in_dim,topic_hidden_dim]),name="weight_t_in")
}

bias={
    'q_in':tf.Variable(tf.random_normal([question_hidden_dim]),name="bias_q_in"),
    't_in':tf.Variable(tf.random_normal([topic_hidden_dim]),name="bias_t_in")
}
#---------train input data ------------
question_input_data = tf.placeholder(tf.float32,[None,max_steps,question_in_dim],name="question_input_data")
topic_input_data = tf.placeholder(tf.float32,[None,max_steps,topic_in_dim],name="topic_input_data")
question_sequenceLength = tf.placeholder(tf.int32,[None],name="question_sequenceLength")
topic_sequenceLength = tf.placeholder(tf.int32,[None],name="topic_sequenceLength")
label_input = tf.placeholder(tf.float32,[None],name="label_input")

vs = tf.trainable_variables()
print "node num 1: " + str(len(vs))
# print 'There are %d train_able_variables in the Graph: ' % len(vs)
# for v in vs:
#     print v


q_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(question_hidden_dim,forget_bias=1.0),output_keep_prob=0.75)


vs = tf.trainable_variables()
print "node num 2: " + str(len(vs))
# print 'There are %d train_able_variables in the Graph: ' % len(vs)
# for v in vs:
#     print v



t_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(question_hidden_dim,forget_bias=1.0),output_keep_prob=0.75)


vs = tf.trainable_variables()
print "node num 3: " + str(len(vs))
# print 'There are %d train_able_variables in the Graph: ' % len(vs)
# for v in vs:
#     print v



# print q_cell,t_cell
def question_rnn(question_input,weight,bias,sequence_length): # batch X steps X 256
    # print "------------------question input:" + str(x.shape)
    # with tf.name_scope('qrnn_inlayer') as scope_qin:
    x = tf.reshape(question_input, [-1, question_in_dim])
    #x = question_input.reshape([-1,question_in_dim])
    x_in = tf.matmul(x,weight['q_in']) + bias['q_in']
    x_in = tf.reshape(x_in,[-1,max_steps,question_hidden_dim])
    # with tf.name_scope('q_rnn_cell') as scope_qcell:
    #cell = tf.nn.rnn_cell.BasicLSTMCell(question_hidden_dim,forget_bias=1.0,reuse=True)

    # lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=q_cell,output_keep_prob=0.5)
    outputs, states = tf.nn.dynamic_rnn(q_cell, x_in, dtype=tf.float32, sequence_length=None,time_major=False)
    # print "question outputs shape: " + str(outputs.shape)
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))

    return outputs[-1] # batch , 1024


def topic_rnn(topic_input,weight,bias,sequence_length): # batch X 30 X step X 256
    # with tf.name_scope('trnn_inlayer',) as scope_tin:
    x = tf.reshape(topic_input,[-1,topic_in_dim])
    #x = topic_input.reshape()
    x_in = tf.matmul(x,weight['t_in']) + bias['t_in']
    x_in = tf.reshape( x_in,[-1,max_steps,topic_hidden_dim])

    sequence_length = tf.reshape(sequence_length,[-1])
    #print tf.shape(x_in), tf.shape(sequence_length)
    # with tf.name_scope('t_rnn_cell') as scope_tcell:
    # cell = tf.nn.rnn_cell.BasicLSTMCell(topic_hidden_dim,forget_bias=1.0,reuse=True)
    # lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell,output_keep_prob=0.5)
    outputs,states = tf.nn.dynamic_rnn(t_cell,x_in,dtype=tf.float32,sequence_length=None,time_major=False)
    outputs = tf.unstack(tf.transpose(outputs,[1,0,2]))
    return outputs[-1] # batch X 30 , 1024



def getTrainSim(question_input,topic_input,question_sequenceLength,topic_sequenceLength,weight,bias):
    # question_steps =
    # topic_steps = int(topic_steps)
    with tf.variable_scope('q_rnn') as qscope:
        try:
            question_output = question_rnn(question_input, weight, bias, question_sequenceLength)  # batch , 1024
        except:
            tf.get_variable_scope().reuse_variables()
            question_output= question_rnn(question_input,weight,bias,question_sequenceLength) # batch , 1024
    vs = tf.trainable_variables()
    print "node num q_train: " + str(len(vs))
    # print 'There are %d train_able_variables in the Graph: ' % len(vs)
    # for v in vs:
    #     print v

    with tf.variable_scope('t_rnn'):
        try:
            topic_output = topic_rnn(topic_input,weight,bias,topic_sequenceLength)
        except:
            tf.get_variable_scope().reuse_variables()
            topic_output = topic_rnn(topic_input, weight, bias, topic_sequenceLength)
    vs = tf.trainable_variables()
    print "node num t_train: " + str(len(vs))
    #------------------------------
    # vs = tf.trainable_variables()
    # print 'There are %d train_able_variables in the Graph: ' % len(vs)
    # for v in vs:
    #     print v
    #-----------------------------
    topic_output = tf.reshape(topic_output,[-1,topic_hidden_dim]) # batch X 30 , 1024

    question_output_extend = tf.tile(question_output,[1,30]) # batch, 30 X 1024
    question_output_extend = tf.reshape(question_output_extend,[-1,question_hidden_dim])#batch X 30 , 1024

    # element wise multiply
    res_elem_wise_mul = tf.multiply(question_output_extend,topic_output) #batch X 30,1024
    res_vector_dot_priduct = tf.reduce_sum(res_elem_wise_mul,axis=1)# batch X 30 , 1

    res_square_sum_question = tf.reduce_sum(tf.square(question_output_extend),axis=1)
    res_square_sum_topic = tf.reduce_sum(tf.square(topic_output),axis=1)
    res_tmp = tf.multiply(res_square_sum_question,res_square_sum_topic)
    res_cosSim = tf.nn.sigmoid(tf.truediv(res_vector_dot_priduct,res_tmp))
    # res_cosSim = tf.truediv(res_vector_dot_priduct,res_tmp)
    #return res_cosSim # batch X 30
    # question_output = tf.reshape(question_output,[-1,1024])
    # question_output = tf.reduce_sum(question_output,axis=1)
    # question_output = tf.reshape(question_output,[100,20])
    return res_cosSim






pred = getTrainSim(question_input=question_input_data,
                   topic_input=topic_input_data,
                   question_sequenceLength=question_sequenceLength,
                   topic_sequenceLength=topic_sequenceLength,
                   weight=weight,bias=bias)


cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_input,logits=pred))

train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

testQuestiondata = tf.placeholder(tf.float32,[None,max_steps,question_in_dim],name="testQuestiondata")
testTopicdata = tf.placeholder(tf.float32,[1999,max_steps,topic_in_dim],name="testTopicdata")
Q_sequenceLength = tf.placeholder(tf.int32,[None],name="Q_sequenceLength")
T_sequenceLength = tf.placeholder(tf.int32,[None],name="T_sequenceLength")
extend_size = tf.placeholder(tf.int32,name="topic_pred_extend")

def predict(testQdata,testTdata,weight,bias,question_seqLength,topic_seqLength,e_size):
    with tf.variable_scope('q_rnn'):
        try:
            q_res = question_rnn(testQdata,weight,bias,question_seqLength) # test data size , 1024
        except:
            tf.get_variable_scope().reuse_variables()
            q_res = question_rnn(testQdata, weight, bias, question_seqLength)  # test data size , 1024
    vs = tf.trainable_variables()
    print 'predict There are %d train_able_variables in the Graph: ' % len(vs)


    # topic_seqLength = tf.
    with tf.variable_scope('t_rnn'):
        try:
            t_res = topic_rnn(testTdata,weight,bias,topic_seqLength) # 1999 , 1024
        except:
            tf.get_variable_scope().reuse_variables()
            t_res = topic_rnn(testTdata, weight, bias, topic_seqLength)  # 1999 , 1024
    vs = tf.trainable_variables()
    print 'predict There are %d train_able_variables in the Graph: ' % len(vs)



    #--------------------------
    # vs = tf.trainable_variables()
    # print 'There are %d train_able_variables in the Graph: ' % len(vs)
    # for v in vs:
    #     print v
    #----------------------------
    q_res = tf.reshape( q_res,[-1,question_hidden_dim]) # test size,1024
    t_res = tf.reshape( t_res,[-1,topic_hidden_dim]) # 1999 , 1024

    q_extend = tf.tile(q_res,[1,1999]) # test size , 1999 X 1024
    q_extend = tf.reshape(q_extend,[-1,question_hidden_dim]) #  test size X 1999, 1024
    # print "test data size: " + str(dataSet.testSetNum)
    t_res = tf.tile(t_res,[e_size,1]) #1999 X test size,1024
    t_res = tf.reshape(t_res,[-1,topic_hidden_dim])#

    res_elem_wise_mul = tf.multiply(q_extend,t_res) # test size X1999, 1024
    res_dot_product = tf.reduce_sum(res_elem_wise_mul,axis=1) # test size x 1999,1

    res_square_question_sum = tf.reduce_sum(tf.square(q_extend),axis=1)
    res_square_sum_topic = tf.reduce_sum(tf.square(t_res),axis=1)
    res_tmp = tf.multiply(res_square_question_sum,res_square_sum_topic)
    res_cosSim = tf.nn.sigmoid(tf.truediv(res_dot_product,res_tmp))# test size X 1999


    res_cosSim = tf.reshape(res_cosSim,[-1,1999]) # test size , 1999

    return res_cosSim # test size ， 1999


def scoreEval(predict_label_and_marked_label_list):
    """
    :param predict_label_and_marked_label_list: 一个元组列表。例如
    [ ([1, 2, 3, 4, 5], [4, 5, 6, 7]),
      ([3, 2, 1, 4, 7], [5, 7, 3])
     ]
    需要注意这里 predict_label 是去重复的，例如 [1,2,3,2,4,1,6]，去重后变成[1,2,3,4,6]

    marked_label_list 本身没有顺序性，但提交结果有，例如上例的命中情况分别为
    [0，0，0，1，1]   (4，5命中)
    [1，0，0，0，1]   (3，7命中)

    """
    right_label_num = 0  # 总命中标签数量
    right_label_at_pos_num = [0, 0, 0, 0, 0]  # 在各个位置上总命中数量
    sample_num = 0  # 总问题数量
    all_marked_label_num = 0  # 总标签数量
    for predict_labels, marked_labels in predict_label_and_marked_label_list:
        sample_num += 1
        marked_label_set = set(marked_labels)
        all_marked_label_num += len(marked_label_set)
        for pos, label in zip(range(0, min(len(predict_labels), 5)), predict_labels):
            if label in marked_label_set:  # 命中
                right_label_num += 1
                right_label_at_pos_num[pos] += 1

    precision = 0.0
    for pos, right_num in zip(range(0, 5), right_label_at_pos_num):
        precision += ((right_num / float(sample_num))) / math.log(2.0 + pos)  # 下标0-4 映射到 pos1-5 + 1，所以最终+2
    recall = float(right_label_num) / all_marked_label_num
    tmp = precision + recall
    if tmp == 0:
        tmp += 1
    return (precision * recall) / tmp

def evalDataConstruction(predLaels,trueLabels):
    topicList = dataSet.topicInfo.topicList
    # predLaelsID = []
    evalData = []
    for i in range(0,len(predLaels)):
        tmp = []
        tmp.append([topicList[predLaels[i][j]] for j in range(0,len(predLaels[i]))])
        tmp.append(trueLabels[i])
        evalData.append(tuple(tmp))

    return evalData



#--------------------------------------------
test_pred = predict(testQdata=testQuestiondata,
                                testTdata=testTopicdata,
                                weight=weight,bias=bias,
                                question_seqLength=Q_sequenceLength,
                                topic_seqLength=T_sequenceLength,e_size=extend_size)

#-------------------------------------------
saver = tf.train.Saver()
init = tf.global_variables_initializer()

'''

question_input_data = tf.placeholder(tf.float32,[None,None,question_in_dim])
topic_input_data = tf.placeholder(tf.float32,[None,None,topic_in_dim])
question_steps = tf.placeholder(tf.float32,[None])
topic_steps = tf.placeholder(tf.float32,[None])
label_input = tf.placeholder(tf.float32,[None,1])
'''
with tf.Session() as sess:
    sess.run(init)
    train_step = 0
    testSet_question_data,test_topic_labels,test_question_sequence_length = dataSet.getTestSet() # 测试数据表示，相关话题ID，长度列表
    testSet_topic_data,test_topic_sequence_length = dataSet.getAllTopicRepresentation()
    test_question_sequence_length = np.array(test_question_sequence_length)
    test_topic_sequence_length = np.array(test_topic_sequence_length)
    test_topic_sequence_length = test_topic_sequence_length.reshape([-1])
    testSet_question_data = np.array(testSet_question_data)
    print testSet_question_data.shape
    testSet_question_data = testSet_question_data.reshape([-1,max_steps,question_in_dim]) # 测试集大小 , 词数 ,256
    testSet_topic_data = np.array(testSet_topic_data).reshape([1999,max_steps,topic_in_dim]) # 1999 , 词数 , 256
    while train_step * batch_size < 4000000:

        train_batch_question_data, train_batch_topic_data ,train_labels,train_question_seq_len,train_topic_seq_len = dataSet.nextBatch(20)
        train_batch_question_data = np.array(train_batch_question_data)
        train_batch_topic_data = np.array(train_batch_topic_data)
        train_labels = np.array(train_labels)
        train_question_seq_len = np.array(train_question_seq_len)
        train_topic_seq_len = np.array(train_topic_seq_len)
        train_topic_seq_len = train_topic_seq_len.reshape([-1])
        train_batch_question_data = train_batch_question_data.reshape([-1,max_steps,question_in_dim])
        # train_batch_question_data = train_batch_question_data.reshape([-1,question_in_dim])
        train_batch_topic_data = train_batch_topic_data.reshape([-1,max_steps,topic_in_dim])
        # train_batch_topic_data = train_batch_topic_data.reshape([-1,topic_in_dim])
        # print "train data"
        # print train_batch_question_data
        # print "train seq len"
        # print train_question_seq_len
        # print "rnn result"
        res =  sess.run(pred,feed_dict={question_input_data:train_batch_question_data,
                                     topic_input_data:train_batch_topic_data,
                                     question_sequenceLength:train_question_seq_len,
                                     topic_sequenceLength:train_topic_seq_len,
                                     label_input:train_labels})
        # print res
        # print res[99]
        # print res[98]
        # print res[97]
        # print res[96]
        # print res[95]
        # print res[94]
        # print res[93]
        # print res[92]
        # print res[91]
        # print res[90]
        # print res[89]
        # print res[88]
        # print res.shape





        sess.run(train_op,feed_dict={question_input_data:train_batch_question_data,
                                     topic_input_data:train_batch_topic_data,
                                     question_sequenceLength:train_question_seq_len,
                                     topic_sequenceLength:train_topic_seq_len,
                                     label_input:train_labels})

        # break
        train_step = train_step + 1
        # if train_step % 10 < 10:

        result = sess.run(test_pred,feed_dict={testQuestiondata:testSet_question_data,
                                               testTopicdata:testSet_topic_data,
                                               Q_sequenceLength:test_question_sequence_length,
                                               T_sequenceLength:test_topic_sequence_length,
                                               extend_size:dataSet.testSetNum
                                               })

        tmpres = tf.unstack(result)
        result = sess.run(tmpres)
        result_list = [np.ndarray.tolist(result[i]) for i in range(0,len(result))]
        result_array = np.array(result_list)
        result_sorted = np.argsort(result_array,axis=1)
        result_final = np.array([result_sorted[j][0:5] for j in range(0,len(result_sorted))])
        score = scoreEval(evalDataConstruction(result_final,test_topic_labels))
        if not os.path.exists("models/save"):
            os.mkdir("models/save")
        name = "modelOfTrainingStep_"+ str(train_step * batch_size) + ".ckpt"
        if train_step % 10 == 0:
            saver.save(sess=sess,save_path=os.path.join("./models/save",name))
        training_cost = sess.run(cost,feed_dict={question_input_data:train_batch_question_data,
                                 topic_input_data:train_batch_topic_data,
                                 question_sequenceLength:train_question_seq_len,
                                 topic_sequenceLength:train_topic_seq_len,
                                 label_input:train_labels})
        print "training steps:" + str(train_step * batch_size) + ", public score:  " + str(score) + ", training cost: " + str(training_cost)
