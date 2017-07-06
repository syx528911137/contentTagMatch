#-*-coding:UTF-8-*-
import loadData
import numpy as np
import random
import os
import Queue


class DataSetFeed:
    def __init__(self,charEmbaddingPath=None,wordEmbaddingPath=None,topicInfoPath=None,questionTrainSetPath=None,questionTopicTrainSetPath=None):
        self.wordEmbadding=None
        self.charEmbadding=None
        self.index = 0
        self.trainSetNum = 11500 # 文件个数
        self.testSetNum = 1 # 文件个数
        if charEmbaddingPath is not None:
            readcEmbadding = loadData.LoadEmbadding(charEmbaddingPath)
            self.charEmbadding = readcEmbadding.loadData()
        if wordEmbaddingPath is not None:
            readwEmbadding = loadData.LoadEmbadding(wordEmbaddingPath)
            self.wordEmbadding = readwEmbadding.loadData()
        self.topicInfo = loadData.LoadTopicInfo(topicInfoPath)

        self.preProcess()
    def preProcess(self):
        print "pre process : read data in topicInfo, questionTrainSet and questionTopicTrainSet"
        self.topicInfo.readData()
        print "traindata size is: " + str(self.trainSetNum) + ", test data size is : " + str(self.testSetNum)
    def nextBatch(self,batch_size): # batch size:200
        print "get next batch...."
        questionRepresentation = [] #所有词的embadding拼接起来，维度是 [batch size,词数*256]
        topicRepresentation = [] # [batch_size,[[topic_id,topic_embadding,topic_label],...]] topic_embadding: 词数*256
        topic_seq_len = []
        if self.trainSetNum == 0:
            return None

        max_num_question_word = 100
        max_num_topic_word = 100
        name = "part-" + str(self.index % self.trainSetNum)
        self.index = self.index + 1
        path = os.path.join("./chunk_data",name)
        questionData = loadData.LoadQuestionData(filepath=path)
        questionData.readData(max_num_question_word)

        for tmp_i in range(0,batch_size): #200


            # loop question index

            # get question id
            qid = questionData.questionIdList[tmp_i]
            # get words
            questionwordlist = questionData.map_question_word[qid]
            #if len(questionwordlist) > max_num_question_word:
                #max_num_question_word = len(questionwordlist)
            #add embadding of words to tmp_save_embadding
            tmp_save_embadding = []
            for word in questionwordlist:
                if word in self.wordEmbadding:

                    tmp_save_embadding.append(self.wordEmbadding[word])#(self.wordEmbadding[word] if word in self.wordEmbadding else [])
            if len(questionwordlist) > max_num_question_word:
                tmp_save_embadding = self.processEmbaddings(tmp_save_embadding,max_num_question_word)
            else:
                num = max_num_question_word - len(tmp_save_embadding)

                for need in range(0,num):

                    tmp_save_embadding.append([0]*256 )

            tmp_save_embadding = np.array(tmp_save_embadding).reshape([-1]).tolist()
            questionRepresentation.append(tmp_save_embadding)
            #get topic ids
            tids = questionData.map_question_topic[qid]
            while len(tids) < 30:
                sample_ids = random.randint(0,1998)
                #print "sample:" + str(sample_ids)
                genTopicId = self.topicInfo.topicList[sample_ids]
                if genTopicId not in tids:
                    tids.append(genTopicId)

            pairs = [] # [[topic_id,topic_embadding,topic_label],[],...]
            tmp = []
            for tid in tids:
                topicwordlist = self.topicInfo.map_topic_word[tid]
                tmp.append(len(topicwordlist) if len(topicwordlist) < max_num_topic_word else max_num_topic_word)
                # if len(topicwordlist) > max_num_topic_word:
                #     max_num_topic_word = len(topicwordlist)

                single_topic_pair = [] # topic id, topic embadding, topic label
                tmp_topic_wordembadding = []
                for tword in topicwordlist:
                    if tword in self.wordEmbadding:

                        tmp_topic_wordembadding.append(self.wordEmbadding[tword])#(self.wordEmbadding[tword] if tword in self.wordEmbadding else [])
                if len(topicwordlist) > max_num_topic_word:
                    tmp_topic_wordembadding = self.processEmbaddings(tmp_topic_wordembadding,max_num_topic_word)
                else:
                    num = max_num_topic_word - len(tmp_topic_wordembadding)
                    for need in range(0,num):
                        tmp_topic_wordembadding.append([0] * 256 )
                tmp_topic_wordembadding = np.array(tmp_topic_wordembadding).reshape([-1]).tolist()

                single_topic_pair.append(tid)
                single_topic_pair.append(tmp_topic_wordembadding)
                single_topic_pair.append(1 if tid in questionData.map_question_topic[qid] else 0)
                pairs.append(single_topic_pair)
            topic_seq_len.append(tmp) # batch size , 30
            topicRepresentation.append(pairs)

        # #补0
        # for i in range(0,len(questionRepresentation)):
        #     questionRepresentation[i].extend([0 for m in range (0,max_num_question_word * 256 - len(questionRepresentation[i]))])
        #     topic_pairs = topicRepresentation[i]
        #     for j in range(0,len(topic_pairs)):
        #         topic_pairs[j][1].extend([0 for n in range(0,max_num_topic_word * 256 - len(topic_pairs[j][1]))])


        topEmbadding = [] # batch size X 30 ， 词数 X 256
        labels = [] # batch size X 30，1
        for i in range(0,len(topicRepresentation)):
            for j in range(0,len(topicRepresentation[i])):
                topEmbadding.append(topicRepresentation[i][j][1])
                labels.append(topicRepresentation[i][j][2])

        print "get batch end"
        #print np.array(labels).reshape([-1]).shape
        return questionRepresentation,topEmbadding,labels,questionData.sequence_length,topic_seq_len # 问题表示，话题表示，标签，问题长度列表，话题长度列表

    def getTestSet(self):
        print "getTestData..."
        questionRepresentation = []  # 所有词的embadding拼接起来，维度是 [测试集数量 ， 词数*256]
        relatedTopicsIndex = []  # [[topic ids],[]]
        max_num_question_word = 100
        max_num_topic_word = 100
        name = "part-11500"
        path = os.path.join("./chunk_data", name)
        questionData = loadData.LoadQuestionData(filepath=path)
        questionData.readData(max_num_question_word)
        self.testSetNum = len(questionData.questionIdList)
        print "read data end, data set size is " + str(self.testSetNum)
        print "process test data..."



        for index in range(0,len(questionData.questionIdList)):
            if index % 500 == 0:
                print "process embadding:" + str(index)

            qid = questionData.questionIdList[index]
            #print "qid: " + str(qid)
            question_word_list = questionData.map_question_word[qid]
            # if len(question_word_list) > max_num_question_word:
            #     max_num_question_word = len(question_word_list)
            questionVectors = []
            for wordid in question_word_list:
                if wordid in self.wordEmbadding:

                    questionVectors.append(self.wordEmbadding[wordid])#(self.wordEmbadding[wordid] if wordid in self.wordEmbadding else [])

            #print "extend..., vectors size : " + str(len(questionVectors))
            if len(questionVectors) > max_num_question_word:
                #print "reduce..."
                questionVectors = self.processEmbaddings(questionVectors,max_num_question_word)
            else:
                num = max_num_question_word - len(questionVectors)
                #print "extend..." +str(len(questionVectors)) + "," + str(num)

                for need in range(0,num):
                    questionVectors.append([0 for nn in range(0, 256)] )

            questionVectors = np.array(questionVectors)
            #print questionVectors.shape
            questionVectors = questionVectors.reshape([-1]).tolist()

            questionRepresentation.append(questionVectors)
            #topicIdIndex = [0] * 1999
            #print "questionVectors size is : " + str(len(questionVectors))
            topicIds = questionData.map_question_topic[qid]
            # for tid in topicIds:
            #     tmpIndex = self.topicInfo.topicList.index(tid)
            #     #topicIdIndex[tmpIndex] = 1

            relatedTopicsIndex.append(topicIds)
        # print "add 0..."
        # for i in range(0,len(questionRepresentation)):
        #     if i % 500 == 0:
        #         print "processed:"+str(i)
        #     questionRepresentation[i].extend([0 for m in range (0,max_num_question_word * 256 - len(questionRepresentation[i]))])



        print "get test data end"
        return questionRepresentation,relatedTopicsIndex,questionData.sequence_length # 问题的序列长度

    def combainEmbadding(self, embaddings, max_word):
        # print "embadding size is: " + str(len(embaddings))
        queue = Queue.Queue()
        vectors = []
        for vec in embaddings:
            queue.put(vec)
        while queue.qsize() + len(vectors) > max_word and queue.qsize() > 1:
            # print "queue size:" + str(queue.qsize())
            a = queue.get()
            b = queue.get()

            a = np.array(a)
            b = np.array(b)
            c = a + b
            vectors.append(c.tolist())
        while queue.qsize() > 0:
            vectors.append(queue.get())
        return vectors

    def processEmbaddings(self, vector, max_word):
        while len(vector) > max_word:
            vector = self.combainEmbadding(vector, max_word)
        return vector




    def getAllTopicRepresentation(self):
        print "read topic representation..."
        topicRepresentation = []
        max_topic_word = 100
        sequence_length = []
        for i in range(0,len(self.topicInfo.topicList)):
            tid = self.topicInfo.topicList[i]
            topicwordlist = self.topicInfo.map_topic_word[tid]
            sequence_length.append(max_topic_word if len(topicwordlist) > max_topic_word else len(topicwordlist))
            # if len(topicwordlist) > max_topic_word:
            #     max_topic_word = len(topicwordlist)
            vector = []
            for wid in topicwordlist:
                if wid in self.wordEmbadding:
                    vector.append(self.wordEmbadding[wid])#(self.wordEmbadding[wid] if wid in self.wordEmbadding else [])
            if len(topicwordlist) > max_topic_word:
                vector = self.processEmbaddings(vector,max_topic_word)
            else:
                num = max_topic_word - len(vector)
                # print "extend..." +str(len(questionVectors)) + "," + str(num)

                for need in range(0, num):
                    vector.append([0] * 256)
            vector = np.array(vector).reshape([-1]).tolist()
            topicRepresentation.append(vector)


        print "read topic representation end"
        return topicRepresentation,sequence_length # 话题的序列长度







