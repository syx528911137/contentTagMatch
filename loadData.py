

class LoadEmbadding:
    def __init__(self,path):

        self.filePath = path
        self.map = {}
    def loadData(self):
        print "load embadding..."
        embaddingMap = {}
        f = open(self.filePath,'r')
        line = f.readline()
        tmp = line.strip().split(" ")
        num = int(tmp[0])
        dim = int(tmp[1])
        for i in range(0,num):
            line = f.readline().strip().split(" ")
            id = line[0]
            vec = [float(line[j]) for j in range(1,len(line))]
            embaddingMap[id] = vec
            if len(vec) in self.map:
                self.map[len(vec)] += 1
            else:
                self.map[len(vec)] = 1
        print self.map

        return embaddingMap




class LoadTopicInfo:
    def __init__(self,path):

        self.topicList = [] #sorted topic id
        self.path = path
        self.maxNum_char = 0 # max char num of single topic
        self.maxNum_word = 0 # max word num of single topic
        self.map_topic_char = {}
        self.map_topic_word = {}

    def readData(self):
        print "load topic info..."
        ids = []

        f = open(self.path,'r')
        line = f.readline()
        while len(line) > 0:
            charlist = []
            wordlist = []
            contents = line.strip().split('\t')
            ids.append(contents[0].strip())
            topicDisByChar = []
            topicDisByWord = []
            topicTitleByChar = contents[2].strip().split(',')
            topicTitleByWord = contents[3].strip().split(',')
            if len(contents) > 4:
                topicDisByChar = contents[4].strip().split(',')
            if len(contents) > 5:
                topicDisByWord = contents[5].strip().split(',')
            for c in topicTitleByChar:
                charlist.append(c)
            for c in topicDisByChar:
                charlist.append(c)
            for w in topicTitleByWord:
                wordlist.append(w)
            for w in topicDisByWord:
                wordlist.append(w)
            if len(charlist) > self.maxNum_char:
                self.maxNum_char = len(charlist)
            if len(wordlist) > self.maxNum_word:
                self.maxNum_word = len(wordlist)
            self.map_topic_char[contents[0].strip()] = charlist
            self.map_topic_word[contents[0].strip()] = wordlist
            line = f.readline()
        ids.sort()
        self.topicList = ids
        print "topic : " + str(len(self.topicList))

class LoadQuestionTrainSet:
    def __init__(self,path):

        self.questionIdList=[]
        self.path = path
        self.maxNum_char = 0  # max char num of single topic
        self.maxNum_word = 0  # max word num of single topic
        self.map_question_char = {}
        self.map_question_word = {}

    def readData(self):
        print "load question info..."
        f = open(self.path,'r')
        line = f.readline()
        qidList = []
        count = 0
        fw = open('errorData.txt','w')
        while len(line) > 0:
            charlist = []
            wordlist = []
            contents = line.strip().split('\t')
            id = contents[0].strip()
            questionDisByChar = []
            questionDisByWord = []
            questionTitleByChar = contents[1].strip().split(',')
            questionTitleByWord = contents[2].strip().split(',')
            if len(contents) > 3:
                questionDisByChar = contents[3].strip().split(',')
            if len(contents) > 4:
                questionDisByWord = contents[4].strip().split(',')
            for c in questionTitleByChar:
                charlist.append(c)
            for c in questionDisByChar:
                charlist.append(c)
            for w in questionTitleByWord:
                wordlist.append(w)
            for w in questionDisByWord:
                wordlist.append(w)
            if len(charlist) > self.maxNum_char:
                self.maxNum_char = len(charlist)
            if len(wordlist) > self.maxNum_word:
                self.maxNum_word = len(wordlist)
            self.map_question_char[id] = charlist
            self.map_question_word[id] = wordlist
            qidList.append(id)
            count = count + 1
            print "line: " + str(count)
            line = f.readline()
        self.questionIdList = qidList.sort()
        fw.close()


class LoadQuestionTopicTrainSet:
    def __init__(self,path):

        self.maxNum = 0
        self.path = path
        self.map_question_topic = {}
    def readData(self):
        print "load question topic train set..."
        f = open(self.path,'r')
        line = f.readline()
        while len(line) > 0:
            contents = line.strip().split('\t')
            id = contents[0].strip()
            topics = []
            tmp = contents[1].strip().split(',')
            for topic in tmp:
                topics.append(topic)
            if len(topics) > self.maxNum:
                self.maxNum = len(topics)
            self.map_question_topic[id] = topics
            line = f.readline()

class LoadQuestionData:
    def __init__(self,filepath):
        self.questionIdList = []
        self.path = filepath
        self.maxNum_char = 0  # max char num of single question
        self.maxNum_word = 0  # max word num of single question
        self.map_question_char = {}
        self.map_question_word = {}
        self.map_question_topic = {}
        self.sequence_length = []

    def readData(self,max_length):
        print "load question training data..."
        f = open(self.path,'r')
        line = f.readline()
        qidList = []
        count = 0
        while len(line) > 0:
            # question
            charlist_q = []
            wordlist_q = []
            contents_q = line.strip().split('#')[0].strip().split('\t')
            id_q = contents_q[0].strip()
            questionDisByChar_q = []
            questionDisByWord_q = []
            questionTitleByWord_q = []
            questionTitleByChar_q = contents_q[1].strip().split(',')
            if len(contents_q) > 2:
                questionTitleByWord_q = contents_q[2].strip().split(',')
            if len(contents_q) > 3:
                questionDisByChar_q = contents_q[3].strip().split(',')
            if len(contents_q) > 4:
                questionDisByWord_q = contents_q[4].strip().split(',')
            for c in questionTitleByChar_q:
                charlist_q.append(c)
            for c in questionDisByChar_q:
                charlist_q.append(c)
            for w in questionTitleByWord_q:
                wordlist_q.append(w)
            for w in questionDisByWord_q:
                wordlist_q.append(w)
            if len(charlist_q) > self.maxNum_char:
                self.maxNum_char = len(charlist_q)
            if len(wordlist_q) > self.maxNum_word:
                self.maxNum_word = len(wordlist_q)
            length = 0
            if len(contents_q) > 2:
                length += len(contents_q[2].split(','))

            if len(contents_q) > 4:
                length += len(contents_q[4].split(','))
            self.sequence_length.append(max_length if max_length < length else length)
            self.map_question_char[id_q] = charlist_q
            self.map_question_word[id_q] = wordlist_q
            qidList.append(id_q)


            # question <-> topic
            contents = line.strip().split('#')[1].strip().split('\t')
            id = contents[0].strip()
            topics = []
            tmp = contents[1].strip().split(',')
            for topic in tmp:
                topics.append(topic)
            self.map_question_topic[id] = topics
            count += 1
            if count % 10000 == 0:
                print count
            if count > 2000:
                break
            line = f.readline()
        self.questionIdList = qidList