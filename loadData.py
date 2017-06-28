


class LoadEmbadding:
    def __init__(self,path):
        self.filePath = path
    def loadData(self):
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
        ids = []

        f = open(self.path,'r')
        line = f.readline()
        while len(line) > 0:
            charlist = []
            wordlist = []
            contents = line.strip().split('\t')
            ids.append(contents[0].strip())
            topicTitleByChar = contents[2].strip().split(',')
            topicTitleByWord = contents[3].strip().split(',')
            topicDisByChar = contents[4].strip().split(',')
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

class LoadQuestionTrainSet:
    def __init__(self,path):
        self.path = path
        self.maxNum_char = 0  # max char num of single topic
        self.maxNum_word = 0  # max word num of single topic
        self.map_question_char = {}
        self.map_question_word = {}

    def readData(self):
        f = open(self.path,'r')
        line = f.readline()
        while len(line) > 0:
            charlist = []
            wordlist = []
            contents = line.strip().split('\t')
            id = contents[0].strip()
            questionTitleByChar = contents[1].strip().split(',')
            questionTitleByWord = contents[2].strip().split(',')
            questionDisByChar = contents[3].strip().split(',')
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
            line = f.readline()

class LoadQuestionTopicTrainSet:
    def __init__(self,path):
        self.path = path
        self.map_question_topic = {}
    def readData(self):
        f = open(self.path,'r')
        line = f.readline()
        while len(line) > 0:
            contents = line.strip().split('\t')
            id = contents[0].strip()
            topics = []
            tmp = contents[1].strip().split(',')
            for topic in tmp:
                topics.append(topic)
            self.map_question_topic[id] = topics
            line = f.readline()

class DataFormat:
    def __init__(self,embadding):
        self.embadding = embadding

    #def
