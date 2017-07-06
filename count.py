#-*-coding:UTF-8-*-



if __name__ == '__main__':
    f = open('/home/sunyx/local/zhihuCup/ieee_zhihu_cup/question_train_set.txt','r')
    line = f.readline()
    wordlength = {}
    while len(line) > 0:
        contents = line.strip().split('\t')
        length = len(contents[2].split(','))
        if len(contents) > 4:
            length += len(contents[4].split(','))
        if length in wordlength:
            wordlength[length] += 1
        else:
            wordlength[length] = 1

        line = f.readline()
    fw = open('question_result.txt','w')
    wordlength = sorted(wordlength.iteritems(),key=lambda d:d[0])
    for l in wordlength:
        fw.write(str(l) + "," + str(wordlength[l]))
        fw.write("\r\n")
        fw.flush()
    fw.close()