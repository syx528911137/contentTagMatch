#-*-coding:UTF-8-*-
import os





if __name__ == '__main__':


    qt_train = open('/home/sunyx/local/zhihuCup/ieee_zhihu_cup/question_topic_train_set.txt','r')

    q_train = open('/home/sunyx/local/zhihuCup/ieee_zhihu_cup/question_train_set.txt','r')
    line_qt = qt_train.readline()
    line_q = q_train.readline()
    part = 0
    name = "part-" + str(part)
    result = open(os.path.join("chunk_data",name),'w')
    count = 0
    flag = False
    while len(line_q) > 0 and len(line_qt) > 0:
        if count % 200 == 0 and count != 0 and count <= 2300000 and flag:
            result.close()
            part += 1
            name = "part-" + str(part)
            result = open(os.path.join("chunk_data",name),'w')
            flag = False

        if count % 100000 == 0:

            print "line: " + str(count)
        contents = line_q.strip().split('\t')

        length = 0
        if len(contents) > 2:
            length += len(contents[2].split(','))

        if len(contents) > 4:
            length += len(contents[4].split(','))
        if length <= 600:
            a = line_q.split('\t')[0].strip()
            b = line_qt.split('\t')[0].strip()
            if a == b:
                wcontent = line_q.strip() + "#" + line_qt.strip()
                result.write(wcontent)
                result.write('\r\n')
                result.flush()
                flag = True
                count += 1
        line_qt = qt_train.readline()
        line_q = q_train.readline()
    result.close()




