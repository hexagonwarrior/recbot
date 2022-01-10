#!/usr/bin/env python3

import pymysql
import jieba.analyse

import rl


HOST = '127.0.0.1'
USER = 'rec'
PASSWORD = 'shujuku'
PORT = 3306
DATABASE = 'news_with_keyword'
CHAREST = 'utf8'

CATEGORY = {
    1: "Politics",
    2: "Entertainment", 
    3: "Sports",
    4: "Games", 
    5: "Technology", 
    6: "Business",
    7: "Society",
    8: "Military"
}

# 根据推荐结果从数据库中随机取内容

def get_rec_datalist():

    # STEP0
    datalist = []
    cnn = pymysql.connect(host=HOST, user=USER, password=PASSWORD, port=PORT, database=DATABASE,
                          charset=CHAREST)
    cursor = cnn.cursor()
    base_sql = 'select * from guanchazhe'

    # STEP1 更新evn，并调用test.py获取推荐动作
    rec_result = rl.get_rec_result()
    print("REC_RESULT = ", rec_result)
    
    # 分类记数
    d = {}
    for c in rec_result:
        if c in d:
            d[c] += 1
        else:
            d[c] = 1
    
    # STEP2 
    for k in d.keys():
        category = CATEGORY[k]
        sql = base_sql + " WHERE author='" + category + "' ORDER BY rand() LIMIT " + str(d[k])
        
        print(sql)
        cursor.execute(sql)
        for item in cursor.fetchall():
            datalist.append(item)

    # STEP3
    cursor.close()
    cnn.close()
    return datalist



# 连接数据库并提取数据库内容
def get_datalist():
    datalist = []
    cnn = pymysql.connect(host=HOST, user=USER, password=PASSWORD, port=PORT, database=DATABASE,
                          charset=CHAREST)
    cursor = cnn.cursor()
    sql = 'select * from guanchazhe ORDER BY publish_time DESC'
    cursor.execute(sql)
    for item in cursor.fetchall():
        datalist.append(item)
    cursor.close()
    cnn.close()
    return datalist


# 对数据库文本内容进行分词，并返回 data_inf0 = [新闻数，词云数，词汇数，作者人数] ->首页展示的三个内容
def get_datalist_info(datalist):
    text = ""
    for item in datalist:
        text = text + item[4]

    # 分词
    cut = jieba.cut(text)
    string = ' '.join(cut)
    data_info = [len(datalist), 1, len(string), 1]
    return data_info,string


# 对输入文本进行分词，并返回词汇权重
def get_word_weights(string, topK):
    words = []
    weights = []
    for x, w in jieba.analyse.textrank(string, withWeight=True, topK=topK):
        words.append(x)
        weights.append(w)
    return words,weights


# 文本关键字提取
def get_keyword_from_content(content):
    print(content)
    cut = jieba.cut(content)
    string = ' '.join(cut)
    words,_=get_word_weights(string, topK=5)
    return words.append('（自动生成）')
