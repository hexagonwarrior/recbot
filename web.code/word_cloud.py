#!/usr/bin/env python3

# import jieba        #分词
from matplotlib import pyplot as plt    #绘图，数据可视化
from wordcloud import WordCloud         #词云
from PIL import Image                   #图片处理
import numpy as np                      #矩阵运算
import pymysql                          #数据库
import jieba.analyse


#准备词云所需的文字（词）

# stopwords = {}.fromkeys(['的', '了', '将'])
stopwords = {}.fromkeys([k.strip() for k in open('stopwords.list', encoding="utf-8").readlines() if k.strip() != ''])


datalist = []
cnn = pymysql.connect(host='127.0.0.1', user='rec', password='shujuku', port=3306, database='news_with_keyword',
                      charset='utf8')
cursor = cnn.cursor()
sql = ' select * from guanchazhe'
cursor.execute(sql)
for item in cursor.fetchall():
    datalist.append(item)
cursor.close()
cnn.close()

text = " "
# 取出数据库中的新闻标题，进行分词
for item in datalist:
    text =  text + item[1]
cut = jieba.cut(text)

# 去停用词
string = ""
for item in cut:
    if item not in stopwords:
        string += item + " "

img = Image.open(r'static/assets/img/tree.jpg')   #打开遮罩图片
img_array = np.array(img)   #将图片转换为数组
wc = WordCloud(
    background_color='white',
    mask=img_array,
    font_path="wqy-microhei.ttc",
    width="400",
    height=400
)
wc.generate_from_text(string)

#绘制图片
fig = plt.figure(1)
plt.imshow(wc)
plt.axis('off')     #是否显示坐标轴

# plt.show()    #显示生成的词云图片

#输出词云图片到文件
plt.savefig(r'static/assets/img/key_word.jpg',dpi=500)
