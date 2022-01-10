#!/usr/bin/env python3

import requests
import pymysql
from lxml import etree
import threading
import re
import useful_functions
import fake_user_agent

# 代理池
headers = fake_user_agent.useragent_random()

# 爬取线程
class Spider():
    def __init__(self):
        self.urls = []

        # 连接Mysql数据库
        self.cnn = pymysql.connect(host='127.0.0.1', user='rec', password='shujuku', port=3306, database='news_with_keyword', charset='utf8')
        self.cursor = self.cnn.cursor()
        self.sql = 'insert into guanchazhe(title, author, publish_time, content, url, key_word) values(%s, %s, %s, %s, %s, %s)'

        # 获取已爬取的url数据并写入列表，用于判断
        sql = 'select url from guanchazhe'
        self.cursor.execute(sql)
        for url in self.cursor.fetchall():
            self.urls.append(url[0])

    def get_info(self, url, title):
        item = {}
        if self.check_url(url):
            item['author'] = 'Business'
            item['url'] = url
            item['title'] = title
            item['publish_time'] = ''
            item['content'] = ''
            item['key_word'] = ''
            self.save(item)

    def save(self, item):
        self.cursor.execute(self.sql,
            [item['title'], item['author'], item['publish_time'],item['content'], item['url'],item['key_word']])
        self.cnn.commit()

    def check_url(self, url):
        # 查看数据库中是否存在当前爬取的url，如果存在则放弃爬取
        if url in self.urls:
            print(f'{url} 已存在')
            return False
        else:
            self.urls.append(url)
            return True

    def get_url(self):
        baseurl = 'https://money.163.com/special/00252G50/'
        for i in range(2, 10):
            url = baseurl + 'macro_0{}.html'.format(i)
            # print(url)
            response = requests.get(url, headers=headers).text
            # print(response)
            html = etree.HTML(response)
            urls = html.xpath("//div[@class='item_top']/h2/a/@href")
            titles = html.xpath("//div[@class='item_top']/h2/a/text()")
            
            for u, t in zip(urls, titles):
                # print(u, t)
                self.get_info(u, t)

# 爬虫运行程序
def run():
    spider = Spider()
    spider.get_url()

if __name__ == '__main__':
    run()
