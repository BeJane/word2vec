#!/usr/bin/env python
# -*- coding: utf-8 -*-
import jieba
import numpy as np
import random
from wordcloud import WordCloud
import time

class DataSet:
    def __init__(self, path=None, tablesize = 1000000,debug = False,min_count=5):
        if not path:
            path = "data/git_zhihu.txt"

        self.path = path
        self.tablesize = tablesize
        self.debug = debug
        self.min_count = min_count
        self._tokens = self.tokens()
        self.sampleTable()

    def tokens(self):
        print("=" * 10 + "生成tokens字典" + "=" * 10)
        if hasattr(self, "_tokens") and self._tokens:
            return self._tokens

        tokens = dict()
        tokenfreq = dict()
        wordcount = 0
        revtokens = []
        idx = 0

        for sentence in self.sentences():
            for w in sentence:
                wordcount += 1
                if not w in tokens:
                    tokens[w] = idx
                    revtokens += [w]
                    tokenfreq[w] = 1
                    idx += 1
                else:
                    tokenfreq[w] += 1

        tokens["UNK"] = idx
        revtokens += ["UNK"]
        tokenfreq["UNK"] = 1
        wordcount += 1

        self._tokens = tokens
        self._tokenfreq = tokenfreq
        self._wordcount = wordcount
        self._revtokens = revtokens

        print('%d tokens' % (len(self._tokens)))
        return self._tokens

    def sentences(self):
        print("=" * 10 + "读取句子" + "=" * 10)
        if hasattr(self, "_sentences") and self._sentences:
            return self._sentences

        sentences = []
        with open(self.path, "r",encoding="utf-8") as f:
            for line in f:
                splitted = line.strip().split()[1:]
                # Deal with some peculiar encoding issues with this file
                sentences += [[w.lower() for w in splitted]]
        if self.debug:
            sentences = sentences[:50000]
        self._sentences = sentences
        self._sentlengths = np.array([len(s) for s in sentences])
        self._cumsentlen = np.cumsum(self._sentlengths)

        print('%d sentences, %.3f billion words'%(len(self._sentences),
                                                  np.sum(self._sentlengths)/1000000000))

        return self._sentences
    def pair(self,windowSize=5):
        print("=" * 10 + "生成v,u对" + "=" * 10)
        centerWords = []
        outsideWords = []
        sentences = []
        # 去掉低频词
        for s in self._sentences:
            new_sentence = []
            for w in s:
                if self._tokenfreq[w] >= self.min_count:
                    new_sentence.append(w)
            if len(new_sentence) >1:
                sentences.append(new_sentence)

        for sentence in sentences:
            l = len(sentence)
            for i in range(l):
                context = sentence[max(0,i-windowSize):i] + sentence[min(l,i+1):min(l,i+windowSize+1)]

                centerWords += [self._tokens[sentence[i]]] * len(context)
                for w in context:
                    outsideWords.append(self._tokens[w])
        print('%d pairs'%(len(centerWords)))
        return centerWords,outsideWords

    def sampleTable(self):
        """
        负取样概率分布表
        :return:
        """
        if hasattr(self,"_sampleTable") and self._sampleTable:
            return self._sampleTable
        print("=" * 10 + "生成负取样概率分布表" + "=" * 10)
        self._sampleTable = [0] * self.tablesize
        freq = np.array(list(self._tokenfreq.values()) )** 0.75

        """
        for w in self._revtokens:
            if w in self._tokenfreq:
                freq.append(self._tokenfreq[w] ** 0.75)
            else:
                freq.append(0.0)
        """
        freq /= np.sum(freq)
        freq = np.cumsum(freq) * self.tablesize

        j = 0
        for i in range(self.tablesize):
            while i > freq[j]:
                j = j + 1
            self._sampleTable[i] = j
        return self._sampleTable
    def sampleTokenId(self):
        return self._sampleTable[random.randint(0,self.tablesize-1)]
    def getNegativeSample(self,centerWords,outsideWords,K):

        negSample = []
        for c,o in zip(centerWords,outsideWords):
            sample = np.zeros((K,))
            for k in range(K):
                sample[k] = self.sampleTokenId()
                while sample[k] == c or sample[k] == o:
                    sample[k] = self.sampleTokenId()

            negSample.append(sample)
        """np.random.choice(self._sampleTable,size=(len(centerWords),K)).tolist()"""
        return negSample

    def draw_word_cloud(self):
        print("=" * 10 + "生成词云" + "=" * 10)
        text = open(self.path,'r',encoding='utf-8').read()
        # 读取背景图片
        #color_mask = imread('background.jpg')
        cloud = WordCloud(
            font_path='C:\Windows\Fonts\STXINGKA.TTF',  # 若有中文需要设置才会显示中文
            background_color = 'White',
            max_words = 120,
            max_font_size = 150,
            width= 1920,
            height= 1080
        )
        # 产生词云
        word_cloud = cloud.generate(text)
        # 保存词云
        word_cloud.to_file(self.path.replace('txt','jpg'))

def cut_words(paragraph):
    end_sign = ['。', '！', '？']
    # 去掉引号
    #paragraph = paragraph.replace('“', '')
    #paragraph = paragraph.replace('”', '')
    # 分词
    paragraph = paragraph.replace(" ", "")
    paragraph = " ".join(jieba.cut(paragraph))
    # 分句
    for i in range(len(paragraph)-3):
        if paragraph[i] in end_sign:
            if i+3 < len(paragraph) and paragraph[i+2] == '”':
                paragraph = paragraph[:i + 3] + "\n" + paragraph[i + 4:]
            else:
                paragraph = paragraph[:i+1]+"\n"+paragraph[i+2:]
    return paragraph




if __name__ == '__main__':

    windowSize = 5
    K = 5
    dataset = DataSet(path='data/renminribao.txt', debug=False)
    tokens = dataset._tokens
    nWords = len(tokens)
    print('%d tokens' % (nWords))
    dataset.draw_word_cloud()
    """
    output = open('data/hongloumeng.txt','w',encoding='utf-8')
    with open('data/红楼梦.txt',encoding='utf-8') as f:
        for line in f:
            if len(line) > 1:
                p = cut_words(line)
                output.write(p)
                print(p)
    """