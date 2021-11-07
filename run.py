# @Author: https://github.com/bejane
# @Date: 2021/9/29
import math
import random
import numpy as np

from data_util import DataSet
from skip_gram_model import SkipGramModel
from plt_word import plt_word
import torch
from torch import optim
import torch.utils.data as Data
from matplotlib import pyplot as plt
from tqdm import tqdm

"""参数"""
debug = False
batchSize = 50
EPOCHS = 100
learning_rate = 0.025
use_cuda = torch.cuda.is_available()


"""数据预处理"""
# Reset the random seed to make sure that everyone gets the same results
random.seed(31415)
np.random.seed(9265)
# We are going to train 10-dimensional vectors for this assignment
dimVectors = 100

windowSize = 5
K = 5
dataset = DataSet(path='data/renminribao.txt',debug=debug)
tokens = dataset._tokens
nWords = len(tokens)
# dataset.draw_word_cloud()
centerWords,outsideWords = dataset.pair(windowSize=windowSize)
batch_count = math.ceil(len(centerWords)/batchSize)
print("="*10+"数据加载完成"+"="*10)
"""加载模型"""
model = SkipGramModel(emb_size=nWords,emb_dimension=dimVectors,use_cuda=use_cuda)

if use_cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # 优化器
# 指数衰减学习率控制器
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 80], 0.9)

"""开始训练"""
print("="*10+"开始训练"+"="*10)
lossList = []
min_loss = 89
last_loss = 0.0
for epoch in range(EPOCHS):  # loop over the dataset multiple times

    running_loss = 0.0
    process_bar = tqdm(range(batch_count))
    #trainSet = Data.TensorDataset(torch.LongTensor(centerWords[:batch_count//2]),
    #                             torch.LongTensor(outsideWords[:batch_count//2]))

    #trainloader = Data.DataLoader(trainSet, batch_size=batchSize, shuffle=True)

    for i in process_bar:
        c = centerWords[i*batchSize:(i+1)*batchSize]
        o = outsideWords[i*batchSize:(i+1)*batchSize]
        neg = dataset.getNegativeSample(centerWords=c,outsideWords=o,K=K)
        c = torch.LongTensor(c)
        o = torch.LongTensor(o)
        neg = torch.LongTensor(neg)
        if use_cuda:
            c = c.cuda()
            o = o.cuda()
            neg = neg.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        loss = model.forward(v=c,u_pos=o,u_neg=neg)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item() # 用于将一个零维张量转换成浮点数

        process_bar.set_description('[epoch: %d, batch: %5d] loss: %.5f lr: %.5f' %
                  (epoch + 1, i + 1, loss.item(),optimizer.param_groups[0]['lr']))

        if i % 20000 == 19999:
            lossList.append(running_loss/(20000))
            running_loss = 0.0

    if epoch % 1 == 0:

        plt.plot(np.arange(len(lossList)),lossList)
        plt.title("renminribao_ExponentialLR(0.99) epoch: "+str(epoch+1))
        plt.xlabel('20000 iteration')
        plt.ylabel('loss')
        plt.show()

    scheduler.step()
    last_loss = sum(lossList) / len(lossList)
    print('last loss: %.5f'%(last_loss))
    if last_loss < min_loss:
        min_loss = last_loss - 1
        """save embeddings"""
        output_file = 'renminribao/embeddings_%depoch_%.3floss.txt'%(epoch+1,last_loss)
        model.save_embeddings(tokens=tokens, output_file=output_file)
print('Finished Training')

"""save embeddings"""
output_file = 'renminribao/embeddings_%.3floss.txt'%(last_loss)
model.save_embeddings(tokens=tokens, output_file=output_file)