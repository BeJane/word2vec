# @Author: https://github.com/bejane
# @Date: 2021/10/2
import numpy as np

import matplotlib.pyplot as plt
def plt_word(tokens,visualizeWords,embeddings,plt_file='word_vectors.png'):

    visualizeIdx = [tokens[word] for word in visualizeWords]
    visualizeVecs = [embeddings[id] for id in visualizeIdx]
    temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
    covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
    U, S, V = np.linalg.svd(covariance)
    coord = temp.dot(U[:, 0:2])

    for i in range(len(visualizeWords)):
        plt.text(coord[i, 0], coord[i, 1], visualizeWords[i],
                 bbox=dict(facecolor='green', alpha=0.1))
    # 支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # 支持坐标轴负号
    plt.rcParams['axes.unicode_minus']=False

    plt.xlim((np.min(coord[:, 0]), np.max(coord[:, 0])))
    plt.ylim((np.min(coord[:, 1]), np.max(coord[:, 1])))

    plt.savefig(plt_file)

def load_embeeding(path):
    embeddings = []
    tokens = dict()
    words = []
    with open(path, 'r', encoding='utf-8') as f:
        i = 0
        f.readline()
        for line in f:
            line = line.split(" ")
            words.append(line[0])
            tokens[line[0]] = i
            i += 1
            embeddings.append([float(e) for e in line[1:]])
    return words,tokens,embeddings
if __name__ == '__main__':
    path = 'renminribao\embeddings_9epoch_81.413loss.txt'
    words, tokens, embeddings = load_embeeding(path)
    visualizeWords = ['新冠','肺炎','健康','疫情','人民','群众','卫生',
                      '发展','绿色','武汉','南京','厦门','女','男','先生','女士',
                      '数字','产业','经济','医院','老人','孩子']
    plt_file = 'renminribao\word_vector.png'
    plt_word(visualizeWords=visualizeWords, tokens=tokens, embeddings=embeddings, plt_file=plt_file)