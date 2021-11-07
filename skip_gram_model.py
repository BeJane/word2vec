import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SkipGramModel(nn.Module):
    def __init__(self,emb_size,emb_dimension,use_cuda=False):
        '''
        Initialize model parameters.

        :param emb_size (int): number of embeddings
        :param emb_dimension: dimension of embedding,typically from 50 to 500
        '''
        super(SkipGramModel,self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.use_cuda = use_cuda

        # initialize embeddings
        # embeddings of center words
        self.v_embeddings = nn.Embedding(emb_size,emb_dimension,sparse=True)
        # embeddings of context words
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        init_range = 0.5 / emb_dimension
        self.v_embeddings.weight.data.uniform_(-init_range,init_range)
        self.u_embeddings.weight.data.uniform_(-0,0)
    
    def forward(self,v,u_pos,u_neg):
        """
        Forward process

        :param v (Tensor): a list of center wordc ids (batch_size)
        :param u_pos (Tensor): a list of outside word ids for positive word pairs  (batch_size)
        :param u_neg (Tensor): a list of outside word ids for negative word pairs  (batch_size, neg_sampling_count)
        :return:
        """

        emb_v = self.v_embeddings(v) # batch_size x emb_dimension
        emb_u_pos = self.u_embeddings(u_pos) # batch_size x emb_dimension
        emb_u_neg = self.u_embeddings(u_neg) # batch_size x neg_sampling_count x emb_dimension

        score = torch.mul(emb_v,emb_u_pos)
        score = score.squeeze()
        score = torch.sum(score,dim=1) # batch_size
        score = F.logsigmoid(score)

        neg_score = torch.bmm(emb_u_neg,emb_v.unsqueeze(2)).squeeze() # batch_size x neg_sampling_count
        neg_score = F.logsigmoid(-1 * neg_score)

        return -1 * (torch.sum(score)+torch.sum(neg_score))
    def save_embeddings(self,tokens,output_file):
        """

        :param token: (dict) map word to word id
        :param output_file: (str)
        :param use_cuda:
        :return:
        """
        embeddings = self.get_embedding()

        f = open(output_file,'w',encoding='utf-8')
        f.write(len(tokens) + ' ' + self.emb_dimension)
        for word,id in tokens.items():
            e = embeddings[id]
            e = " ".join(map(lambda x:str(x),e))
            f.write("\n%s %s"%(word,e))


    def get_embedding(self):
        if self.use_cuda:
            embeddings = self.v_embeddings.weight.cpu().data.numpy()
        else:
            embeddings = self.v_embeddings.weight.data.numpy()

        return embeddings