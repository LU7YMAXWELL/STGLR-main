import torch
import torch.nn as nn
import faiss
from modules import Encoder, LayerNorm, CrossAttention

# 组原型注意力网络
class GroupPrototypeAttention(nn.Module):
    def __init__(self, args):
        super(GroupPrototypeAttention, self).__init__()
        self.args = args
        self.num_groups = args.num_group_prototypes  # N_g（组原型数量）
        self.hidden_size = args.hidden_size
        self.max_seq_len = args.max_seq_length

        # 1. Group Interest Pooling（论文公式10）：可学习池化矩阵 W_A^P (Ng × T)
        self.pooling_matrix = nn.Parameter(torch.Tensor(self.num_groups, self.max_seq_len))
        nn.init.xavier_uniform_(self.pooling_matrix)  # 初始化（与原代码一致）

        # 池化后过MLP得到组相关性 C^A（Ng × 1）
        self.pool_mlp = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
        LayerNorm(self.hidden_size, eps=1e-12),nn.ReLU(),nn.Linear(self.hidden_size, 1)  # 输出每个组的相关性分数
        )

        # 2. Group Interest Aggregation（论文公式11）：交叉注意力 + MLP
        self.cross_attn = CrossAttention(args)
        self.agg_mlp = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),LayerNorm(self.hidden_size, eps=1e-12),nn.ReLU())

        # 可学习组原型 G（Ng × D）（论文3.2.3）
        self.group_prototypes = nn.Parameter(torch.Tensor(self.num_groups, self.hidden_size))
        nn.init.xavier_uniform_(self.group_prototypes)  # 初始化组原型

    # 组原型注意力机制前向传播
    def forward(self, sequence_encoding, attention_mask):
        # sequence_encoding: [B, T, D]（SASRec输出的S^A）
        # attention_mask: [B, T]（原始掩码，1=有效，0=padding）
        batch_size = sequence_encoding.shape[0]

        # -------------------------- 1. 组兴趣池化（计算C^A） --------------------------
        # 扩展池化矩阵到batch维度：[Ng, T] → [B, Ng, T]
        pooling_matrix_expand = self.pooling_matrix.unsqueeze(0).expand(batch_size, -1, -1) # W_A^P
        # 屏蔽padding位置（池化权重置0）
        mask =  attention_mask.squeeze().float()  # [B, T, T]  T=50
        #mask = None
        pooling_matrix_masked = pooling_matrix_expand #* mask  # [B, Ng, T]

        # 池化计算：W_A^P × S^A → [B, Ng, D]
        pooled = torch.matmul(pooling_matrix_masked, sequence_encoding)  # [B, Ng, D]  #  B  10  64
        # MLP输出相关性 C^A（[B, Ng, 1]）
        C = self.pool_mlp(pooled)   #论文公式10   [B, Ng, 1]

        # -------------------------- 2. 组兴趣聚合（计算G^A_u） --------------------------
        # 扩展组原型到batch维度：[Ng, D] → [B, Ng, D]
        G = self.group_prototypes.unsqueeze(0).expand(batch_size, -1, -1)
        # 交叉注意力：CA(G, S^A)（论文公式11）
        G_CA = self.cross_attn(Q=G, K=sequence_encoding, V=sequence_encoding, attention_mask=mask)
        # MLP聚合得到 G^A
        G_A = self.agg_mlp(G_CA)  # [B, Ng, D]

        # 用softmax(C^A)加权得到用户的组表示 G^A_u（论文公式12）
        C_softmax = nn.Softmax(dim=1)(C)  # [B, Ng, 1]（组维度归一化）\
        C_softmax = torch.mean(C_softmax,dim=1).unsqueeze(1)
        G_A_u = G_A * C_softmax  # [B, Ng, D]（加权组表示）

        return G_A_u, C  # 返回用户组表示、组相关性

    # 3. 组原型解纠缠损失（论文公式13）：L^g = -λ_g * sum((G_i - G_j)^2)
    def calculate_disentanglement_loss(self):
        if self.num_groups <= 1:
            return 0.0  # 仅1个组无需解纠缠
        G = self.group_prototypes  # [Ng, D]
        disent_loss = 0.0
        # 遍历所有i<j的组对，计算平方差之和
        counter = 0
        for i in range(self.num_groups):
            for j in range(i+1, self.num_groups):
                disent_loss += torch.norm(G[i] - G[j], p=2) ** 2/(G[i].size()[0])
                counter +=1
        return 1 -self.args.lambda_g * disent_loss/counter

class KMeans(object):
    def __init__(self, num_cluster, seed, hidden_size, gpu_id=0, device="cpu"):

        self.seed = seed
        self.num_cluster = num_cluster
        self.max_points_per_centroid = 4096
        self.min_points_per_centroid = 0
        self.gpu_id = 0
        self.device = device
        self.first_batch = True
        self.hidden_size = hidden_size
        self.clus, self.index = self.__init_cluster(self.hidden_size)
        self.centroids = []

    def __init_cluster(self, hidden_size, verbose=False, niter=20, nredo=5, max_points_per_centroid=4096, min_points_per_centroid=0):
        print(" cluster train iterations:", niter)
        clus = faiss.Clustering(hidden_size, self.num_cluster)
        clus.verbose = verbose
        clus.niter = niter
        clus.nredo = nredo
        clus.seed = self.seed
        clus.max_points_per_centroid = max_points_per_centroid
        clus.min_points_per_centroid = min_points_per_centroid

        res = faiss.StandardGpuResources()
        res.noTempMemory()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = self.gpu_id
        index = faiss.GpuIndexFlatL2(res, hidden_size, cfg)
        return clus, index

    #聚类操作
    def train(self, x):
        #如果输入样本数大于聚类中心数，则进行训练
        if x.shape[0] > self.num_cluster:
            #使用FAISS库训练KMeans聚类对象
            self.clus.train(x, self.index)
        #获取聚类中心
        centroids = faiss.vector_to_array(self.clus.centroids).reshape(self.num_cluster, self.hidden_size)
        #转化为pytorch格式
        centroids = torch.Tensor(centroids).to(self.device)
        #归一化
        self.centroids = nn.functional.normalize(centroids, p=2, dim=1)

    # 查询操作（x为输入的嵌入）（使用FAISS库进行最近邻搜索）
    def query(self, x):
        #D为每个样本到最近聚类中心的距离，I为每个样本对应的最近聚类中心索引
        D, I = self.index.search(x, 1)
        #将聚类索引转化为整形列表
        seq2cluster = [int(n[0]) for n in I]
        #进一步转化为pytorch格式
        seq2cluster = torch.LongTensor(seq2cluster).to(self.device)
        #返回每个样本所属的聚类索引极其对应的聚类中心
        return seq2cluster, self.centroids[seq2cluster]

class SASRecModel(nn.Module):
    def __init__(self, args):
        super(SASRecModel, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        # self.group_to_item = nn.Linear(args.hidden_size, args.hidden_size).to("cuda")
        # self.group_weight = nn.Linear(args.hidden_size, 1).to("cuda")


        # 组原型注意力模块
        self.group_proto_attn = GroupPrototypeAttention(args)
        self.args = args

        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

    #插入位置向量
    def add_position_embedding(self, sequence):
        #获取当前序列的长度
        seq_length = sequence.size(1)
        #按照序列长度创建位置索引
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        #扩展位置索引到与输入序列相同的维度
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        #将序列转化为嵌入
        item_embeddings = self.item_embeddings(sequence)
        #将位置向量转化为嵌入
        position_embeddings = self.position_embeddings(position_ids)
        #两个嵌入相加得到最终嵌入
        sequence_emb = item_embeddings + position_embeddings
        #归一化
        sequence_emb = self.LayerNorm(sequence_emb)
        #正则化
        sequence_emb = self.dropout(sequence_emb)
        return sequence_emb

    def forward(self, input_ids):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        #掩码
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()
        subsequent_mask = subsequent_mask.cuda()
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        sequence_emb = self.add_position_embedding(input_ids)
        item_encoded_layers = self.item_encoder (sequence_emb, extended_attention_mask, output_all_encoded_layers=True) # 每层的输出
        sequence_output = item_encoded_layers[-1] #取最后一层的输出

        #  组原型注意力计算
        G_A_u, C = self.group_proto_attn(sequence_emb, extended_attention_mask)
        return sequence_output +G_A_u
        #return sequence_output # 256（batch）  50 序列长度（左padding 达到的固定长度）   64 编码维度

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()