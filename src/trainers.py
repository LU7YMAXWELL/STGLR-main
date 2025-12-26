import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils import recall_at_k, ndcg_k
from models import KMeans


class Trainer:
    def __init__(self, model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args):
        # 传入参数设置
        self.args = args
        self.cuda_condition = torch.cuda.is_available()
        self.device = torch.device("cuda")
        self.model = model
        self.batch_size = self.args.batch_size
        self.sim = self.args.sim
        # 初始化意图聚类器
        cluster = KMeans(num_cluster=args.intent_num, seed=1, hidden_size=64, device=torch.device("cuda"), ) # 256
        # 存储聚类中心
        self.clusters = [cluster]
        self.clusters_t = [self.clusters]
        if self.cuda_condition:
            self.model.cuda()
        # 设置训练、聚类、验证、测试数据加载器
        self.train_dataloader = train_dataloader
        self.cluster_dataloader = cluster_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        # 优化器初始化
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        # 计算并打印模型总参数量
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    # 训练
    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader, self.cluster_dataloader)

    # 验证
    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)

    # 测试
    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort=full_sort, train=False)

    # 迭代器（待实现）
    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    # 获取的评估指标
    def get_score(self, epoch, answers, pred_list):
        # 初始化存储列表
        recall, ndcg = [], []
        # 遍历多个K值计算指标
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        # 创建结果字典
        post_fix = {"Epoch": epoch, "HIT@5": "{:.4f}".format(recall[0]), "HIT@10": "{:.4f}".format(recall[1]),
                    "HIT@20": "{:.4f}".format(recall[3]), "NDCG@5": "{:.4f}".format(ndcg[0]),
                    "NDCG@10": "{:.4f}".format(ndcg[1]), "NDCG@20": "{:.4f}".format(ndcg[3]), }
        print(post_fix)
        print("Top5_Sample:" + self.get_topk_sample(pred_list, answers, k=5, num_samples=3))
        # 将结果保存到目录文件
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        # 返回数值结果和字符串结果
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def get_topk_sample(self, pred_list, answer_list, k=5, num_samples=3):
        k = min(k, pred_list.shape[1])

        # 随机选择num_samples个用户
        num_users = len(pred_list)
        sample_indices = np.random.choice(num_users, min(num_samples, num_users), replace=False)

        samples = []
        for idx in sample_indices:
            # 获取当前用户的Top-K预测
            user_pred = pred_list[idx][:k]
            # 获取当前用户的真实答案
            user_answer = answer_list[idx][0] if isinstance(answer_list[idx], np.ndarray) else answer_list[idx]

            # 检查真实答案是否在Top-K中
            hit = user_answer in user_pred

            # 创建用户示例字符串
            sample_str = f"User {idx}: "
            sample_str += f"True Item: {user_answer} | "
            sample_str += f"Top-{k}: {user_pred.tolist()} | "
            sample_str += f"Hit: {'✓' if hit else '✗'}"

            samples.append(sample_str)

        # 返回所有示例的字符串表示
        return "\n".join(samples)

    # 加载模型
    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    # 假负样本掩码
    def mask_correlated_samples(self, label):
        # 将标签重塑为行向量[1,N]
        label = label.view(1, -1)
        # 扩展为两倍行数[2,N]，合并为单行向量[1,2*N]
        label = label.expand((2, label.shape[-1])).reshape(1, -1)
        # 转置为列向量[2*N,1]
        label = label.contiguous().view(-1, 1)
        # 生成掩码矩阵[2*N,2*N]
        mask = torch.eq(label, label.t())
        return mask == 0

    # 对比学习损失函数
    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot', intent_id=None):
        # 向量拼接
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        # 计算余弦相似度或者点积
        if sim == 'cos':
            sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.t()) / temp
        # 获取(z_i,z_j)对角的相似度
        sim_i_j = torch.diag(sim, batch_size)
        # 获取(z_j,z_i)对角的相似度
        sim_j_i = torch.diag(sim, -batch_size)
        # 构建正样本向量
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # 应用假负样本掩码组件
        mask = self.mask_correlated_samples(intent_id)
        # 复制相似度矩阵
        negative_samples = sim
        # 屏蔽假负样本位置
        negative_samples[mask == 0] = float("-inf")
        # 准备标签
        labels = torch.zeros(N).to(positive_samples.device).long()
        # 合并正负样本
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    # 生成完整的物品评分预测
    def predict_full(self, seq_out):
        # 获取所有物品的嵌入向量 
        test_item_emb = self.model.item_embeddings.weight
        # 计算序列标示与所有物品的相似度
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    # 粗粒度对比学习损失
    def cicl_loss(self, coarse_intents, target_item):
        # 获取粗粒度序列表示，即h1和h2
        coarse_intent_1, coarse_intent_2 = coarse_intents[0], coarse_intents[1]
        # 提取序列尾部表示(意图表示)
        view1_repr = coarse_intent_1[:, -1, :]
        view2_repr = coarse_intent_2[:, -1, :]
        # 计算对比损失
        sem_nce_logits, sem_nce_labels = self.info_nce(view1_repr, view2_repr, 1.0,
                                                       view2_repr.shape[0], self.sim, target_item[:, -1])
        # 计算交叉熵损失
        cicl_loss = nn.CrossEntropyLoss()(sem_nce_logits, sem_nce_labels)
        return cicl_loss

    # 细粒度对比学习损失
    def ficl_loss(self, sequences, clusters_t):
        # 计算h1和c1的损失
        output = sequences[0][:, -1, :]
        intent_n = output.view(-1, output.shape[-1])
        intent_n = intent_n.detach().cpu().numpy()
        # 获取序列表示并查询聚类中心，即query(·)操作
        intent_id, seq_to_v = clusters_t[0].query(intent_n)
        seq_to_v = seq_to_v.view(seq_to_v.shape[0], -1)
        # 计算损失
        a, b = self.info_nce(output.view(output.shape[0], -1), seq_to_v, 1.0, output.shape[0], sim=self.sim,
                             intent_id=intent_id)
        loss_n_0 = nn.CrossEntropyLoss()(a, b)
        # 计算h2和c2的损失
        output_s = sequences[1][:, -1, :]
        intent_n = output_s.view(-1, output_s.shape[-1])
        intent_n = intent_n.detach().cpu().numpy()
        # 获取序列表示并查询聚类中心，即query(·)操作
        intent_id, seq_to_v_1 = clusters_t[0].query(intent_n)
        seq_to_v_1 = seq_to_v_1.view(seq_to_v_1.shape[0], -1)
        a, b = self.info_nce(output_s.view(output_s.shape[0], -1), seq_to_v_1, 1.0, output_s.shape[0], sim=self.sim,
                             intent_id=intent_id)
        # 计算损失
        loss_n_1 = nn.CrossEntropyLoss()(a, b)
        # h1和c1 以及 h2和c2 损失相加并输出
        ficl_loss = loss_n_0 + loss_n_1
        return ficl_loss


class STGLR(Trainer):
    def __init__(self, model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args):
        super(STGLR, self).__init__(model, train_dataloader, cluster_dataloader, eval_dataloader,
                                    test_dataloader, args)

    def iteration(self, epoch, dataloader, cluster_dataloader=None, full_sort=True, train=True):
        # 训练逻辑
        if train:
            # 聚类中心评估模式(模型评估模式)
            self.model.eval()
            # 初始化列表用于存储所有序列的表示向量(用于聚类)
            kmeans_training_data = []
            # 创建带进度条的聚类数据迭代器(可视化处理进度)(正序)
            rec_cf_data_iter = tqdm(enumerate(cluster_dataloader), total=len(cluster_dataloader))
            # 遍历聚类数据集中的每个批次
            for i, (rec_batch) in rec_cf_data_iter:
                # 将批次数据移动到指定设备
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                # 解包批次数据(仅使用序列输入，忽略其它信息)
                _, subsequence, _, _, _ = rec_batch
                # 向前传播获取序列所有位置的表示
                sequence_output_a  = self.model(subsequence)
                # 提取序列最后一个位置的表示(假设包含最完整的信息)
                sequence_output_b = sequence_output_a[:, -1, :] # 取每个样本最后一个token的编码向量，他聚集了整个序列的整体信息编码
                # 收集去梯度的表示数据(避免影响梯度计算)
                kmeans_training_data.append(sequence_output_b.detach().cpu().numpy())
            # 合并所有批次的表示向量
            kmeans_training_data = np.concatenate(kmeans_training_data, axis=0)
            # 将其包装为列表
            kmeans_training_data_t = [kmeans_training_data]
            # 遍历所有的聚类器
            for i, clusters in tqdm(enumerate(self.clusters_t), total=len(self.clusters_t)):
                # 遍历当前层级的聚类器
                for j, cluster in enumerate(clusters):
                    # 使用新数据来训练聚类器
                    cluster.train(kmeans_training_data_t[i])
                    # 更新聚类器的引用
                    self.clusters_t[i][j] = cluster # 得到聚类结果  共 intent_num(256)个簇，每个簇用 64维 的向量表示
            # 显式删除大数据对象释放内存
            del kmeans_training_data
            del kmeans_training_data_t
            # 强制垃圾回收
            import gc
            gc.collect()
            print("Performing Rec model Training:")
            # 正式开始训练主模型
            self.model.train()
            # 初始化损失值(推荐主任务损失、联合损失、对比学习损失)
            rec_avg_loss = 0.0
            joint_avg_loss = 0.0
            icl_losses = 0.0
            disent_avg_loss = 0.0  # added：解纠缠损失累计

            # 显示训练数据集大小(调试信息)
            print(f"rec dataset length: {len(dataloader)}")
            # 创建带进度条的主训练数据迭代器(乱序)
            rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))
            # 遍历主训练数据集中的每一个批次
            for i, (rec_batch) in rec_cf_data_iter: #乱序的dataloader  而  cluter的dataloder 是顺序的
                # 将训练数据移动至计算设备中
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                # 解包批次数据(原始序列，目标位置，增强后的序列)(对应论文的采样S1和S2)
                _, subsequence_1, target_pos_1, subsequence_2, _ = rec_batch
                # 向前传播获取序列表示
                intent_output = self.model(subsequence_1)
                # 用最后一个位置的表示预测整个物品库(与测试高度相关)
                logits = self.predict_full(intent_output[:, -1, :])
                # 计算推荐交叉熵损失(仅预测序列最后一个位置)(下一物品与真实物品的交叉熵)
                rec_loss = nn.CrossEntropyLoss()(logits, target_pos_1[:, -1])
                # 获取两个试图的表示
                coarse_intent_1  = self.model(subsequence_1) #  论文中的 h1
                coarse_intent_2  = self.model(subsequence_2) #  论文中的 h2
                # 计算粗粒度意图对比损失
                cicl_loss = self.cicl_loss([coarse_intent_1, coarse_intent_2], target_pos_1)
                # 计算细粒度聚类对比损失
                ficl_loss = self.ficl_loss([coarse_intent_1, coarse_intent_2], self.clusters_t[0])
                # 组合两种对比损失为总的对比学习损失
                icl_loss = self.args.lambda_0 * cicl_loss + self.args.beta_0 * ficl_loss

                #added 计算组原型解纠缠损失（论文公式13）
                #disentanglement_loss = self.model.group_proto_attn.calculate_disentanglement_loss()


                # 联合优化  总损失：推荐损失 + 对比损失
                joint_loss = self.args.rec_weight * rec_loss + icl_loss # + disentanglement_loss
                # 梯度清零 -> 反向传播 -> 参数更新
                self.optim.zero_grad()
                joint_loss.backward()
                self.optim.step()
                # 累计损失值(转换为Python浮点数)
                rec_avg_loss += rec_loss.item()
                # 处理对比损失可能是张量或浮点数的情况
                if type(icl_loss) != float:
                    icl_losses += icl_loss.item()
                else:
                    icl_losses += icl_loss
                #added 解纠缠损失累计
               # disent_avg_loss += disentanglement_loss.item()

                joint_avg_loss += joint_loss.item()
            # 构造训练结果字典
            post_fix = {
                "epoch": epoch,
                # 计算平均推荐损失
                "rec_avg_loss": "{:.4f}".format(rec_avg_loss / len(rec_cf_data_iter)),
                # 计算平均对比学习损失
                "icl_avg_loss": "{:.4f}".format(icl_losses / len(rec_cf_data_iter)),

             #  "disent_avg_loss": "{:.4f}".format(disent_avg_loss / len(rec_cf_data_iter)),  # 新增
                # 计算平均联合损失
                "joint_avg_loss": "{:.4f}".format(joint_avg_loss / len(rec_cf_data_iter)),
            }
            # 打印本轮训练结果
            print(str(post_fix))
            # 将结果追加到日志中
            with open(self.args.log_file, "a") as f:
                f.write(str(post_fix) + "\n")
        # 测试逻辑
        else:
            # 创建带进度条的评估数据迭代器
            rec_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))
            # 设置模型为评估模式
            self.model.eval()
            # 初始化结果存储(预测的Top-k物品列表，真实答案物品)
            pred_list = None
            answer_list = None
            # 遍历评估数据集中的每个批次
            for i, batch in rec_data_iter:
                # 移动数据到指定设备
                batch = tuple(t.to(self.device) for t in batch)
                # 解包批次数据(用户ID，输入序列，目标位置，真实答案)
                user_ids, input_ids, target_pos, answers = batch
                # 向前传播获取序列表示
                recommend_output = self.model(input_ids)
                # 获取最后一个位置的表示(用户意图表示)
                recommend_output = recommend_output[:, -1, :]
                # 计算出用户对所有物品的偏好得分
                rating_pred = self.predict_full(recommend_output)
                # 拷贝数据到CPU中并转化为numpy数组
                rating_pred = rating_pred.cpu().data.numpy().copy()
                # 屏蔽已经交互过的物品
                batch_user_index = user_ids.cpu().numpy()
                # 从训练矩阵中获取已交互物品，并且屏蔽掉这些物品(设置为0分)
                rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                # Top-k物品选择(使用argpartition选择最大的20个物品)
                ind = np.argpartition(rating_pred, -20)[:, -20:]
                # 提取这20个物品的得分
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                # 对20个物品的得分进行降序排序
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                # 获取排序后的物品ID
                batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
                # 累积结果
                if i == 0:
                    # 如果是第一个批次则直接存储
                    pred_list = batch_pred_list
                    answer_list = answers.cpu().data.numpy()
                else:
                    # 如果是后续的批次则追加结果
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
            # 计算评估指标
            return self.get_score(epoch, answer_list, pred_list)
