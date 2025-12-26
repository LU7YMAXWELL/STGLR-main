import os
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from datasets import RecWithContrastiveLearningDataset
from trainers import STGLR
from models import SASRecModel
from utils import EarlyStopping, get_user_seqs, check_path, set_seed
from datasets import DS


def setParser():
    parser = argparse.ArgumentParser()
    # 系统参数设置
    parser.add_argument("--data_dir", default="../data/", type=str, help="数据读取路径")
    parser.add_argument("--output_dir", default="output", type=str, help="训练保存路径")
    parser.add_argument("--data_name", default="Beauty", type=str, help="数据集选择")
    parser.add_argument("--do_eval", action="store_true", help="训练和测试选择")
    # 对比学习任务设置
    parser.add_argument("--intent_num", default=512, type=int, help="意图数量")
    parser.add_argument("--sim", default='dot', type=str, help="选择点积或者余弦相似度计算")
    # 模型设置
    parser.add_argument("--hidden_size", type=int, default=64, help="Transformer隐藏层维度大小")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="Transformer堆叠层数")
    parser.add_argument("--num_attention_heads", default=2, type=int, help="多头注意力机制的头数")
    parser.add_argument("--hidden_act", default="gelu", type=str, help="隐藏层激活函数")
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="注意力概率的Dropout率")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="隐藏层的Dropout率")
    parser.add_argument("--initializer_range", type=float, default=0.02, help="参数初始化范围")
    parser.add_argument("--max_seq_length", default=50, type=int, help="最大序列长度")
    parser.add_argument("--noise_ratio",default=0.0,type=float,help="噪声设置（鲁棒性实验）",)
    # 训练设置
    parser.add_argument("--lr", type=float, default=0.001, help="Adam优化器的学习率")
    parser.add_argument("--batch_size", type=int, default=256, help="训练批次大小")
    parser.add_argument("--epochs", type=int, default=300, help="训练总轮数")
    parser.add_argument("--seed", default=2022, type=int, help="随机种子")
    # 损失权重设置
    parser.add_argument("--rec_weight", type=float, default=1, help="推荐任务损失权重")
    parser.add_argument("--lambda_0", type=float, default=0.1, help="粗粒度权重")
    parser.add_argument("--beta_0", type=float, default=0.1, help="细粒度权重")
    # 优化器权重衰减
    parser.add_argument("--weight_decay", type=float, default=0.0, help="权重衰减系数")

    # 新增
    parser.add_argument("--num_group_prototypes", default=5, type=int,
                        help="组原型数量(论文中的Ng，搜索范围[1,5,10,20])")
    parser.add_argument("--lambda_g", default=0.01, type=float,
                        help="组解纠缠正则化系数(论文中的λg)")

    return parser.parse_args()

def main():
    args=setParser()
    # 设置随机种子
    set_seed(args.seed)
    # 设置输出路径
    check_path(args.output_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 设置训练数据的文件名和格式
    args.data_file = args.data_dir + args.data_name + ".txt"
    args.train_data_file = args.data_dir + args.data_name + "_1.txt"
    # 通过DS(·)操作实现自监督信号构建
    if not os.path.exists(args.train_data_file):
        DS(args.data_file, args.train_data_file, args.max_seq_length)
    # 从训练文件中获取数据
    _, train_user_seq, _, _, _ = get_user_seqs(args.train_data_file)
    # 从验证和测试文件中获取数据
    _, user_seq, max_item, valid_rating_matrix, test_rating_matrix = get_user_seqs(args.data_file)
    # 设置物品总数
    args.item_size = max_item + 2
    # 设置掩码ID
    args.mask_id = max_item + 1
    # 设置保存模型的路径
    args_str = args.data_name
    # 设置保存模型日志的路径
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")
    # 展示所有的参数
    for arg in vars(args):
        print(f"{arg:<30} : {getattr(args, arg):>35}")
    with open(args.log_file, "a") as f:
        f.write(str(args) + "\n")
    # 在验证集中过滤掉训练集中出现的物品
    args.train_matrix = valid_rating_matrix
    # 设置保存模型的文件和路径
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)
    # 聚类相关（数据集、顺序采样器、数据加载器）(采取顺序采样)
    cluster_dataset = RecWithContrastiveLearningDataset(args, train_user_seq, data_type="train")
    cluster_sampler = SequentialSampler(cluster_dataset)
    cluster_dataloader = DataLoader(cluster_dataset, sampler=cluster_sampler, batch_size=args.batch_size)
    # 训练集相关（数据集、顺序采样器、数据加载器）(采取乱序采样)
    train_dataset = RecWithContrastiveLearningDataset(args, train_user_seq, data_type="train")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
    # 验证集相关（数据集、顺序采样器、数据加载器）
    eval_dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type="valid")
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)
    # 测试集相关（数据集、顺序采样器、数据加载器）
    test_dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type="test")
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)
    # 编码器和训练器选择设置
    model = SASRecModel(args=args)
    trainer = STGLR(model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args)
    # 评估模式
    if args.do_eval:
        trainer.args.train_matrix = test_rating_matrix
        trainer.load(args.checkpoint_path)
        scores, result_info = trainer.test(0, full_sort=True)
    # 训练模式
    else:
        # 早停设置
        early_stopping = EarlyStopping(args.checkpoint_path, patience=40, verbose=True)
        for epoch in range(args.epochs):
            trainer.args.train_matrix = valid_rating_matrix
            trainer.train(epoch)

            trainer.args.train_matrix = test_rating_matrix #added
            scores, _ = trainer.test(epoch, full_sort=True) # valid
            early_stopping(np.array(scores[-1:]), trainer.model,epoch)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        # 测试模式
        trainer.args.train_matrix = test_rating_matrix
        print("---------------Change to test_rating_matrix!-------------------")
        # 加载最佳模型
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, full_sort=True)
    # 结果处理和日志记录
    with open(args.log_file, "a") as f:
        f.write(args_str + "\n")
        f.write(result_info + "\n")

main()