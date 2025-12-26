import random
import torch
import os
import pickle
from torch.utils.data import Dataset
from utils import get_user_seqs
import copy

# 生成标签字典
class Generate_tag():
    def __init__(self, data_path, data_name, save_path):
        self.path = data_path
        self.data_name = data_name + "_1"
        self.save_path = save_path

    def generate(self):
        data_f = self.path + "/" + self.data_name + ".txt"
        train_dic = {}
        valid_dic = {}
        test_dic = {}
        with open(data_f, "r") as fr:
            data = fr.readlines()
            for d_ in data:
                items = d_.split(' ')
                tag_train = int(items[-3])
                tag_valid = int(items[-2])
                tag_test = int(items[-1])
                train_temp = list(map(int, items[:-3]))
                valid_temp = list(map(int, items[:-2]))
                test_temp = list(map(int, items[:-1]))
                if tag_train not in train_dic:
                    train_dic.setdefault(tag_train, [])
                train_dic[tag_train].append(train_temp)
                if tag_valid not in valid_dic:
                    valid_dic.setdefault(tag_valid, [])
                valid_dic[tag_valid].append(valid_temp)
                if tag_test not in test_dic:
                    test_dic.setdefault(tag_test, [])
                test_dic[tag_test].append(test_temp)

        total_dic = {"train": train_dic, "valid": valid_dic, "test": test_dic}
        print("Saving data to ", self.save_path)
        with open(self.save_path + "/" + self.data_name + "_t.pkl", "wb") as fw:
            pickle.dump(total_dic, fw)

    def load_dict(self, data_path):
        if not data_path:
            raise ValueError('invalid path')
        elif not os.path.exists(data_path):
            print("The dict not exist, generating...")
            self.generate()
        with open(data_path, 'rb') as read_file:
            data_dict = pickle.load(read_file)
        return data_dict

    def get_data(self, data_path, mode):
        data = self.load_dict(data_path)
        return data[mode]

# 采样
class RecWithContrastiveLearningDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type="train"):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

        self.sem_tag = Generate_tag(self.args.data_dir, self.args.data_name, self.args.data_dir)
        self.train_tag = self.sem_tag.get_data(self.args.data_dir + "/" + self.args.data_name + "_1_t.pkl", "train")
        self.true_user_id, _, _, _, _ = get_user_seqs(args.train_data_file)

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer, index):
        copied_input_ids = copy.deepcopy(input_ids)
        pad_len = self.max_len - len(copied_input_ids)
        copied_input_ids = [0] * pad_len + copied_input_ids
        copied_input_ids = copied_input_ids[-self.max_len:]

        if type(target_pos) == tuple:
            pad_len_1 = self.max_len - len(target_pos[1])
            target_pos_1 = [0] * pad_len + target_pos[0]
            target_pos_2 = [0] * pad_len_1 + target_pos[1]
            target_pos_1 = target_pos_1[-self.max_len:]
            target_pos_2 = target_pos_2[-self.max_len:]
            assert len(target_pos_1) == self.max_len
            assert len(target_pos_2) == self.max_len
        else:
            target_pos = [0] * pad_len + target_pos
            target_pos = target_pos[-self.max_len:]
            assert len(target_pos) == self.max_len

        assert len(copied_input_ids) == self.max_len
        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]
            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),
                torch.tensor(copied_input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            if type(target_pos) == tuple:
                cur_rec_tensors = (
                    torch.tensor(user_id, dtype=torch.long),
                    torch.tensor(copied_input_ids, dtype=torch.long),
                    torch.tensor(target_pos_1, dtype=torch.long),
                    torch.tensor(target_pos_2, dtype=torch.long),
                    torch.tensor(answer, dtype=torch.long),
                )

            else:
                cur_rec_tensors = (
                    torch.tensor(user_id, dtype=torch.long),
                    torch.tensor(copied_input_ids, dtype=torch.long),
                    torch.tensor(target_pos, dtype=torch.long),
                    torch.tensor(answer, dtype=torch.long),
                )
        return cur_rec_tensors

    #鲁棒性实验
    def _add_noise_interactions(self, items):
        copied_sequence = copy.deepcopy(items)
        insert_nums = max(int(self.args.noise_ratio * len(copied_sequence)), 0)
        if insert_nums == 0:
            return copied_sequence
        insert_idx = random.choices([i for i in range(len(copied_sequence))], k=insert_nums)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                item_id = random.randint(1, self.args.item_size - 2)
                while item_id in copied_sequence:
                    item_id = random.randint(1, self.args.item_size - 2)
                inserted_sequence += [item_id]
            inserted_sequence += [item]
        return inserted_sequence

    def __getitem__(self, index):
        user_id = index
        items = self.user_seq[index]
        assert self.data_type in {"train", "valid", "test"}
        if self.data_type == "train":
            input_ids = items[:-3] #  -2 为标签   -1 用来评估用的   所以从-3之前的为input  [1153, 1155, 2207, 3712, 3481]
            target_pos = items[1:-2] #错位预测  [1155, 2207, 3712, 3481, 2195]
            # 获取相同标签的所有序列
            temp = self.train_tag[items[-3]]
            flag = False
            for t_ in temp:
                # 排除自身序列
                if t_[1:] == items[:-3]:
                    continue
                else:
                    # 选择不同的序列作为对比样本
                    target_pos_ = t_[1:]
                    flag = True
            if not flag:
                # 随机选择一个序列作为对比样本
                target_pos_ = random.choice(temp)[1:]
            answer = [0]
        elif self.data_type == "valid":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]
        else:
            #items_with_noise = self._add_noise_interactions(items)
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]
        if self.data_type == "train":
            target_pos = (target_pos, target_pos_) # 样本对
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer, index)
            return cur_rec_tensors
        elif self.data_type == "valid":
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer, index)
            return cur_rec_tensors
        else:
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer, index)
            return cur_rec_tensors

    def __len__(self):
        return len(self.user_seq)

def DS(i_file, o_file, max_len):
    # 读取原始数据
    with open(i_file, "r+") as fr:
        data = fr.readlines()
    # 设置存放切片后的序列集合
    aug_d = {}
    # 设置序列的最大长度
    max_save_len = max_len + 3
    # 设置窗口的最大长度
    max_keep_len = max_len + 2
    # 处理每行的数据
    for d_ in data:
        # 按照用户ID和物品序列进行切割
        u_i, item = d_.split(' ', 1)
        # 将物品序列拆分成一个个物品列表
        item = item.split(' ')
        # 将每个序列的最后一个物品作为动态滑动窗口的目标项
        item[-1] = str(eval(item[-1]))
        # 初始化切片后的序列集合
        aug_d.setdefault(u_i, [])
        # 滑动窗口的起始位置
        start = 0
        # 初始子序列的结束位置
        j = 3
        # 如果序列过长则需要分段处理
        if len(item) > max_save_len:
            # 滑动窗口算法
            while start < len(item) - max_keep_len:
                j = start + 4
                while j < len(item):
                    # 针对子序列长度不足的情况下生成递增序列
                    if start < 1 and j - start < max_save_len:
                        # 记录当前子序列
                        aug_d[u_i].append(item[start:j])
                        j += 1
                    # 其它情况下则生成固定长度的子序列
                    else:
                        # 记录当前子序列
                        aug_d[u_i].append(item[start:start + max_save_len])
                        break
                # 更新滑动窗口的起始位置
                start += 1
        # 如果序列较短则直接处理
        else:
            while j < len(item):
                # 记录当前子序列
                aug_d[u_i].append(item[start:j + 1])
                j += 1
    all_sequences = []
    for u_i in aug_d:
        for seq in aug_d[u_i]:
            all_sequences.append((u_i, seq))

    # 随机打乱所有序列的顺序
    random.shuffle(all_sequences)
    # 写入处理完后的数据
    with open(o_file, "w+") as fw:
        # 修改后的写入方式：按打乱后的顺序写入
        for u_i, seq in all_sequences:
            fw.write(u_i + " " + ' '.join(seq) + "\n")
    # 写入处理完后的数据
    # with open(o_file, "w+") as fw:
    #     for u_i in aug_d:
    #         for i_ in aug_d[u_i]:
    #             fw.write(u_i + " " + ' '.join(i_) + "\n")

if __name__ == "__main__":
    DS("../data/Beauty.txt", "../data/Beauty_1.txt", 10)
    g = Generate_tag("../data", "Beauty", "../data")
    data = g.get_data("../data/Beauty_1_t.pkl", "train")
    i = 0
    for d_ in data:
        if len(data[d_]) < 2:
            i += 1
            print("less is : ", data[d_], d_)
    print(i)