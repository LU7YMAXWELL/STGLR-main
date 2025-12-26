# STGLR

This repository provides the official implementation of the STGLR model proposed in the paper 
"STGLR: Fusing Short-Term and Group-Level Long-Term Sequential Recommendation".

## Requirements

* Python >= 3.7

Install dependencies:
```
pip install -r requirements.txt
```

## Dataset

We provide three preprocessed datasets, Beauty, Sports_and_Outdoors, Toys_and_Games.

They are originally from [here](http://jmcauley.ucsd.edu/data/amazon/index.html).

You can download from the official website and place under ./data/.

Run preprocessing:

```
python datasets.py --data_name Beauty
```

## Training

```
python main.py --data_name Beauty
```

You can also use Launcher.md to run all the experiments.

## Evaluation

```
python main.py --data_name Beauty --do_eval
```

You can also use Launcher.md to run all the experiments.

## Citation

```
@inproceedings{Qin2024ICSRec,
  author    = {Xiuyuan Qin and Huanhuan Yuan and Pengpeng Zhao and Guanfeng Liu and Fuzhen Zhuang and Victor S. Sheng},
  title     = {Intent Contrastive Learning with Cross Subsequences for Sequential Recommendation},
  booktitle = {Proceedings of the ACM International Conference on Web Search and Data Mining (WSDM)},
  year      = {2024},
  pages     = {548--556},
  doi       = {10.1145/3616855.3635773}
}
```
and
```
@inproceedings{Lin2024MAN,
  author    = {Guanyu Lin and Chen Gao and Yu Zheng and Jianxin Chang and Yanan Niu and Yang Song and Kun Gai and Zhiheng Li and Depeng Jin and Yong Li and Meng Wang},
  title     = {Mixed Attention Network for Cross-domain Sequential Recommendation},
  booktitle = {Proceedings of the ACM International Conference on Web Search and Data Mining (WSDM)},
  year      = {2024},
  pages     = {405--413},
  doi       = {10.1145/3616855.3635801}
}
```