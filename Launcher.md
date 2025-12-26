## Evaluate Model
On Beauty:
```
python main.py --data_name Beauty --do_eval
```
On Sports_and_Outdoors:
```
python main.py --data_name Sports_and_Outdoors --do_eval
```
On Toys_and_Games:
```
python main.py --data_name Toys_and_Games --do_eval
```
On ML-1M:
```
python main.py --data_name ml-1m --do_eval
```
## Train Model
On Beauty:
```
python main.py --data_name Beauty
python main.py --data_name Beauty --lambda_0 0.3 --intent_num 256
python main.py --data_name Beauty --lambda_0 0.3 --noise_ratio 5 --intent_num 256
```
On Sports_and_Outdoors:
```
python main.py --data_name Sports_and_Outdoors
python main.py --data_name Sports_and_Outdoors --lambda_0 0.3 --intent_num 256
```
On Toys_and_Games:
```
python main.py --data_name Toys_and_Games
python main.py --data_name Toys_and_Games --lambda_0 0.2 --intent_num 1024
```
On ML-1M:
```
python main.py --data_name ml-1m
```

Ablation on Beauty:
```w/o CICL
python main.py --data_name Beauty --lambda_0 0 --beta_0 0.1 --intent_num 256
```

```w/o FICL
python main.py --data_name Beauty --lambda_0 0.3 --beta_0 0 --intent_num 256
```

```w/o FICL
python main.py --data_name Beauty --lambda_0 0.3 --beta_0 0.1 --num_group_prototypes 1 --intent_num 256
```

Hyperparameter Sensitivity Analysis on Beauty:
```lamba
python main.py --data_name Beauty --lambda_0 0.1 --beta_0 0.1 --intent_num 256
python main.py --data_name Beauty --lambda_0 0.2 --beta_0 0.1 --intent_num 256
python main.py --data_name Beauty --lambda_0 0.3 --beta_0 0.1 --intent_num 256
python main.py --data_name Beauty --lambda_0 0.4 --beta_0 0.1 --intent_num 256
python main.py --data_name Beauty --lambda_0 0.5 --beta_0 0.1 --intent_num 256
```
```beta
python main.py --data_name Beauty --lambda_0 0.3 --beta_0 0.1 --intent_num 256
python main.py --data_name Beauty --lambda_0 0.3 --beta_0 0.2 --intent_num 256
python main.py --data_name Beauty --lambda_0 0.3 --beta_0 0.3 --intent_num 256
python main.py --data_name Beauty --lambda_0 0.3 --beta_0 0.4 --intent_num 256
python main.py --data_name Beauty --lambda_0 0.3 --beta_0 0.5 --intent_num 256
```
```N
python main.py --data_name Beauty --lambda_0 0.3 --beta_0 0.1 --num_group_prototypes 1 --intent_num 256
python main.py --data_name Beauty --lambda_0 0.3 --beta_0 0.1 --num_group_prototypes 5 --intent_num 256
python main.py --data_name Beauty --lambda_0 0.3 --beta_0 0.1 --num_group_prototypes 10 --intent_num 256
python main.py --data_name Beauty --lambda_0 0.3 --beta_0 0.1 --num_group_prototypes 20 --intent_num 256
python main.py --data_name Beauty --lambda_0 0.3 --beta_0 0.1 --num_group_prototypes 30 --intent_num 256
```