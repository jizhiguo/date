#%%
import argparse
import logging
import os
import pickle as pkl
import sys
from datetime import datetime
from unittest import result
from tqdm import tqdm
import torch
from simpletransformers.language_modeling import LanguageModelingModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description="Outlier Detection")
parser.add_argument('--subset', type=int, default=1)
parser.add_argument('--disc_hid_layers', type=int, default=4)
parser.add_argument('--disc_hid_size', type=int, default=256)
parser.add_argument('--gen_hid_layers', type=int, default=1)
parser.add_argument('--gen_hid_size', type=int, default=16)
parser.add_argument('--lowercase', type=int, default=1)
parser.add_argument('--extract_reps', type=int, default=0)
parser.add_argument('--sched_params_opt', type=str, default="plateau")
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--disc_drop', type=float, default=0.5)
parser.add_argument('--eval_anomaly', type=int, default=500)

parser.add_argument('--masks', type=int, default=50)
parser.add_argument('--mask_prob', type=int, default=50)

parser.add_argument('--warmup', type=int, default=1000)
parser.add_argument('--exp_prefix', type=str, default="tests")
parser.add_argument('--tensorboard_dir', type=str, default="runs")

parser.add_argument('--optimizer', type=str, default="AdamW")
parser.add_argument('--min_lr', type=float, default=1e-4)
parser.add_argument('--max_lr', type=float, default=1e-5)

parser.add_argument('--random_generator', type=int, default=1)

parser.add_argument('--anomaly_batch_size', type=int, default=16)
parser.add_argument('--train_batch_size', type=int, default=16)
parser.add_argument('--eval_batch_size', type=int, default=16)
parser.add_argument('--rtd_loss_weight', type=int, default=50)
parser.add_argument('--rmd_loss_weight', type=int, default=100)
parser.add_argument('--mlm_loss_weight', type=int, default=1)

parser.add_argument('--sched_patience', type=int, default=100000)

parser.add_argument('--eval_anomaly_after', type=int, default=0)

parser.add_argument('--train_just_generator', type=int, default=0)
parser.add_argument('--seq_len', type=int, default=128)
parser.add_argument('--preprocessed', type=int, default=1)
parser.add_argument('--replace_tokens', type=int, default=0)
parser.add_argument('--mlm_lr_ratio', type=float, default=1)
parser.add_argument('--contamination', type=float, default=0)
parser.add_argument('--dump_histogram', type=float, default=0)

args = parser.parse_args()
subset = args.subset
disc_hid_layers = args.disc_hid_layers
disc_hid_size = args.disc_hid_size
gen_hid_layers = args.gen_hid_layers
gen_hid_size = args.gen_hid_size
lowercase = args.lowercase
extract_reps = args.extract_reps
masks = args.masks
sched_params_opt = args.sched_params_opt
weight_decay = args.weight_decay
disc_drop = args.disc_drop
eval_anomaly = args.eval_anomaly
mask_prob = args.mask_prob
warmup = args.warmup
max_lr = args.max_lr
exp_prefix = args.exp_prefix
random_generator = args.random_generator
anomaly_batch_size = args.anomaly_batch_size
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
optimizer = args.optimizer
min_lr = args.min_lr
rtd_loss_weight = args.rtd_loss_weight
rmd_loss_weight = args.rmd_loss_weight
mlm_loss_weight = args.mlm_loss_weight
sched_patience = args.sched_patience
eval_anomaly_after = args.eval_anomaly_after
train_just_generator = args.train_just_generator
seq_len = args.seq_len
preprocessed = args.preprocessed
replace_tokens = args.replace_tokens
tensorboard_dir = args.tensorboard_dir
mlm_lr_ratio = args.mlm_lr_ratio
contamination = args.contamination
dump_histogram = args.dump_histogram

masks_ = pkl.load(open('./pseudo_labels128_p50.pkl', 'rb'))

print('Using subset ', subset)

if subset == 1:
    subset = ['business']
elif subset == 2:
    subset = ['sci']
elif subset == 3:
    subset = ['sports']
elif subset == 4:
    subset = ['world']

def run_exps():
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    now = datetime.now()
    time = now.strftime("%H:%M:%S")
    date_time = now.strftime("%m%d%Y_%H%M%S")

    if sched_params_opt == "plateau":
        sched_params = {}
        sched_params['sched_name'] = 'plateau'
        sched_params['factor'] = 0.1
        sched_params['patience'] = sched_patience
        sched_params['verbose'] = True
        sched_params['threshold'] = 0.001
        sched_params['min_lr'] = min_lr
    else:
        sched_params = None

    if sched_params is not None:
        run_name = f'{subset[0]}_slen{seq_len}_wd{weight_decay}_lr{max_lr}-{min_lr}_msk{masks}_p{mask_prob}_dl{disc_hid_layers}_sz{disc_hid_size}_gl{gen_hid_layers}_sz{gen_hid_size}_rgen{random_generator}_drop{disc_drop}_w{rtd_loss_weight}_{rmd_loss_weight}_{mlm_loss_weight}_replace{replace_tokens}_mlmr{mlm_lr_ratio}_{date_time}_cont{contamination}'
    else:
        run_name = f'{subset[0]}_slen{seq_len}_wd{weight_decay}_mlr{max_lr}_minlr{min_lr}_msk{masks}_p{mask_prob}_dl{disc_hid_layers}_sz{disc_hid_size}_gl{gen_hid_layers}_sz{gen_hid_size}_rgen{random_generator}_drop{disc_drop}_w{rtd_loss_weight}_{rmd_loss_weight}_{mlm_loss_weight}_replace{replace_tokens}_mlmr{mlm_lr_ratio}_{date_time}_cont{contamination}'

    print(f'RUN: {run_name}')

    train_args = {
        "fp16": False,
        "use_multiprocessing": False,
        "reprocess_input_data": False,
        "overwrite_output_dir": True,
        "num_train_epochs": 20,#was 20
        "save_eval_checkpoints": False,
        "save_model_every_epoch": False,
        "learning_rate": max_lr,
        "warmup_steps": warmup,
        "train_batch_size": train_batch_size,  #was 32
        "eval_batch_size": eval_batch_size,  #was 32
        "gradient_accumulation_steps": 1,
        "block_size": seq_len + 2,
        "max_seq_length": seq_len + 2,
        "dataset_type": "simple",
        "logging_steps": 107400,  #was 500
        "evaluate_during_training": True,
        "evaluate_during_training_steps": 107400,  #was 500
        # "evaluate_during_training_steps_anomaly": eval_anomaly,  #was 500
        "evaluate_during_training_steps_anomaly": 107400,  #was 500
        "anomaly_batch_size": anomaly_batch_size,
        "evaluate_during_training_verbose": True,
        "use_cached_eval_features": True,
        "sliding_window": True,
        "vocab_size": 21128,
        "eval_anomalies": True,
        "random_generator": random_generator,
        "use_rtd_loss": True,
        "rtd_loss_weight": rtd_loss_weight,
        "rmd_loss_weight": rmd_loss_weight,
        "mlm_loss_weight": mlm_loss_weight,
        "dump_histogram": dump_histogram,
        "eval_anomaly_after": eval_anomaly_after,
        "train_just_generator": train_just_generator,
        "replace_tokens": replace_tokens,
        "extract_scores": 1,
        "subset_name": subset[0],
        "extract_repr": 0,
        # "vanilla_electra": {
        #     "no_masks": masks,
        # },
        # "vanilla_electra": False,
        "train_document": True,
        # "tokenizer_name": "bert-base-uncased",
        "tokenizer_name": "hfl/chinese-electra-180g-base-discriminator",
        # "tokenizer_name": "bert-base-cased",
        # "tokenizer_name": "bert-base-chinese",
        # "tokenizer_name": "hfl/chinese-electra-180g-base-generator",
        # "tokenizer_name": "/mnt/e/temp/date1/experiments/outputs",
        # "tokenizer_name": None,#None for train your tokenizer
        "tensorboard_dir": f'{tensorboard_dir}/{exp_prefix}/{run_name}',
        "extract_reps": extract_reps,
        "weight_decay": weight_decay,
        "optimizer": optimizer,
        "scores_export_path": f"./token_scores/{run_name}/",
        "generator_config": {
            "embedding_size": 128,
            "hidden_size": gen_hid_size,
            "num_hidden_layers": gen_hid_layers,
        },
        "discriminator_config": {
            "hidden_dropout_prob": disc_drop,
            "attention_probs_dropout_prob": disc_drop,
            "embedding_size": 128,
            "hidden_size": disc_hid_size,
            "num_hidden_layers": disc_hid_layers,
        },
        "mlm_lr_ratio": mlm_lr_ratio,
        "plot_conf_mtx":0,#plot confuse mtx
    }

    for subset_r in tqdm(subset):
        print('-' * 10, '\n', f'SUBSET: {subset_r}\n', '-' * 10)

        if preprocessed:
            train_file = f"../datasets/ag_od/train/{subset_r}.txt"
            test_file = f"../datasets/ag_od/test/{subset_r}.txt"
            outlier_file = f"../datasets/ag_od/test/{subset_r}-outliers.txt"
        if contamination != 0:
            train_file = f"../datasets/ag_od/train/{subset_r}-contaminated/{subset_r}_c{int(contamination)}.txt"

        model = LanguageModelingModel("electra",
                                      None,
                                    #   "/mnt/e/temp/date1/experiments/outputs",
                                      masks=masks_,
                                      args=train_args,
                                      train_files=train_file,
                                      use_cuda=True)

        # model.tb_add_embedding(train_file,
        #                        outlier_file,
        #                        tag="before_fine_tune")

        model.train_model_anomaly(train_file,
                                  eval_file=test_file,
                                  eval_file_outlier=outlier_file,
                                  sched_params=sched_params)

        # model.tb_add_embedding(train_file,
        #                        outlier_file,
        #                        tag="after_fine_tune")

        # result = model.predicts([
        #     "前方8公里3车事故，封闭左车道",#should be normal
        #     # "自然语言处理是一门融语言学、计算机科学、数学于一体的科学",#should be anomaly
        #     # "在自然语言处理领域中，预训练模型（pre-trainedmodels）已成为非常重要的基础技术。",#should be anomaly
        #     # "我们发现，放大或抑制知识神经元的激活可以相应地影响对事实的记忆。",#should be anomaly
        #     ],
        #     anom_word_threshold =.15,
        #     anom_word_alpha = 0,
        #     oov_alpha = 0,#
        #     positive_cutoff = .8
        #     )
        # result
run_exps()
