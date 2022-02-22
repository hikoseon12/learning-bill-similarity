import os
import ast
import math
import argparse
from pathlib import Path
from datetime import datetime
import logging

from utils.utils import open_csv, get_input_samples, save_test_val_result_csv
from torch.utils.data import DataLoader
import torch.nn as nn
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder_parallel
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyF1Evaluator


#### Some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--train_data_name', type=str, default='human_annoated', help='Data name: "human_annoated" or "synthetic"')
parser.add_argument('--train_data_size', type=int, default=3305, help='Data_size')
parser.add_argument('--pretrained_model', type=str, default='roberta-large', help='Pretrain model')
parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch_size')
parser.add_argument('--num_epochs', type=int, default=4, help='Num_epochs')
parser.add_argument('--gpu_list', type=str, default='[0,1,2,3]', help='GPU number list')
parser.add_argument('--nth_result', type=int, default=0, help='nth result')
parser.add_argument('--memo', type=str, default='', help='memo')
args = parser.parse_args()
logging.info(args)


train_data_name = args.train_data_name
train_data_size = args.train_data_size
pretrained_model = args.pretrained_model
lr = args.lr
batch_size = args.batch_size
num_epochs = args.num_epochs
gpu_list = ast.literal_eval(args.gpu_list)
nth_result = args.nth_result
memo = args.memo


# Load dataset (human_annotated or synthetic)
dataset_path = '../data/subsection_pairs/{}_pairs'.format(train_data_name)
train_sub_sec_pairs = open_csv('{}/{}_sub_sec_pairs.csv'.format(dataset_path,'train'))

# Validation & Testset are human-annotated set
human_annotated_dataset_path = '../data/subsection_pairs/human_annotated_pairs'
val_sub_sec_pairs = open_csv('{}/{}_sub_sec_pairs.csv'.format(human_annotated_dataset_path,'val'))
test_sub_sec_pairs = open_csv('{}/{}_sub_sec_pairs.csv'.format(human_annotated_dataset_path,'test'))


train_samples = get_input_samples(train_sub_sec_pairs)[:train_data_size]
val_samples = get_input_samples(val_sub_sec_pairs)
test_samples = get_input_samples(test_sub_sec_pairs)

# The dataset size of samples
logging.info("Train : {}, Validation: {}, Test : {}".format(len(train_samples), len(val_samples), len(test_samples)))

pretrained_model_short_name = pretrained_model
if os.path.exists('../trained_model/{}'.format(pretrained_model)):
    pretrained_model_short_name = "_".join(pretrained_model.split('/'))
    partent_path = Path(os.getcwd()).parent
    pretrained_model = '{}/trained_model/{}'.format(partent_path, pretrained_model)


max_length = None
if 'legal' in pretrained_model: # 'nlpaueb/legal-bert-small-uncased'
    pretrained_model_short_name = 'legal' 
    max_length = 512 # Define the number of max token for legal-bert


device_num = 'cuda:'+str(gpu_list[0])
model = CrossEncoder_parallel(pretrained_model, device=device_num, max_length=max_length, device_ids=gpu_list)
model_save_path = '../trained_model/{}_{}/{}/{}_{}_{}'.format(pretrained_model_short_name, train_data_name, train_data_size, lr, memo, nth_result)
save_csv_result = '{}_{}_{}_{}_{}_{}'.format(pretrained_model_short_name, train_data_name, train_data_size, lr, memo, nth_result)


save_setting_infos = [save_csv_result, pretrained_model_short_name, train_data_size, lr, memo, nth_result]
save_setting_names = ["name","model", "train_data_size", "lr", "memo", "nth"]

#We wrap train_samples, which is a list ot InputExample, in a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)

#During training, we use CESoftmaxAccuracyF1Evaluator to measure the accuracy & F1 on the validation set.
train_evaluator = CESoftmaxAccuracyF1Evaluator.from_input_examples(train_samples, save_setting_infos=save_setting_infos, save_setting_names=save_setting_names,  name=save_csv_result + '_train')
val_evaluator = CESoftmaxAccuracyF1Evaluator.from_input_examples(val_samples, save_setting_infos=save_setting_infos, save_setting_names=save_setting_names,name=save_csv_result + '_val')
test_evaluator = CESoftmaxAccuracyF1Evaluator.from_input_examples(test_samples, save_setting_infos=save_setting_infos, save_setting_names=save_setting_names,name=save_csv_result + '_test')


warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=val_evaluator,
          other_evaluators=[test_evaluator],
          epochs=num_epochs,
          evaluation_steps=10000,
          warmup_steps=warmup_steps,
          optimizer_params={'lr': lr, 'eps': 1e-6, 'correct_bias': False},
          output_path=model_save_path)

# Save test, validation result
save_test_val_result_csv(model_save_path, save_csv_result)
