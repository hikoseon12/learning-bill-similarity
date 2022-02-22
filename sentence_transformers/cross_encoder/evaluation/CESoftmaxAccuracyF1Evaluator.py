import logging
import os
import csv
from typing import List
from ... import InputExample
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from torch import nn
import torch

class CESoftmaxAccuracyF1Evaluator:
    """
    This evaluator can be used with the CrossEncoder class.

    It is designed for CrossEncoders with 2 or more outputs. It measure the
    accuracy and F1 of the predict class vs. the gold labels.
    """

    def __init__(self, sentence_pairs: List[List[str]], save_setting_infos: List[str], save_setting_names: List[str], labels: List[int], name: str = ''):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.name = name
        self.datatype = name.split("_")[-1]
        self.save_setting_infos = save_setting_infos
        self.save_setting_names = save_setting_names

        self.csv_file = (name if name else '') + "_results.csv"
        self.csv_headers = ["data_type", "epoch",  "steps", "train_loss", "loss", "accuracy", "f1"]

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], save_setting_infos, save_setting_names, ** kwargs):
        sentence_pairs = []
        labels = []
        

        for example in examples:
            sentence_pairs.append(example.texts)
            labels.append(example.label)
            
        return cls(sentence_pairs, save_setting_infos, save_setting_names, labels, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, train_loss: float = None, device: int=4) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        num_label_class = list(set(self.labels))
        loss_fct = nn.CrossEntropyLoss()

        logging.info("CESoftmaxAccuracyF1Evaluator: Evaluating the model on " + self.name + " dataset" + out_txt)
        pred_scores, model_predictions_list = model.predict(self.sentence_pairs, convert_to_numpy=True, show_progress_bar=False)
        pred_labels = np.argmax(pred_scores, axis=1)


        assert len(pred_labels) == len(self.labels)
        labels_loss = torch.tensor(self.labels, dtype=torch.int64).to(str(device))
        pred_scores_loss = model_predictions_list.clone().detach().requires_grad_(True).view(len(model_predictions_list),-1)
        
        loss = loss_fct(pred_scores_loss, labels_loss).item()
        
        acc = np.sum(pred_labels == self.labels) / len(self.labels)
        classification_report_result = classification_report(self.labels, pred_labels, digits=4)
        confusion_matrix_result = confusion_matrix(self.labels, pred_labels)
        result_dict = classification_report(self.labels, pred_labels, output_dict=True)

        logging.info("{} Accuracy: {:.2f}".format(self.datatype, acc*100))
        logging.info("{} F1: {:.2f}".format(self.datatype, result_dict['macro avg']['f1-score']))
        logging.info("{} loss: {:.6f}".format(self.datatype, loss))

        if output_path is not None:
            df = pd.DataFrame(result_dict).transpose()

            save_setting_infos = self.save_setting_infos + \
                [self.datatype,epoch, steps, train_loss, loss, acc, result_dict['macro avg']['f1-score']]
            save_setting_names = self.save_setting_names + self.csv_headers
            for i, (info, name) in enumerate(zip(save_setting_infos, save_setting_names)):
                df.insert(i, name, [info]+[None]*(len(df)-1))

            df.insert(len(save_setting_infos), 'index', df.index)
            df_cm = pd.DataFrame(confusion_matrix_result)

            for i, label in enumerate(df_cm.index):
                df[str(label)] = df_cm[label].tolist()+[None]*(len(df)-len(df_cm))
            
            df.columns = [self.datatype+"_"+col for col in df.columns]
            
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                df.to_csv(csv_path, index=False)
            else:
                ori_df = pd.read_csv(csv_path)
                df = pd.concat([ori_df, df], ignore_index=True)
                df.to_csv(csv_path, index=False)

        return acc, loss
