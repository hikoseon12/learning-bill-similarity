"""
This example shows the prediction of a label for a given subsection pair with the trained model
"""
import re
import string
import numpy as np
import pandas as pd
from utils.utils import open_csv
from sentence_transformers.cross_encoder import CrossEncoder_parallel

# You can use text_preprocessing() to remove special characters
def text_preprocessing(text):
    text = text.strip('\"').strip('\'').strip()
    text = re.sub(r'([{}])'.format(string.punctuation), r' ', text)
    text = re.sub('\s{2,}', ' ', text)  # pad punctuations for bpe
    text = text.strip()
    return text

# Preprocessed texts from text_preprocessing()
sample1_a_text = """
    The training developed and implemented under subsection a shall include the following 1 An overview of the fundamentals 
    of clinical pharmacology 2 Familiarization with principles on the utilization of pharmaceuticals in rehabilitation therapies
    3 Case studies on the utilization of pharmaceuticals for individuals with multiple complex injuries including Traumatic 
    Brain Injury TBI and Post Traumatic Stress Disorder PTSD 4 Familiarization with means of finding additional resources for 
    information on pharmaceuticals 5 Familiarization with basic elements of pain and pharmaceutical management 6 Familiarization 
    with complementary and alternative therapies
"""
sample1_b_text = """
    The training developed and implemented under subsection a shall include the following 1 An overview of the fundamentals of safe
    prescription drug use 2 Familiarization with the benefits and risks of using pharmaceuticals in rehabilitation therapies 3 
    Examples of the use of pharmaceuticals for individuals with multiple complex injuries including traumatic brain injury and post 
    traumatic stress disorder 4 Familiarization with means of finding additional resources for information on pharmaceuticals 5 
    Familiarization with basic elements of pain and pharmaceutical management 6 Familiarization with complementary and alternative 
    therapies
"""

sample2_a_text = """
    The program required under subsection c shall include 1 a delineation of key impact evaluation research and operations research 
    questions for main components of assistance provided under title I of this division 2 an identification of measurable performance 
    goals for each of the main components of assistance provided under title I of this division to be expressed in an objective and 
    quantifiable form at the inception of the program 3 the use of appropriate methods based on rigorous social science tools to measure
    program impact and operational efficiency and 4 adherence to a high standard of evidence in developing recommendations for adjustments
    to the assistance to enhance the impact of the assistance
"""
sample2_b_text = """
The program required under subsection c shall include 1 a delineation of key impact evaluation research and operations research
    questions for main components of assistance provided under the Merida Initiative 2 an identification of measurable performance
    goals for each of the main components of assistance provided under the Merida Initiative to be expressed in an objective and
    quantifiable form at the inception of the program 3 the use of appropriate methods based on rigorous social science tools to
    measure program impact and operational efficiency and 4 adherence to a high standard of evidence in developing recommendations
    for adjustments to such assistance to enhance the impact of such assistance
"""

sentence_combinations = [[sample1_a_text, sample1_b_text], [sample2_a_text, sample2_b_text]]

# Load trained model
model_path_root = '../trained_model/'
model_path = 'roberta-base_best_subsect_sim_model'
max_length = None # 512 for legal-bert
gpu_list = [0, 1, 2, 3]



device_num = 'cuda:'+str(gpu_list[0])
model = CrossEncoder_parallel(model_path_root + model_path, device=device_num, max_length=max_length, device_ids=gpu_list)
pred_scores, logits = model.predict(sentence_combinations,
                            convert_to_numpy=True, show_progress_bar=False)
pred_labels = np.argmax(pred_scores, axis=1)

# Outputs indicate the predicted classes among 0 (Unrelated) - 4 (Identical)
print("Predicted Results: \n{}".format(pred_labels))

