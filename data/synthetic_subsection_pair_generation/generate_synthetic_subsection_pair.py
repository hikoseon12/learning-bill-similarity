from nltk.util import ngrams
import pandas as pd
import numpy as np
import collections
from collections import defaultdict
import string
import random
import json
import time
import csv
import os
import re
import nltk
import spacy
import itertools

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
from nltk.corpus import wordnet, stopwords
from utils.utils import open_csv, save_csv, open_pkl, save_pkl


random.seed(0)

def get_different_longer_bill_sentence_sec(given_sec_id, sec_id_list, sec_bill_id_list, mix_proportion, sec_id_text_dict, sec_lengths):
    given_bill_id = '_'.join(given_sec_id.split('_')[:2])
    given_sec_text_length = sec_lengths[sec_id_list.index(given_sec_id)]
    longer_length = given_sec_text_length*mix_proportion

    different_bill_sec_list = [sec_id for sec_id, bill_id, sec_length in zip(sec_id_list, sec_bill_id_list, sec_lengths)
                               if (given_bill_id != bill_id) and sec_length > longer_length]
    different_bill_sec_id = random.sample(different_bill_sec_list, 1)
    return different_bill_sec_id[0]
 

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


# https://github.com/jasonwei20/eda_nlp/blob/14e43c68c875b9c086ee6f5281bae2a1f24a918a/code/eda.py#L65
def replace_to_synonyms(words, n):
    if n == 0:
        return words
    words = words.split()
    new_words = words.copy()
    stop_words = set(stopwords.words('english'))

    random_word_idx_list = list(range(len(words)))
    random.shuffle(random_word_idx_list)
    num_replaced = 0

    for random_word_idx in random_word_idx_list:
        random_word = words[random_word_idx]
        if (random_word.lower() in stop_words) or (len(random_word)<=1):
            continue
        synonyms = get_synonyms(random_word)

        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words[random_word_idx] = synonym
            num_replaced += 1
        if num_replaced >= n: #only replace up to n words
            break

    sentence = ' '.join(new_words)
    return sentence


def swap_text(text, n):
    if n == 0:
        return text
    token = text.split()
    new_words = token.copy()
    random_word_idx_list1 = list(range(len(token)))
    random_word_idx_list2 = list(range(len(token)))
    random.shuffle(random_word_idx_list1)
    random.shuffle(random_word_idx_list2)
    for i in range(n):
        random_word1 = new_words[random_word_idx_list1[i]]
        random_word2 = new_words[random_word_idx_list2[i]]
        new_words[random_word_idx_list1[i]] = random_word2
        new_words[random_word_idx_list2[i]] = random_word1
        
    sentence = ' '.join(new_words)
    return sentence


def apply_synonym_swap_for_class3(text, n):
    num_modification = random.choice(list(range(1, n)))
    num_synonym = random.choice(list(range(num_modification)))
    num_swap = num_modification - num_synonym
    text = replace_to_synonyms(text, num_synonym) # ablation study: removing this line, synonym replacement operation is not applied
    text = swap_text(text, num_swap) # ablation study: removing this line, swap operation is not applied
    return text


def apply_synonym_swap(text, n):
    num_modification = random.choice(list(range(n)))
    if num_modification != 0:
        num_synonym = random.choice(list(range(num_modification)))
        num_swap = num_modification - num_synonym 
        text = replace_to_synonyms(text, num_synonym) # ablation study: removing this line, synonym replacement operation is not applied
        text = swap_text(text, num_swap) # ablation study: removing this line, swap operation is not applied
    return text


def divide(lst, min_size, split_size):
    it = iter(lst)
    from itertools import islice
    size = len(lst)
    for i in range(split_size - 1, 0, -1):
        s = random.randint(min_size, size - min_size * i)
        yield list(islice(it, 0, s))
        size -= s
    yield list(it)


def sample_chunk_texts(text, target_len):
    text = text.split()
    min_words = 10
    max_num_chunk = max(int(target_len/min_words), 1)
    num_chunk = random.choice(list(range(1, min(max_num_chunk, 3)+1)))

    chunk_lens = []
    chunk_text_list = []

    if num_chunk > 1:
        random_n_list = list(divide(range(target_len), min_words, num_chunk))
        chunk_lens = [len(elem) for elem in random_n_list]
    else:
        chunk_lens = [target_len]

    start_idx = 0   
    for i, chunk_len in enumerate(chunk_lens[:-1]):
        add_pos = random.choice(
            range(start_idx, len(text)-sum(chunk_lens[i:])+1))
        start_idx = add_pos+chunk_len
        chunk_text = text[add_pos:add_pos+chunk_len]
        chunk_text_list.append(chunk_text)

    add_pos = random.choice(range(start_idx, len(text)-chunk_lens[-1]+1))
    chunk_text = text[add_pos:add_pos+chunk_lens[-1]]
    chunk_text_list.append(chunk_text)
    return chunk_text_list


def add_delete_subsection_chunk(text1, text2, min_ratio, max_ratio):
    target_text_len = len(text1.split())
    add_text2_ratio = random.uniform(min_ratio, max_ratio)
    delete_text1_ratio = random.uniform(min_ratio, max_ratio)
    text2_len = round(target_text_len*add_text2_ratio)
    text1_len = round(target_text_len*(1 - delete_text1_ratio))
    chunk1_lens = sample_chunk_texts(text1, text1_len)
    chunk2_lens = sample_chunk_texts(text2, text2_len)

    all_chunk_num = len(chunk1_lens) + len(chunk2_lens)
    chunk1_indices = random.sample(range(all_chunk_num), all_chunk_num)
    all_chunks = chunk1_lens + chunk2_lens

    text = ''
    for idx in range(all_chunk_num):
        text += ' '.join(all_chunks[chunk1_indices[idx]]) + ' '
    return text


def generate_synthetic_4_dict(text1, text2):
    return text1, text1

def generate_synthetic_3_dict(text1, text2):
    text = text1

    max_num_word_modification = min(int(len(text1.split())*0.1), 21)
    text = apply_synonym_swap_for_class3(text, max_num_word_modification)
    return text1, text


def generate_synthetic_2_dict(text1, text2):
    text = text1
    text = add_delete_subsection_chunk(text1, text2, 0.2, 0.4)

    max_num_word_modification = min(int(len(text1.split())*0.1), 21)
    text = apply_synonym_swap(text, max_num_word_modification)
    return text1, text


def generate_synthetic_1_dict(text1, text2):
    text = text2
    text = add_delete_subsection_chunk(text1, text2, 0.6, 0.8)

    max_num_word_modification = min(int(len(text1.split())*0.1), 21)
    text = apply_synonym_swap(text, max_num_word_modification)
    return text1, text


def generate_synthetic_0_dict(text1, text2):
    text = text2
    max_num_word_modification = min(int(len(text1.split())*0.1), 21)
    text = apply_synonym_swap(text, max_num_word_modification)
    
    return text1, text


def generate_synthetic_pairs_dict(synthetic_pairs_dict, class_num, label, synthetic_pair_func,
                                               sec_id_text_dict, sec_bill_id_list, sec_lengths, max_text_length, sec_id_list, class_size):
    random_seed = 0
    print("random_seed: ", random_seed, label, label+random_seed)
    random.seed(label+random_seed)

    target_idea_sec_list = random.choices(sec_id_list, k=class_size) # for short sentences

    for i, target_idea_sec_id in enumerate(target_idea_sec_list):
        if i % 10000 == 0:
            print("label ith", class_num, i)
        sec_a_id = target_idea_sec_id        
        sec_b_id = get_different_longer_bill_sentence_sec(
            sec_a_id, sec_id_list, sec_bill_id_list, 0.5, sec_id_text_dict, sec_lengths)

        if class_num == 1:
            sec_b_id = get_different_longer_bill_sentence_sec(
                sec_a_id, sec_id_list, sec_bill_id_list, 0.8, sec_id_text_dict, sec_lengths)

        subsec_a_text = " ".join(sec_id_text_dict[sec_a_id].split()[:max_text_length])
        subsec_b_text = " ".join(sec_id_text_dict[sec_b_id].split()[:max_text_length])        
        sec_a_text, sec_b_text = synthetic_pair_func(subsec_a_text, subsec_b_text)

        if class_num == 4:
            sec_b_id = sec_a_id

        sec_b_text = ' '.join(sec_b_text.split()[:max_text_length])

        synthetic_pairs_dict[i]['sec_a_id'] = sec_a_id
        synthetic_pairs_dict[i]['sec_b_id'] = sec_b_id
        synthetic_pairs_dict[i]['sec_a_text'] = sec_a_text
        synthetic_pairs_dict[i]['sec_b_text'] = sec_b_text
        synthetic_pairs_dict[i]['label'] = label

    return synthetic_pairs_dict


def generate_synthetic_section_pairs(class_size=10000, ablation_type=""):
    sec_id_text_dict = dict()
    
    synthetic_0_dict = defaultdict(dict)
    synthetic_1_dict = defaultdict(dict)
    synthetic_2_dict = defaultdict(dict)
    synthetic_3_dict = defaultdict(dict)
    synthetic_4_dict = defaultdict(dict)

    labels = [0,1,2,3,4]
    class_sizes = [class_size]*len(labels)
    class_nums = [0,1,2,3,4]

    synthetic_pairs_dicts = [synthetic_0_dict, synthetic_1_dict,
                                          synthetic_2_dict, synthetic_3_dict, synthetic_4_dict]
                                          
    synthetic_funcs = [generate_synthetic_0_dict, generate_synthetic_1_dict,
                       generate_synthetic_2_dict, generate_synthetic_3_dict, generate_synthetic_4_dict]

    max_text_length = 400
    sec_id_text_dict = open_pkl('../subsection_list/subsection_list.pkl')
    
    sec_id_list = list(sec_id_text_dict.keys())
    sec_bill_id_list = ['_'.join(sec_id.split('_')[:2]) for sec_id in sec_id_list]
    sec_lengths = [len(sec_id_text_dict[sec_id].split(" ")) for sec_id in sec_id_list] 

    synthetic_all_label_dict_list = []
    for synthetic_pairs_dict, synthetic_func, class_num, label, class_size in zip(synthetic_pairs_dicts, synthetic_funcs, class_nums, labels, class_sizes):
        synthetic_pairs_dict = generate_synthetic_pairs_dict(
            synthetic_pairs_dict, class_num, label, synthetic_func,
            sec_id_text_dict, sec_bill_id_list, sec_lengths, max_text_length, sec_id_list, class_size)
        synthetic_all_label_dict_list.append(synthetic_pairs_dict)

    final_synthetic_pair_dict = defaultdict(dict)
    for j in range(len(synthetic_all_label_dict_list[0])):
        for i, synthetic_pairs_dict in enumerate(synthetic_all_label_dict_list):
            final_synthetic_pair_dict[j*5+i] = synthetic_pairs_dict[j]

    final_synthetic_pairs_csv = pd.DataFrame.from_dict(final_synthetic_pair_dict, orient='index')

    save_pkl('../subsection_pairs/synthetic_pairs/sub_sec_pairs_'+ablation_type+'.pkl', final_synthetic_pair_dict)
    save_csv('../subsection_pairs/synthetic_pairs/sub_sec_pairs_'+ablation_type+'.csv', final_synthetic_pairs_csv)
    return


def main():
    # class_size: The size for each class. When we set class_size=200, the size of whole synthetic pairs is 200*5 = 10K) 
    # ablation_type: ['swap_syn', 'swap', 'syn', 'none']. Please refer `apply_synonym_swap()` and `apply_synonym_swap_for_class3()` for detailed settings.
    generate_synthetic_section_pairs(class_size=200, ablation_type='swap_syn')


if __name__ == '__main__':
    main()
