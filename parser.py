#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-

import os, sys, re, collections
import json
from itertools import combinations, permutations
import random
import datetime
import shutil

def concentrate_dialogue_topic_act_emotion(dialogue_filename, output_filename, topic_filename = 'data/dialogues_topic.txt', act_filename = 'data/dialogues_action.txt', emotion_filename = 'data/dialogues_emotion.txt'):
    topic_list = [line.strip() for line in open(topic_filename, 'r').readlines()]
    act_list = [line.strip() for line in open(act_filename, 'r').readlines()]
    emotion_list = [line.strip() for line in open(emotion_filename, 'r').readlines()]
    dialogue_list = [line.strip() for line in open(dialogue_filename, 'r').readlines()]
    if len(dialogue_list) == len(topic_list) == len(act_list) == len(emotion_list):
        output_file = open(output_filename, 'w')
        for i in range(len(dialogue_list)):
            output_file.write(dialogue_list[i] + '\t' + topic_list[i] + '\t' + act_list[i] + '\t' + emotion_list[i] + '\n')
        output_file.close()
    else:
        print('Data Error!')

def generate_data_for_monolingual(input_filename, output_filename, target_filename, flag, language_flag):
    def not_empty(s):
        return s and s.strip()
    output_file = open(output_filename, 'w')
    target_file = open(target_filename, 'w')
    for line in open(input_filename, 'r').readlines():
        lineArr = line.strip().split('\t')
        # data = lineArr[0].strip()[::-1].replace('__uoe__', '', 1)[::-1].strip().split('__eou__')
        data = lineArr[0].strip().split('__eou__')
        data = list(filter(not_empty,data))
        for i in range(1, len(data)-1, 2):
            context = data[:i]
            response = data[i].strip()
            current_tae = [lineArr[1][0], lineArr[2].strip().split(' ')[i], lineArr[3].strip().split(' ')[i]]
            output_file.write(' '.join(current_tae) + '</s>' + '</s>'.join(context) + ' ' + language_flag + '\n')
            target_file.write(response + '\n')
    output_file.close()
    target_file.close()

def generate_data_for_multilingual(input_filename, input_target_filename, output_filename, target_filename):
    output_file = open(output_filename, 'a')
    for line in open(input_filename, 'r').readlines():
        output_file.write(line)
    output_file.close()

    target_file = open(target_filename, 'a')
    for line in open(input_target_filename, 'r').readlines():
        target_file.write(line)
    target_file.close()

def get_utterance_pair(en_input_filename, other_input_filename, output_filename):
    en_utterance = []
    other_utterance = []
    
    other_input_file_name = other_input_filename
    en_input_file_name = en_input_filename

    en_input_file_list = [en_input_filename, en_input_file_name.replace('train','dev'), en_input_file_name.replace('train','test')]
    other_input_file_list = [other_input_filename, other_input_file_name.replace('train','dev'), other_input_file_name.replace('train','test')]
    for en_input_filename in en_input_file_list:
        for line in open(en_input_filename, 'r').readlines():
            lineArr = line.strip().split('__eou__')
            for la in lineArr:
                en_utterance.append(la.strip())
    for other_input_filename in other_input_file_list:
        for line in open(other_input_filename, 'r').readlines():
            lineArr = line.strip().split('__eou__')
            for la in lineArr:
                other_utterance.append(la.strip())
    if len(en_utterance) == len(other_utterance):
        output_file = open(output_filename, 'w')
        for i in range(len(en_utterance)):
            output_file.write(en_utterance[i] + '\t' + other_utterance[i] + '\n')
        output_file.close()
    else:
        print('Data Error!')

def generate_data_for_crosslingual(input_src_filename, language_flag, dict_filename, output_src_filename, output_tgt_filename, flag):
    def not_empty(s):
        return s and s.strip()
    current_dict = {}
    for line in open(dict_filename, 'r').readlines():
        lineArr = line.strip().split('\t')
        if len(lineArr) == 2:
            if flag == 'En':
                current_dict[lineArr[0].strip()] = lineArr[1].strip()
            else:
                current_dict[lineArr[1].strip()] = lineArr[0].strip()

    output_file = open(output_src_filename, 'w')
    output_tgt_file = open(output_tgt_filename, 'w')
    for line in open(input_src_filename, 'r').readlines():
        lineArr = line.strip().split('\t')
        # data = lineArr[0].strip()[::-1].replace('__uoe__', '', 1)[::-1].strip().split('__eou__')
        data = lineArr[0].strip().split('__eou__')
        data = list(filter(not_empty,data)) 
        for i in range(1, len(data), 2):
            context = data[:i]
            response = data[i].strip()
            crosslingual_context = []
            for c in context:
                crosslingual_context.append(current_dict[c.strip()])
            current_tae = [lineArr[1][0], lineArr[2].strip().split(' ')[i], lineArr[3].strip().split(' ')[i]]
            # print(current_tae, crosslingual_context, response, i, len(current_tae))
            output_file.write(' '.join(current_tae) + '</s>' + '</s>'.join(crosslingual_context) + ' ' + language_flag + '\n')
            output_tgt_file.write(response + '\n')
    output_file.close()
    output_tgt_file.close()

def get_dialog_dict(en2a_filename, en2b_filename): #a2en + en2b -->a2b
    a2en = {}
    for line in open(en2a_filename, 'r').readlines():
        lineArr = line.strip().split('\t')
        if len(lineArr) == 2:
            a2en[lineArr[1].strip()] = lineArr[0].strip()
    en2b = {}
    for line in open(en2b_filename, 'r').readlines():
        lineArr = line.strip().split('\t')
        if len(lineArr) == 2:
            en2b[lineArr[0].strip()] = lineArr[1].strip()
    a2b = {}
    for k in a2en:
        a2b[k] = en2b[a2en[k]]
        # print(k + '\t' + en2b[a2en[k]])
    return a2b

def generate_data_for_crosslingual_no_En(input_src_filename, language_flag, en2a_filename, en2b_filename, output_src_filename, output_tgt_filename):
    current_dict = get_dialog_dict(en2a_filename, en2b_filename)

    output_file = open(output_src_filename, 'w')
    output_tgt_file = open(output_tgt_filename, 'w')
    for line in open(input_src_filename, 'r').readlines():
        lineArr = line.strip().split('\t')
        # data = lineArr[0].strip()[::-1].replace('__uoe__', '', 1)[::-1].strip().split('__eou__')
        data = lineArr[0].strip().split('__eou__')
        for i in range(1, len(data), 2):
            context = data[:i]
            response = data[i].strip()
            crosslingual_context = []
            for c in context:
                crosslingual_context.append(current_dict[c.strip()])
            current_tae = [lineArr[1][0], lineArr[2].strip().split(' ')[i], lineArr[3].strip().split(' ')[i]]
            # print(current_tae, crosslingual_context, response, i, len(current_tae))
            output_file.write(' '.join(current_tae) + '</s>' + '</s>'.join(crosslingual_context) + ' ' + language_flag + '\n')
            output_tgt_file.write(response + '\n')
    output_file.close()
    output_tgt_file.close()

def mkdirs(root):
    try:
        os.mkdir(os.path.join(root, "multilingual"))
        lans = ["En", "Zh", "De", "It"]
        os.mkdir(os.path.join(root, "monolingual"))
        for lan in lans:
            os.mkdir(os.path.join(root, "monolingual", lan))
        lans = ["En_Zh", "Zh_En", "En_De", "De_En"]
        os.mkdir(os.path.join(root, "crosslingual"))
        for lan in lans:
            os.mkdir(os.path.join(root, "crosslingual", lan))
    except:
        pass

if __name__ == '__main__':
    mkdirs("./data")
    #concentrate_dialogue_topic_act_emotion('data/dialogues_text_En.txt', 'data/En.txt')
    #concentrate_dialogue_topic_act_emotion('data/dialogues_text_Zh.txt', 'data/Zh.txt')
    #concentrate_dialogue_topic_act_emotion('data/dialogues_text_De.txt', 'data/De.txt')
    #concentrate_dialogue_topic_act_emotion('data/dialogues_text_It.txt', 'data/It.txt')

    
    #monolingual
    generate_data_for_monolingual('data/en_train_human.txt', 'data/monolingual/En/train.src', 'data/monolingual/En/train.tgt', 1, '<En>')
    generate_data_for_monolingual('data/zh_train_human.txt', 'data/monolingual/Zh/train.src', 'data/monolingual/Zh/train.tgt', 1, '<Zh>')
    generate_data_for_monolingual('data/de_train_human.txt', 'data/monolingual/De/train.src', 'data/monolingual/De/train.tgt', 1, '<De>')
    generate_data_for_monolingual('data/it_train_human.txt', 'data/monolingual/It/train.src', 'data/monolingual/It/train.tgt', 1, '<It>')

    generate_data_for_monolingual('data/en_dev_human.txt', 'data/monolingual/En/dev.src', 'data/monolingual/En/dev.tgt', 1, '<En>')
    generate_data_for_monolingual('data/zh_dev_human.txt', 'data/monolingual/Zh/dev.src', 'data/monolingual/Zh/dev.tgt', 1, '<Zh>')
    generate_data_for_monolingual('data/de_dev_human.txt', 'data/monolingual/De/dev.src', 'data/monolingual/De/dev.tgt', 1, '<De>')
    generate_data_for_monolingual('data/it_dev_human.txt', 'data/monolingual/It/dev.src', 'data/monolingual/It/dev.tgt', 1, '<It>')
    
    generate_data_for_monolingual('data/en_test_human.txt', 'data/monolingual/En/test.src', 'data/monolingual/En/test.tgt', 1, '<En>')
    generate_data_for_monolingual('data/zh_test_human.txt', 'data/monolingual/Zh/test.src', 'data/monolingual/Zh/test.tgt', 1, '<Zh>')
    generate_data_for_monolingual('data/de_test_human.txt', 'data/monolingual/De/test.src', 'data/monolingual/De/test.tgt', 1, '<De>')
    generate_data_for_monolingual('data/it_test_human.txt', 'data/monolingual/It/test.src', 'data/monolingual/It/test.tgt', 1, '<It>')
    #multilingual
    try:
        os.remove("data/multilingual/train.src")
        os.remove("data/multilingual/train.tgt")
    except:
        pass
    generate_data_for_multilingual('data/monolingual/De/train.src', 'data/monolingual/De/train.tgt', 'data/multilingual/train.src', 'data/multilingual/train.tgt')
    generate_data_for_multilingual('data/monolingual/En/train.src', 'data/monolingual/En/train.tgt', 'data/multilingual/train.src', 'data/multilingual/train.tgt')
    generate_data_for_multilingual('data/monolingual/It/train.src', 'data/monolingual/It/train.tgt', 'data/multilingual/train.src', 'data/multilingual/train.tgt')
    generate_data_for_multilingual('data/monolingual/Zh/train.src', 'data/monolingual/Zh/train.tgt', 'data/multilingual/train.src', 'data/multilingual/train.tgt')

    generate_data_for_multilingual('data/monolingual/De/dev.src', 'data/monolingual/De/dev.tgt', 'data/multilingual/dev.src', 'data/multilingual/dev.tgt')
    generate_data_for_multilingual('data/monolingual/En/dev.src', 'data/monolingual/En/dev.tgt', 'data/multilingual/dev.src', 'data/multilingual/dev.tgt')
    generate_data_for_multilingual('data/monolingual/It/dev.src', 'data/monolingual/It/dev.tgt', 'data/multilingual/dev.src', 'data/multilingual/dev.tgt')
    generate_data_for_multilingual('data/monolingual/Zh/dev.src', 'data/monolingual/Zh/dev.tgt', 'data/multilingual/dev.src', 'data/multilingual/dev.tgt')
    
    generate_data_for_multilingual('data/monolingual/De/test.src', 'data/monolingual/De/test.tgt', 'data/multilingual/test.src', 'data/multilingual/test.tgt')
    generate_data_for_multilingual('data/monolingual/En/test.src', 'data/monolingual/En/test.tgt', 'data/multilingual/test.src', 'data/multilingual/test.tgt')
    generate_data_for_multilingual('data/monolingual/It/test.src', 'data/monolingual/It/test.tgt', 'data/multilingual/test.src', 'data/multilingual/test.tgt')
    generate_data_for_multilingual('data/monolingual/Zh/test.src', 'data/monolingual/Zh/test.tgt', 'data/multilingual/test.src', 'data/multilingual/test.tgt')
    #crosslingual
    get_utterance_pair('data/en_train_human.txt', 'data/zh_train_human.txt', 'data/En2Zh.txt')
    get_utterance_pair('data/en_train_human.txt', 'data/de_train_human.txt', 'data/En2De.txt')
    get_utterance_pair('data/en_train_human.txt', 'data/it_train_human.txt', 'data/En2It.txt')
    # generate_data_for_crosslingual_no_En('data/De.txt', '<De>', 'data/En2De.txt', 'data/En2Nl.txt', 'data/crosslingual/Nl_De/train.src', 'data/crosslingual/Nl_De/train.tgt')
    generate_data_for_crosslingual('data/zh_train_human.txt', '<Zh>', 'data/En2Zh.txt', 'data/crosslingual/En_Zh/train.src', 'data/crosslingual/En_Zh/train.tgt', 'Zh')
    generate_data_for_crosslingual('data/en_train_human.txt', '<En>', 'data/En2Zh.txt', 'data/crosslingual/Zh_En/train.src', 'data/crosslingual/Zh_En/train.tgt', 'En')
    generate_data_for_crosslingual('data/de_train_human.txt', '<De>', 'data/En2De.txt', 'data/crosslingual/En_De/train.src', 'data/crosslingual/En_De/train.tgt', 'De')
    generate_data_for_crosslingual('data/en_train_human.txt', '<En>', 'data/En2De.txt', 'data/crosslingual/De_En/train.src', 'data/crosslingual/De_En/train.tgt', 'En')

    generate_data_for_crosslingual('data/zh_dev_human.txt', '<Zh>', 'data/En2Zh.txt', 'data/crosslingual/En_Zh/dev.src', 'data/crosslingual/En_Zh/dev.tgt', 'Zh')
    generate_data_for_crosslingual('data/en_dev_human.txt', '<En>', 'data/En2Zh.txt', 'data/crosslingual/Zh_En/dev.src', 'data/crosslingual/Zh_En/dev.tgt', 'En')
    generate_data_for_crosslingual('data/de_dev_human.txt', '<De>', 'data/En2De.txt', 'data/crosslingual/En_De/dev.src', 'data/crosslingual/En_De/dev.tgt', 'De')
    generate_data_for_crosslingual('data/en_dev_human.txt', '<En>', 'data/En2De.txt', 'data/crosslingual/De_En/dev.src', 'data/crosslingual/De_En/dev.tgt', 'En')

    generate_data_for_crosslingual('data/zh_test_human.txt', '<Zh>', 'data/En2Zh.txt', 'data/crosslingual/En_Zh/test.src', 'data/crosslingual/En_Zh/test.tgt', 'Zh')
    generate_data_for_crosslingual('data/en_test_human.txt', '<En>', 'data/En2Zh.txt', 'data/crosslingual/Zh_En/test.src', 'data/crosslingual/Zh_En/test.tgt', 'En')
    generate_data_for_crosslingual('data/de_test_human.txt', '<De>', 'data/En2De.txt', 'data/crosslingual/En_De/test.src', 'data/crosslingual/En_De/test.tgt', 'De')
    generate_data_for_crosslingual('data/en_test_human.txt', '<En>', 'data/En2De.txt', 'data/crosslingual/De_En/test.src', 'data/crosslingual/De_En/test.tgt', 'En')
    
    os.mkdir("./data/raw")
    shutil.move("./data/monolingual", "./data/raw/monolingual")
    shutil.move("./data/crosslingual", "./data/raw/crosslingual")
    shutil.move("./data/multilingual", "./data/raw/multilingual")
