import clip
import numpy as np
import torch
import string

def produce_labels(sampled_text, anno, attr_list, gender_list, non_represents):
    clip_text = clip.tokenize(sampled_text)

    anno = np.array([int(a) > 0 for a in anno if a == '1' or a == '-1'], dtype = float)
    labels = anno.copy()

    all_labels = np.where(anno==1)[0]
    exist_mask = torch.zeros(40)
    sampled_text = sampled_text.lower()
    sampled_text_nosign = ''.join([i for i in sampled_text if i not in string.punctuation])
    sampled_tokens = sampled_text_nosign.split(' ')
    ############## process gender ##############
    for token in gender_list:
        if token in sampled_tokens:
            exist_mask[20] = 1
            break
    ###########################################

    for i in range(len(all_labels)):
        attr_label = attr_list[all_labels[i]]  
        if attr_label == 'male':
            continue                  
        if attr_label == 'no beard' and attr_label in sampled_text_nosign:
            exist_mask[all_labels[i]] = 1
            # import pdb; pdb.set_trace()
            continue
        if attr_label == 'big nose' and attr_label in sampled_text_nosign:
            exist_mask[all_labels[i]] = 1
            continue

        tmp_exist = False
        split_attr_label = attr_label.split(' ')
        for a in split_attr_label:
            if a not in non_represents and a in sampled_text_nosign:
                tmp_exist = True
                exist_mask[all_labels[i]] = 1
                break
        if tmp_exist == False:
            labels[all_labels[i]] = 0

    return clip_text, labels, exist_mask