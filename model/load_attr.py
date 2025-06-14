import os
import torch
import torch.nn as nn
import clip
import numpy as np
import json
from tqdm import tqdm


def attr_clustering(dataset_name='pascal_voc', num_atrr_clusters=112, json_file='./gpt4.0_cluster_a_photo_of4.json'):

    with open(json_file) as json_file:
        descriptions = json.load(json_file)
    num_classes = len(tuple(descriptions.keys()))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_type = "ViT-B/16"
    model_type_ = model_type.replace('/','-')
    model_name= 'clip'
    model, preprocess = clip.load(model_type, device)
    model.eval()

    all_descriptions_embeddings = []
    for class_name, desc in tqdm(descriptions.items()):
        
        sentences = [item.lower() for item in desc]
        # Tokenize sentences
        encoded_input = clip.tokenize(sentences).to(device)    
        # Compute token embeddings
        with torch.no_grad():
            sentence_embeddings = model.encode_text(encoded_input)
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1).cpu()
            all_descriptions_embeddings.append(sentence_embeddings)
    all_descriptions_embeddings_tensor = torch.cat(all_descriptions_embeddings, dim=0)

    # Choose proper cluster number for different datasets. We set 256 for ADE20K.
    n_clusters = num_atrr_clusters
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_descriptions_embeddings_tensor.numpy())
    desc_class_idx_list = []
    for class_idx, class_desc in enumerate(all_descriptions_embeddings):
        desc_class_idx_list.append(torch.tensor([class_idx] * len(class_desc)))
    desc_class_idx_list = torch.cat(desc_class_idx_list)
    ground_truth_all_classes = []
    activated_clusters_all_classes = []
    for class_idx in range(num_classes):
        class_cluster_info = kmeans.labels_[desc_class_idx_list == class_idx]
        activated_clusters = np.unique(class_cluster_info)
        print(activated_clusters)
        activated_clusters_all_classes.append(activated_clusters)
        ground_truth = np.zeros(n_clusters)
        ground_truth[activated_clusters] = 1
        ground_truth_all_classes.append(ground_truth)

    unique_label, unique_count =  np.unique(np.array(ground_truth_all_classes), axis=0, return_counts=True)
    print(unique_count)
    if not (unique_count == 1).all():
        confused_labels = np.where((np.array(ground_truth_all_classes) == unique_label[unique_count>1]).all(axis=1))
        for class_idx in confused_labels[0]:
            print('----------------')
            print(tuple(descriptions.keys())[class_idx])
            print(descriptions[tuple(descriptions.keys())[class_idx]])
            print('----------------')

    ground_truth_all_classes = torch.tensor(np.array(ground_truth_all_classes)).float()

    cluster_embedding_bank = kmeans.cluster_centers_.transpose(1, 0)
    cluster_embedding_bank = torch.tensor(cluster_embedding_bank)

    cluster_bank = [cluster_embedding_bank, ground_truth_all_classes]

    print(f'{dataset_name}_desc_{model_name}_{model_type_}_gpt4.0_attr_cluster_{n_clusters} loaded.')
    torch.save(cluster_bank, f'./attributes_text/{dataset_name}_desc_{model_name}_{model_type_}_gpt4.0_cluster_{n_clusters}_embedding_bank.pth')

    # ###TODO Mean_avgdes
    # Avg_des = all_descriptions_embeddings_tensor.reshape(20,20,512).mean(1).transpose(1,0)
    # cluster_bank = [Avg_des, ground_truth_all_classes]


    ###TODO select descs
    # Avg_des = all_descriptions_embeddings_tensor.reshape(20,20,512)[:,10,:].transpose(1,0)
    # cluster_bank = [Avg_des, ground_truth_all_classes]

    return cluster_bank

def attr_aggregate(text_features, dataset_name='pascal_voc', num_classes=20, num_atrr_clusters=112, json_file='./gpt4.0_cluster_a_photo_of4.json',topK=0.9):

    attr_pth_file = f'./attributes_text/{dataset_name}_desc_clip_ViT-B-16_gpt4.0_cluster_{num_atrr_clusters}_embedding_bank.pth'
    if not os.path.exists(attr_pth_file):
        attr_features, attr_flag = attr_clustering(dataset_name, num_atrr_clusters, json_file)
    else:
        attr_features, attr_flag = torch.load(attr_pth_file)
    
    attr_features, attr_flag = attr_features.cuda(), attr_flag.cuda()
    fg_text = text_features[:num_classes]           
    bg_text = text_features[num_classes:]

    corr = (fg_text @ attr_features).softmax(dim=-1)
    if topK is not None:
        topk = int((1-topK)* attr_features.shape[1])
        attn_logit = (fg_text @ attr_features)
        corr, idx = torch.sort(attn_logit, dim=-1, descending=True)
        # ### TODO select attr
        # idx_select = idx[:,57]
        # attr_se = torch.gather(attr_features,1,idx_select.unsqueeze(0).expand(512,-1))
        # ###
        corr[:,-topk:] = float('-inf')
        restored_corr = torch.zeros_like(corr)
        restored_corr.scatter_(-1, idx, corr)
        corr = restored_corr

        corr = corr.softmax(dim=-1)
    text_attri_agg = corr @ attr_features.t() + fg_text
    # text_attri_agg = attr_se.permute(1,0)
    # text_attri_agg = attr_features.t() + fg_text

    text_attri_agg = torch.cat([text_attri_agg,bg_text], dim=0)
    text_attri_agg = (text_attri_agg / text_attri_agg.norm(dim=1, keepdim=True)).permute(1,0)

    return text_attri_agg, attr_flag