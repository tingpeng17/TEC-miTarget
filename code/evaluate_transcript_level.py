# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 01:42:22 2023

@author: Tingpeng Yang
"""

# Make new predictions with a pre-trained model.
from __future__ import annotations
import argparse
import datetime
import sys
import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils import log
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Alphabet import generic_rna
import regex
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score,precision_recall_curve,roc_auc_score,roc_curve
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

def find_utr(seq):
    start=seq.find('AUG')
    if start != -1:
        while True:
            start=start+3
            if start+3 > len(seq):
                end=None
                break
            else:
                if seq[start:(start+3 )]=='UAG' or seq[start:(start+3 )]=='UAA' or seq[start:(start+3 )]=='UGA':
                    end=start+3
                    break
        if end != None:
            seq=list(seq)
            for i in range(0,end):
                seq[i]='N'
            temp=''
            for i in seq:
                temp=temp+i
            return temp
        else:
            print('End not found!')
            return seq # generally speaking, it's impossible for a mRNA
    else:
        print('Start not found!')
        return seq # generally speaking, it's impossible for a mRNA

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def find_candidate(mirna_sequence, mrna_sequence, seed_match):
    positions = set()
    
    '''
    mrna_sequence=list(mrna_sequence)
    start=int(0.75*len(mrna_sequence))
    for i in range(0,start):
        mrna_sequence[i]='N'
    temp=''
    for i in mrna_sequence:
        temp=temp+i
    mrna_sequence=temp
    '''
    '''
    mrna_sequence=find_utr(mrna_sequence)
    if mrna_sequence == None:
        return []
    '''

    if seed_match == '10-mer-m6':
        SEED_START = 1
        SEED_END = 10
        SEED_OFFSET = SEED_START - 1
        MIN_MATCH = 6
        TOLERANCE = (SEED_END-SEED_START+1) - MIN_MATCH
    elif seed_match == '10-mer-m7':
        SEED_START = 1
        SEED_END = 10
        SEED_OFFSET = SEED_START - 1
        MIN_MATCH = 7
        TOLERANCE = (SEED_END-SEED_START+1) - MIN_MATCH
    elif seed_match == 'offset-9-mer-m7':
        SEED_START = 2
        SEED_END = 10
        SEED_OFFSET = SEED_START - 1
        MIN_MATCH = 7
        TOLERANCE = (SEED_END-SEED_START+1) - MIN_MATCH 
    elif seed_match == 'custom':
        SEED_START = 1
        SEED_END = 13
        SEED_OFFSET = SEED_START - 1
        MIN_MATCH = 9
        TOLERANCE = (SEED_END-SEED_START+1) - MIN_MATCH 
    elif seed_match == 'strict':
        positions = find_strict_candidate(mirna_sequence, mrna_sequence)

        return positions
    else:
        raise ValueError("seed_match expected 'strict', '10-mer-m6', '10-mer-m7', or 'offset-9-mer-m7', got '{}'".format(seed_match))

    seed = mirna_sequence[(SEED_START-1):SEED_END]
    rc_seed = str(Seq(seed, generic_rna).complement())
    match_iter = regex.finditer("({}){{e<={}}}".format(rc_seed, TOLERANCE), mrna_sequence)

    for match_index in match_iter:
        #positions.add(match_index.start()) # slice-start indicies
        positions.add(match_index.end()+SEED_OFFSET) # slice-stop indicies

    positions = list(positions)

    return positions


def find_strict_candidate(mirna_sequence, mrna_sequence):
    positions = set()

    SEED_TYPES = ['8-mer', '7-mer-m8', '7-mer-A1', '6-mer', '6-mer-A1', 'offset-7-mer', 'offset-6-mer']
    for seed_match in SEED_TYPES:
        if seed_match == '8-mer':
            SEED_START = 2
            SEED_END = 8
            SEED_OFFSET = 0
            seed = 'U' + mirna_sequence[(SEED_START-1):SEED_END]
        elif seed_match == '7-mer-m8':
            SEED_START = 1
            SEED_END = 8
            SEED_OFFSET = 0
            seed = mirna_sequence[(SEED_START-1):SEED_END]
        elif seed_match == '7-mer-A1':
            SEED_START = 2
            SEED_END = 7
            SEED_OFFSET = 0
            seed = 'U' + mirna_sequence[(SEED_START-1):SEED_END]
        elif seed_match == '6-mer':
            SEED_START = 2
            SEED_END = 7
            SEED_OFFSET = 1
            seed = mirna_sequence[(SEED_START-1):SEED_END]
        elif seed_match == '6mer-A1':
            SEED_START = 2
            SEED_END = 6
            SEED_OFFSET = 0
            seed = 'U' + mirna_sequence[(SEED_START-1):SEED_END]
        elif seed_match == 'offset-7-mer':
            SEED_START = 3
            SEED_END = 9
            SEED_OFFSET = 0
            seed = mirna_sequence[(SEED_START-1):SEED_END]
        elif seed_match == 'offset-6-mer':
            SEED_START = 3
            SEED_END = 8
            SEED_OFFSET = 0
            seed = mirna_sequence[(SEED_START-1):SEED_END]

        rc_seed = str(Seq(seed, generic_rna).complement())
        match_iter = regex.finditer(rc_seed, mrna_sequence)

        for match_index in match_iter:
            #positions.add(match_index.start()) # slice-start indicies
            positions.add(match_index.end()+SEED_OFFSET) # slice-stop indicies

    positions = list(positions)

    return positions


def get_candidate(mirna_sequence, mrna_sequence, cts_size, seed_match):
    cts_size=len(mirna_sequence)
    positions = find_candidate(mirna_sequence, mrna_sequence, seed_match)

    candidates = []
    for i in positions:
        #site_sequence = mrna_sequence[max(0, i-cts_size):i]
        site_sequence = mrna_sequence[max(0, i-2*cts_size):min(i+cts_size,len(mrna_sequence))]
        rev_site_sequence = site_sequence[::-1]
        rc_site_sequence = str(Seq(rev_site_sequence, generic_rna).complement())
        candidates.append(rev_site_sequence) # miRNAs: 5'-ends to 3'-ends,  mRNAs: 3'-ends to 5'-ends
        #candidates.append(rc_site_sequence)

    return candidates, positions


def make_pair(mirna_sequence, mrna_sequence, cts_size, seed_match):
    candidates, positions = get_candidate(mirna_sequence, mrna_sequence, cts_size, seed_match)

    mirna_querys = []
    mrna_targets = []
    if len(candidates) == 0:
        return (mirna_querys, mrna_targets, positions)
    else:
        mirna_sequence = mirna_sequence[0:cts_size]
        for i in range(len(candidates)):
            mirna_querys.append(mirna_sequence)
            mrna_targets.append(candidates[i])

    return mirna_querys, mrna_targets, positions


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def read_fasta(mirna_fasta_file, mrna_fasta_file):
    mirna_list = list(SeqIO.parse(mirna_fasta_file, 'fasta'))
    mrna_list = list(SeqIO.parse(mrna_fasta_file, 'fasta'))

    mirna_ids = []
    mirna_seqs = []
    mrna_ids = []
    mrna_seqs = []

    for i in range(len(mirna_list)):
        mirna_ids.append(str(mirna_list[i].id))
        mirna_seqs.append(str(mirna_list[i].seq))

    for i in range(len(mrna_list)):
        mrna_ids.append(str(mrna_list[i].id))
        mrna_seqs.append(str(mrna_list[i].seq))

    return mirna_ids, mirna_seqs, mrna_ids, mrna_seqs


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def get_negative_pair(mirna_fasta_file, mrna_fasta_file, ground_truth_file=None, cts_size=30, seed_match='offset-9-mer-m7', header=False, predict_mode=True):
    mirna_ids, mirna_seqs, mrna_ids, mrna_seqs = read_fasta(mirna_fasta_file, mrna_fasta_file)

    dataset = {
        'query_ids': [],
        'target_ids': [],
        'predicts': []
    }

    if ground_truth_file is not None:
        query_ids, target_ids, labels = read_ground_truth(ground_truth_file, header=header)

        for i in range(len(query_ids)):
            try:
                j = mirna_ids.index(query_ids[i])
            except ValueError:
                continue
            try:
                k = mrna_ids.index(target_ids[i])
            except ValueError:
                continue

            query_seqs, target_seqs, locations = make_pair(mirna_seqs[j], mrna_seqs[k], cts_size=cts_size, seed_match=seed_match)

            n_pairs = len(locations)
            if (n_pairs == 0) and (predict_mode is True):
                dataset['query_ids'].append(query_ids[i])
                dataset['target_ids'].append(target_ids[i])
                dataset['predicts'].append(0)
            elif (n_pairs == 0) and (predict_mode is False):
                dataset['query_ids'].append(query_ids[i])
                dataset['target_ids'].append(target_ids[i])
                dataset['predicts'].append(labels[i])
    else:
        for i in range(len(mirna_ids)):
            for j in range(len(mrna_ids)):
                query_seqs, target_seqs, locations = make_pair(mirna_seqs[i], mrna_seqs[j], cts_size=cts_size, seed_match=seed_match)

                n_pairs = len(locations)
                if n_pairs == 0:
                    dataset['query_ids'].append(mirna_ids[i])
                    dataset['target_ids'].append(mrna_ids[j])
                    dataset['predicts'].append(0)

    dataset['target_locs'] = [-1 for i in range(len(dataset['query_ids']))]
    dataset['probabilities'] = [0.0 for i in range(len(dataset['query_ids']))]

    return dataset


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def read_ground_truth(ground_truth_file, header=True, train=False):
    # input format: [MIRNA_ID, MRNA_ID, LABEL]
    if header is True:
        records = pd.read_csv(ground_truth_file, header=0, sep='\t')
    else:
        records = pd.read_csv(ground_truth_file, header=None, sep='\t')

    query_ids = np.asarray(records.iloc[:, 0].values)
    target_ids = np.asarray(records.iloc[:, 1].values)
    if train is True:
        labels = np.asarray(records.iloc[:, 2].values)
    else:
        labels = np.full((len(records),), fill_value=-1)

    return query_ids, target_ids, labels


def make_input_pair(mirna_fasta_file, mrna_fasta_file, ground_truth_file, cts_size=30, seed_match='offset-9-mer-m7', header=True, train=True):
    mirna_ids, mirna_seqs, mrna_ids, mrna_seqs = read_fasta(mirna_fasta_file, mrna_fasta_file)
    query_ids, target_ids, labels = read_ground_truth(ground_truth_file, header=header, train=train)

    dataset = {
        'mirna_fasta_file': mirna_fasta_file,
        'mrna_fasta_file': mrna_fasta_file,
        'ground_truth_file': ground_truth_file,
        'query_ids': [],
        'query_seqs': [],
        'target_ids': [],
        'target_seqs': [],
        'target_locs': [],
        'labels': []
    }

    for i in range(len(query_ids)):
        try:
            j = mirna_ids.index(query_ids[i])
        except ValueError:
            continue
        try:
            k = mrna_ids.index(target_ids[i])
        except ValueError:
            continue

        query_seqs, target_seqs, locations = make_pair(mirna_seqs[j], mrna_seqs[k], cts_size=cts_size, seed_match=seed_match)

        n_pairs = len(locations)
        if n_pairs > 0:
            queries = [query_ids[i] for n in range(n_pairs)]
            dataset['query_ids'].extend(queries)
            dataset['query_seqs'].extend(query_seqs)
            targets = [target_ids[i] for n in range(n_pairs)]
            dataset['target_ids'].extend(targets)
            dataset['target_seqs'].extend(target_seqs)
            dataset['target_locs'].extend(locations)
            dataset['labels'].extend([labels[i] for p in range(n_pairs)])
        else:
            dataset['query_ids'].extend([query_ids[i]])
            dataset['query_seqs'].extend(['Not found'])
            dataset['target_ids'].extend([target_ids[i]])
            dataset['target_seqs'].extend(['Not found'])
            dataset['target_locs'].extend([0])
            dataset['labels'].extend([labels[i]])

    return dataset
def add_args(parser):
    # Create parser for command line utility
    parser.add_argument('--mirna_file', type=str,
                        help="miRNA fasta file (default: data/mirna.fasta)")
    parser.add_argument('--mrna_file', type=str,
                        help="mRNA fasta file (default: data/mrna.fasta)")
    parser.add_argument('--query_file', type=str,
                        help="query file to be queried in 'predict' mode (sample: templates/query_set.csv)")
    parser.add_argument("--model", help="Pretrained Model", required=True)
    parser.add_argument("-o", "--outfile", help="File for predictions")
    parser.add_argument("-d", "--device", type=int, default=-1, help="Compute device to use")
    parser.add_argument("--thresh",type=float,default=0.5,help="Positive prediction threshold - used to store contact maps and predictions in a separate file. [default: 0.5]")
    return parser
def plot_eval_predictions(labels, predictions,output, path="figure"):
    """
    Plot histogram of positive and negative predictions, precision-recall curve, and receiver operating characteristic curve.
    :param y: Labels
    :type y: np.ndarray
    :param phat: Predicted probabilities
    :type phat: np.ndarray
    :param path: File prefix for plots to be saved to [default: figure]
    :type path: str
    """
    pos_phat = predictions[labels == 1]
    neg_phat = predictions[labels == 0]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Distribution of Predictions")
    ax1.hist(pos_phat)
    ax1.set_xlim(0, 1)
    ax1.set_title("Positive")
    ax1.set_xlabel("p-hat")
    ax2.hist(neg_phat)
    ax2.set_xlim(0, 1)
    ax2.set_title("Negative")
    ax2.set_xlabel("p-hat")
    plt.savefig(path + "phat_dist.svg")
    plt.close()
    precision, recall, pr_thresh = precision_recall_curve(labels, predictions)
    aupr = average_precision_score(labels, predictions)
    log(f"AUPR: {aupr}",file=output)
    plt.step(recall, precision, color="b", alpha=0.2, where="post")
    plt.fill_between(recall, precision, step="post", alpha=0.2, color="b")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Precision-Recall (AUPR: {:.3})".format(aupr))
    plt.savefig(path + "aupr.svg")
    plt.close()
    fpr, tpr, roc_thresh = roc_curve(labels, predictions)
    auroc = roc_auc_score(labels, predictions)
    log(f"AUROC: {auroc}",file=output)
    plt.step(fpr, tpr, color="b", alpha=0.2, where="post")
    plt.fill_between(fpr, tpr, step="post", alpha=0.2, color="b")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Receiver Operating Characteristic (AUROC: {:.3})".format(auroc))
    plt.savefig(path + "auroc.svg")
    plt.close()
def get_tokens(rna_list,base_number_dict):
    # rna_list: [ [">AGCT-AA","AGCUAA"], [">AGCT-AA","AGCUAA"] ]
    for i in range(len(rna_list)):
        rna_list[i][1]=list(rna_list[i][1])
        for j in range(len(rna_list[i][1])):
            rna_list[i][1][j]=base_number_dict[rna_list[i][1][j]]
    return rna_list
def main(args):
    # Run new prediction from arguments.
    modelPath = args.model
    outPath = args.outfile
    device = args.device
    threshold = args.thresh
    # Set Outpath
    if outPath is None:
        outPath = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-predictions")
    logFilePath = outPath + "log.txt"
    logFile=logFilePath
    # Set Device
    if device == -1 :
        args.device='cpu'
        device = args.device
        log(
            "Using cpu",
            file=logFile,
            print_also=True,
        )
    elif device > -1:
        log(f"Using CUDA device {device} - {torch.cuda.get_device_name(device)}",file=logFile,print_also=True)
    else:
        log(
            "device must be an integer >= -1!",
            file=logFile,
            print_also=True,
        )
        sys.exit('device must be an integer >= -1!')
    # Load Model
    try:
        log(f"Loading model from {modelPath}", file=logFile, print_also=True)
        model = torch.load(modelPath)
        model=model.to(args.device)
        model.use_cuda = True
    except FileNotFoundError:
        log(f"Model {modelPath} not found", file=logFile, print_also=True)
        sys.exit(1)
    # Load Pairs
    dataset = make_input_pair(args.mirna_file, args.mrna_file, args.query_file, cts_size=30, seed_match='custom', header=True, train=True) # dict
    # 'strict', '10-mer-m6', '10-mer-m7', or 'offset-9-mer-m7','custom'
    temp=dataset.copy()
    del temp['mirna_fasta_file']
    del temp['mrna_fasta_file']
    del temp['ground_truth_file']
    pairs=pd.DataFrame(temp)
    # Make Predictions
    log("Making Predictions...", file=logFile, print_also=True)
    outPathAll = f"{outPath}predict.tsv"
    outPathPos = f"{outPath}predict-positive.tsv"
    cmap_file = h5py.File(f"{outPath}cmaps.h5", "w")
    model.eval()
    with open(outPathAll, "w+") as f:
        f.write('query_ids\tquery_seqs\ttarget_ids\ttarget_seqs\ttarget_locs\tlabels\tpredictions\n')
        with open(outPathPos, "w+") as pos_f:
            pos_f.write('query_ids\tquery_seqs\ttarget_ids\ttarget_seqs\ttarget_locs\tlabels\tpredictions\n')
            base_number_dict={"A":1,"G":2,"C":3,"U":4}
            with torch.no_grad():
                for _, (query_ids, query_seqs, target_ids, target_seqs, target_locs, labels) in tqdm(pairs.iloc[:, :].iterrows(), total=len(pairs)):
                    if query_seqs == 'Not found':
                        f.write(f"{query_ids}\t{query_seqs}\t{target_ids}\t{target_seqs}\t{target_locs}\t{labels}\t{0}\n")
                    else:
                        n0=query_seqs
                        n1=target_seqs
                        rnas = list(set([n0]).union([n1]))
                        rnas=[[i,i.replace('-','').replace('>','').replace("T", "U")] for i in rnas]
                        rna_list=get_tokens(rnas,base_number_dict)
                        for i in range(len(rna_list)):
                            rna_list[i][1]=torch.LongTensor(rna_list[i][1]).to(args.device)
                        embeddings={rna_list[i][0]:rna_list[i][1] for i in range(len(rna_list))}
                        p0 = embeddings[n0].reshape(1,-1,1)
                        p1 = embeddings[n1].reshape(1,-1,1)
                        p0=p0.to(args.device)
                        p1=p1.to(args.device)
                        cm, p = model.map_predict(p0, p1)
                        p = p.item()
                        f.write(f"{query_ids}\t{query_seqs}\t{target_ids}\t{target_seqs}\t{target_locs}\t{labels}\t{p}\n")
                        if p >= threshold:
                            pos_f.write(f"{query_ids}\t{query_seqs}\t{target_ids}\t{target_seqs}\t{target_locs}\t{labels}\t{p}\n")
                            cm_np = cm.squeeze().cpu().numpy()
                            dset = cmap_file.require_dataset(f"{n0}x{n1}", cm_np.shape, np.float32)
                            dset[:] = cm_np            
    cmap_file.close()
    df=pd.read_csv(outPathAll,sep='\t')
    idx = df.groupby(['query_ids', 'target_ids'])['predictions'].idxmax()
    df=df.loc[idx]
    predicted_labels=np.array(df['predictions'])
    labels=np.array(df['labels'])
    plot_eval_predictions(labels, predicted_labels, logFile,outPath)
    predicted_labels[predicted_labels>=threshold]=1
    predicted_labels[predicted_labels<threshold]=0
    df['predictions']=predicted_labels
    tn, fp, fn, tp = confusion_matrix(labels, predicted_labels).ravel()
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    sensitivity = tp/(tp+fn) 
    specificity = tn/(tn+fp)
    ppv = tp/(tp+fp)
    npv = tn/(tn+fn)
    f1score = 2*tp/(2*tp+fp+fn)
    log("accuracy: {:.4f},sensitivity: {:.4f},specificity: {:.4f},ppv: {:.4f},npv: {:.4f},F1 score: {:.4f}\n".format(accuracy,sensitivity,specificity,ppv,npv,f1score), file=logFile, print_also=True)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser = add_args(parser)
    main(parser.parse_args())