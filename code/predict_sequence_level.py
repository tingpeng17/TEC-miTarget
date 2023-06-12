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
def add_args(parser):
    # Create parser for command line utility
    parser.add_argument("--pairs", help="Candidate rna pairs to predict", required=True)
    parser.add_argument("--model", help="Pretrained Model", required=True)
    parser.add_argument("-o", "--outfile", help="File for predictions")
    parser.add_argument("-d", "--device", type=int, default=-1, help="Compute device to use")
    parser.add_argument("--thresh",type=float,default=0.5,help="Positive prediction threshold - used to store contact maps and predictions in a separate file. [default: 0.5]")
    return parser
def get_tokens(rna_list,base_number_dict):
    # rna_list: [ [">AGCT-AA","AGCUAA"], [">AGCT-AA","AGCUAA"] ]
    for i in range(len(rna_list)):
        rna_list[i][1]=list(rna_list[i][1])
        for j in range(len(rna_list[i][1])):
            rna_list[i][1][j]=base_number_dict[rna_list[i][1][j]]
    return rna_list
def main(args):
    # Run new prediction from arguments.
    csvPath = args.pairs
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
    try:
        log(f"Loading pairs from {csvPath}", file=logFile, print_also=True)
        pairs = pd.read_csv(csvPath, sep="\t", header=None)
    except FileNotFoundError:
        log(f"Pairs File {csvPath} not found", file=logFile, print_also=True)
        sys.exit(1)
    # Make Predictions
    log("Making Predictions...", file=logFile, print_also=True)
    outPathAll = f"{outPath}predict.tsv"
    outPathPos = f"{outPath}predict-positive.tsv"
    cmap_file = h5py.File(f"{outPath}cmaps.h5", "w")
    model.eval()
    with open(outPathAll, "w+") as f:
        with open(outPathPos, "w+") as pos_f:
            base_number_dict={"A":1,"G":2,"C":3,"U":4}
            with torch.no_grad():
                for _, (n0, n1) in tqdm(pairs.iloc[:, :2].iterrows(), total=len(pairs)):
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
                    f.write(f"{n0}\t{n1}\t{p}\n")
                    if p >= threshold:
                        pos_f.write(f"{n0}\t{n1}\t{p}\n")
                    cm_np = cm.squeeze().cpu().numpy()
                    dset = cmap_file.require_dataset(f"{n0}x{n1}", cm_np.shape, np.float32)
                    dset[:] = cm_np
    cmap_file.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser = add_args(parser)
    main(parser.parse_args())