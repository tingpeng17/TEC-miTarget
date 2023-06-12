# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 01:41:42 2023

@author: Tingpeng Yang

Train a new model.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import average_precision_score,precision_recall_curve,roc_auc_score,roc_curve
from sklearn.metrics import confusion_matrix
import sys
import random
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
from utils import (
    PairedDataset,
    collate_paired_sequences,
    log,
)
from models.transform import EmbeddingTransform
from models.contact import ContactCNN
from models.interaction import ModelInteraction
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os    

def add_args(parser):
    # Create parser for command line utility.

    # Data
    parser.add_argument("--train", required=True, help="the url of the train file", type=str)
    parser.add_argument("--valid", required=True, help="the url of the valid file", type=str)
    # Embedding model
    parser.add_argument("--input-dim",type=int,default=512,help="dimension of bert (per base of RNA) (default: 512)")
    parser.add_argument("--projection-dim",type=int,default=256,help="dimension of embedding projection layer (default: 256)")
    parser.add_argument("--dropout-p",type=float,default=0.0,help="parameter p for embedding dropout layer (default: 0.0)")
    parser.add_argument("--nhead",type=int,default=1,help="number of heads for Transformer Encoder (default: 1)")
    parser.add_argument("--num-layers",type=int,default=6,help="number of layers for Transformer Encoder (default: 6)")

    # Contact model
    parser.add_argument("--ks",type=int,default=9,help="width of the convolutional filter  (default: 9)")

    # Interaction Model
    parser.add_argument("--no-w",action="store_true",help="no use of weight matrix in interaction prediction model")
    parser.add_argument("--no-sigmoid",action="store_true",help="no use of sigmoid activation at end of interaction model")
    parser.add_argument("--do-pool",action="store_true",help="use max pool layer in interaction prediction model")
    parser.add_argument("--pool-width",type=int,default=9,help="size of max-pool in interaction model (default: 9)")
    parser.add_argument("--p0",type=float,default=0.5,help="p0 of LogisticActivation function (default: 0.5)")

    # Training
    parser.add_argument("--num-epochs",type=int,default=40,help="number of epochs (default: 40)")
    parser.add_argument("--seed",type=int,default=1234,help="random seed")

    parser.add_argument("--batch-size",type=int,default=32,help="minibatch size (default: 32)")
    parser.add_argument("--weight-decay",type=float,default=0,help="L2 regularization (default: 0)")
    parser.add_argument("--lr",type=float,default=0.0001,help="learning rate (default: 0.0001)")
    parser.add_argument("--lambda",dest="interaction_weight",type=float,default=1.0,help="weight on the similarity objective (default: 1.0)")

    # Output
    parser.add_argument("-o", "--outdir", help="output file path",default='./model/')
    parser.add_argument("-d", "--device", type=int, default=0, help="which device to use")
    parser.add_argument("--checkpoint", help="checkpoint model to start training from")

    return parser


def predict_cmap_interaction(args,model, n0, n1, tensors):  
    """
    Predict whether a list of RNA pairs will interact, as well as their contact map.

    :param model: Model to be trained
    :type model: models.interaction.ModelInteraction
    :param n0: First RNA names
    :type n0: list[str]
    :param n1: Second RNA names
    :type n1: list[str]
    :param tensors: Dictionary of RNA names to embeddings
    :type tensors: dict[str, torch.Tensor]
    """

    b = len(n0)
    z_a,z_b=[],[]
    for i in range(b):
        z_a.append(tensors[n0[i]])
        z_b.append(tensors[n1[i]])
    z_a=torch.nn.utils.rnn.pad_sequence(z_a,batch_first=True).reshape(b,-1,1)
    z_b=torch.nn.utils.rnn.pad_sequence(z_b,batch_first=True).reshape(b,-1,1)
    c_map_mag, p_hat = model.map_predict(z_a, z_b)
    return c_map_mag, p_hat
    


def predict_interaction(args,model, n0, n1, tensors):
    """
    Predict whether a list of rna pairs will interact.

    :param model: Model to be trained
    :type model: models.interaction.ModelInteraction
    :param n0: First rna names
    :type n0: list[str]
    :param n1: Second rna names
    :type n1: list[str]
    :param tensors: Dictionary of rna names to embeddings
    :type tensors: dict[str, torch.Tensor]
    """
    _, p_hat = predict_cmap_interaction(args,model, n0, n1, tensors)
    return p_hat


def interaction_grad(
    args,
    model,
    n0,
    n1,
    y,
    tensors,
    accuracy_weight=0.35
):
    """
    Compute gradient and backpropagate loss for a batch.

    :param model: Model to be trained
    :type model: models.interaction.ModelInteraction
    :param n0: First rna names
    :type n0: list[str]
    :param n1: Second rna names
    :type n1: list[str]
    :param y: Interaction labels
    :type y: torch.Tensor
    :param tensors: Dictionary of rna names to embeddings
    :type tensors: dict[str, torch.Tensor]
    :param accuracy_weight: Weight on the accuracy objective. Representation loss is : 1 - accuracy_weight.
    :type accuracy_weight: float

    :return: loss.item(), mse, p_guess.float().tolist(),y.int().tolist(),p_hat.int().tolist()
    :rtype: (float, float, list, list, list)
    """
    c_map_mag, p_hat = predict_cmap_interaction(
        args,model, n0, n1, tensors
    )
    y = y.to(args.device)
    y = Variable(y)
    p_hat = p_hat.float()
    #print(y.shape)
    #print(p_hat.shape)
    bce_loss = F.binary_cross_entropy(p_hat.float(), y.float())
    accuracy_loss = bce_loss
    representation_loss = torch.mean(c_map_mag)
    loss = (accuracy_weight * accuracy_loss) + (
        (1 - accuracy_weight) * representation_loss
    )

    # Backprop Loss
    loss.backward()
    
    y = y.cpu()
    p_hat = p_hat.cpu()
    with torch.no_grad():
        guess_cutoff = 0.5
        p_hat = p_hat.float()
        p_guess = (guess_cutoff * torch.ones(len(p_hat)) < p_hat).float()
        y = y.float()
    
    return loss.item(), p_guess.int().tolist(),y.int().tolist()

def get_tokens(rna_list,base_number_dict):
    # rna_list: [ [">AGCT-AA","AGCUAA"], [">AGCT-AA","AGCUAA"] ]
    for i in range(len(rna_list)):
        rna_list[i][1]=list(rna_list[i][1])
        for j in range(len(rna_list[i][1])):
            rna_list[i][1][j]=base_number_dict[rna_list[i][1][j]]
    return rna_list
def plot_eval_predictions(labels, predictions, path="figure"):
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
    plt.step(fpr, tpr, color="b", alpha=0.2, where="post")
    plt.fill_between(fpr, tpr, step="post", alpha=0.2, color="b")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Receiver Operating Characteristic (AUROC: {:.3})".format(auroc))
    plt.savefig(path + "auroc.svg")
    plt.close()
    return auroc,aupr
def train_model(args, output):
    # set the random seed
    seed=args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    # Create data sets
    train_df = pd.read_csv(args.train, sep="\t", header=None)
    train_df.columns = ["prot1", "prot2", "label"]
    train_p1 = train_df["prot1"]
    train_p2 = train_df["prot2"]
    train_y = torch.from_numpy(train_df["label"].values)
    train_dataset = PairedDataset(train_p1, train_p2, train_y)
    train_iterator = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_paired_sequences,
        shuffle=True
    )
    log(f"Loaded {len(train_p1)} training pairs", file=output)
    
    test_fi = args.valid
    test_df = pd.read_csv(test_fi, sep="\t", header=None)
    
    if args.checkpoint is None:

        # Create embedding model
        embedding_transform = EmbeddingTransform(
            args.input_dim, args.projection_dim, dropout=args.dropout_p,
            nhead=args.nhead,num_layers=args.num_layers
        )
        log("Initializing EmbeddingTransform model with:", file=output)
        log(f"\tprojection_dim: {args.projection_dim}", file=output)
        log(f"\tdropout_p: {args.dropout_p}", file=output)

        # Create contact model
        log("Initializing contact model with:", file=output)
        log(f"\tks: {args.ks}", file=output)
        contact_model = ContactCNN(args.ks,args.projection_dim)

        # Create the full model
        log("Initializing interaction model with:", file=output)
        log(f"\tdo_pool: {args.do_pool}", file=output)
        log(f"\tpool_width: {args.pool_width}", file=output)
        log(f"\tdo_w: {not args.no_w}", file=output)
        log(f"\tdo_sigmoid: {not args.no_sigmoid}", file=output)
        model = ModelInteraction(
            args,
            embedding_transform,
            contact_model,
            do_w=not args.no_w,
            pool_size=args.pool_width,
            do_pool=args.do_pool,
            do_sigmoid=not args.no_sigmoid,
            p0=args.p0
        )
        log(model, file=output)
    else:
        log(
            "Loading model from checkpoint {}".format(args.checkpoint),
            file=output,
        )
        model = torch.load(args.checkpoint)
    model.to(args.device)

    # Train the model    
    save_prefix = args.outdir
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    log(f'Using save prefix "{save_prefix}"', file=output)
    log(f"Training with Adam: lr={args.lr}, weight_decay={args.weight_decay}", file=output)
    log(f"\tnum_epochs: {args.num_epochs}", file=output)
    log(f"\tbatch_size: {args.batch_size}", file=output)
    log(f"\tinteraction weight: {args.interaction_weight}", file=output)
    log(f"\tcontact map weight: {1-args.interaction_weight}", file=output)

    batch_report_fmt="[{}/{}] training {:.1%}: Loss={:.6}, Accuracy={:.3%}"
    N = train_iterator.dataset.__len__()
    base_number_dict={"A":1,"G":2,"C":3,"U":4}
    
    standard_base=0
    number=1
    for epoch in range(args.num_epochs):
        model.train()
        n = 0
        loss_average = 0
        accuracy_average = 0
        # Train batches
        for (z0, z1, y) in train_iterator:
            rnas = list(set(z0).union(z1))
            rnas=[[i,i.replace('-','').replace('>','').replace("T", "U")] for i in rnas]
            rna_list=get_tokens(rnas,base_number_dict)
            for i in range(len(rna_list)):
                rna_list[i][1]=torch.LongTensor(rna_list[i][1]).to(args.device)
            embeddings={rna_list[i][0]:rna_list[i][1] for i in range(len(rna_list))}
            loss, p_guess, label = interaction_grad(
                args,
                model,
                z0,
                z1,
                y,
                embeddings,
                accuracy_weight=args.interaction_weight
            )
            optim.step()
            optim.zero_grad()
            #model.clip()
            
            loss_average=(loss_average*n+loss)/(n+len(z0))
            accuracy=accuracy_score(label,p_guess)
            accuracy_average = (accuracy_average*n+accuracy*len(z0))/(n+len(z0))
            n=n+len(z0)  
            
            # output the training log
            if n%2000==0:
                tokens = [epoch + 1,args.num_epochs,n / N,loss_average,accuracy_average]
                log(batch_report_fmt.format(*tokens), file=output)
            
            # Save the model
            if (save_prefix is not None) and (n%16000==0 or n==N) and (accuracy_average > 0.95):
                tokens = [epoch + 1,args.num_epochs,n / N,loss_average,accuracy_average]
                log(batch_report_fmt.format(*tokens), file=output)
                model.eval()
                outPath = args.outdir+'model' # make the temporary dir to save the model, note that if the modle is not a good model, the dir will be deleted!
                os.mkdir(outPath)
                outFile = open(outPath + "/predictions.tsv", "w+") # the predictions of the modle
                
                with torch.no_grad():
                    phats = []
                    labels = []
                    for _, (n0, n1, label) in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting pairs"):
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
                        pred = model.map_predict(p0, p1)[1].item()
                        phats.append(pred)
                        labels.append(label)
                        outFile.write(f"{n0}\t{n1}\t{label}\t{pred:.5}\n")
                phats = np.array(phats)
                labels = np.array(labels)
                auroc,aupr=plot_eval_predictions(labels, phats,outPath+'/')
                predicted_labels=phats 
                predicted_labels[predicted_labels>=0.5]=1
                predicted_labels[predicted_labels<0.5]=0
                tn, fp, fn, tp = confusion_matrix(labels, predicted_labels).ravel()
                accuracy = (tp+tn)/(tp+tn+fp+fn)
                sensitivity = tp/(tp+fn) 
                specificity = tn/(tn+fp)
                ppv = tp/(tp+fp)
                npv = tn/(tn+fn)
                f1score = 2*tp/(2*tp+fp+fn)                
                outFile.close()
                
                save_path = (outPath + "/model.sav")
                model.cpu()
                torch.save(model, save_path)
                model.to(args.device)
                model.train()
                
                with open(outPath + "/metrics.txt",'w') as f:
                    f.write("Report of model: accuracy: {:.4f},sensitivity: {:.4f},specificity: {:.4f},ppv: {:.4f},npv: {:.4f},F1 score: {:.4f}\n".format(accuracy,sensitivity,specificity,ppv,npv,f1score))
                
                standard=accuracy+sensitivity+specificity+ppv+npv+f1score
                
                if standard > standard_base:
                    standard_base=standard
                    log("Report of model: accuracy: {:.4f},sensitivity: {:.4f},specificity: {:.4f},ppv: {:.4f},npv: {:.4f},F1 score: {:.4f}".format(accuracy,sensitivity,specificity,ppv,npv,f1score),file=output)
                    if 'best_model' in os.listdir(args.outdir):
                        #shutil.rmtree(args.outdir+'best_model')
                        os.rename(args.outdir+'best_model',args.outdir+'best_model_history_'+str(number))
                        number=number+1
                    os.rename(outPath, args.outdir+'best_model')
                    print("update the beat model!")
                else:
                    for file in os.listdir(outPath):
                        os.remove(outPath+'/'+file)
                    os.rmdir(outPath)
                    #log("Report of the previous model {}: accuracy: {:.4f},precision: {:.4f},recall: {:.4f},F1 score: {:.4f},AUROC: {:.4f},AUPR: {:.4f}".format("bad model",accuracy,precision,recall,f1score,auroc,aupr),file=output)
                    print("accuracy: {:.4f},sensitivity: {:.4f},specificity: {:.4f},ppv: {:.4f},npv: {:.4f},F1 score: {:.4f}".format(accuracy,sensitivity,specificity,ppv,npv,f1score))
                    print("can not satisfy the conditions!")
                

def main(args):
    # Run training from arguments.
    output = args.outdir+'log.txt'
    log(f'Called as: {" ".join(sys.argv)}', file=output, print_also=True)
    # Set the device
    device = args.device
    if device == -1 :
        args.device='cpu'
        device = args.device
        log(
            "Using cpu",
            file=output,
            print_also=True,
        )
    elif device > -1 :
        log(
            f"Using CUDA device {device} - {torch.cuda.get_device_name(device)}",
            file=output,
            print_also=True,
        )
    else:
        log(
            "device must be an integer >= -1!",
            file=output,
            print_also=True,
        )
        sys.exit('device must be an integer >= -1!')
    train_model(args, output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    main(parser.parse_args())