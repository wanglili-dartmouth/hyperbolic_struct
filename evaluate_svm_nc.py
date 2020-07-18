import os
import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score


from heat.utils import load_data, hyperboloid_to_klein, load_embedding, poincare_ball_to_hyperboloid, hyperboloid_to_poincare_ball

import functools
import fcntl
import argparse
import htools
import hsvm

def evaluate_kfold_label_classification(embedding, 
    labels, 
    k=10):
    params = {
        'C': 100,
        'epochs': 1000,
        'lr': 0.001,
        'batch_size': 16,
        'pretrained': True,
    }
    assert len(labels.shape) == 2
    
    

    if labels.shape[1] == 1:
        print ("single label clasification")
        labels = labels.flatten()
        sss = StratifiedKFold(n_splits=k, 
            shuffle=True, 
            random_state=0)

            
    f1_micros = []
    f1_macros = []
    print(np.sum(labels==0))
    print(np.sum(labels==1))
    print(np.sum(labels==2))
    print(np.sum(labels==3))
    ii = 1
    #one vs all
    labels0=np.zeros(len(labels))
    labels1=np.zeros(len(labels))
    labels2=np.zeros(len(labels))
    labels3=np.zeros(len(labels))
    
    labels0[labels!=0]=-1
    labels0[labels==0]=1
    print(np.sum(labels0==-1))
    print(np.sum(labels0==1))
    labels1[labels!=1]=-1
    labels1[labels==1]=1
    print(np.sum(labels1==-1))
    print(np.sum(labels1==1))
    labels2[labels!=2]=-1
    labels2[labels==2]=1
    print(np.sum(labels2==-1))
    print(np.sum(labels2==1))
    labels3[labels!=3]=-1
    labels3[labels==3]=1
    print(np.sum(labels3==-1))
    print(np.sum(labels3==1))
    
    for split_train, split_test in sss.split(embedding, labels):
        model = hsvm.LinearHSVM(**params)
        model.fit(embedding[split_train], labels0[split_train])        
        p0 = model.decision_function(embedding[split_test])
        
        model = hsvm.LinearHSVM(**params)
        model.fit(embedding[split_train], labels1[split_train])        
        p1 = model.decision_function(embedding[split_test])
        
        model = hsvm.LinearHSVM(**params)
        model.fit(embedding[split_train], labels2[split_train])        
        p2 = model.decision_function(embedding[split_test])
        
        model = hsvm.LinearHSVM(**params)
        model.fit(embedding[split_train], labels3[split_train])        
        p3 = model.decision_function(embedding[split_test])
        
        predictions=np.zeros(len(p0))
        for i in range(len(p0)):
            if(p0[i]>p1[i] and p0[i]>p2[i] and p0[i]>p3[i]):
                predictions[i]=0
            if(p1[i]>p0[i] and p1[i]>p2[i] and p1[i]>p3[i]):
                predictions[i]=1
            if(p2[i]>p1[i] and p2[i]>p0[i] and p2[i]>p3[i]):
                predictions[i]=2
            if(p3[i]>p1[i] and p3[i]>p0[i] and p3[i]>p2[i]):
                predictions[i]=3
        print(predictions)
        f1_micro = f1_score(labels[split_test], predictions, average="micro")
        f1_macro = f1_score(labels[split_test], predictions, average="macro")
        f1_micros.append(f1_micro)
        f1_macros.append(f1_macro)
        print ("Done {}/{} folds".format(ii, k))
        print(f1_macro)
        ii += 1
    return np.mean(f1_micros), np.mean(f1_macros)


def touch(path):
    with open(path, 'a'):
        os.utime(path, None)

def read_edgelist(fn):
    edges = []
    with open(fn, "r") as f:
        for line in (l.rstrip() for l in f.readlines()):
            edge = tuple(int(i) for i in line.split("\t"))
            edges.append(edge)
    return edges

def lock_method(lock_filename):
    ''' Use an OS lock such that a method can only be called once at a time. '''

    def decorator(func):

        @functools.wraps(func)
        def lock_and_run_method(*args, **kwargs):

            # Hold program if it is already running 
            # Snippet based on
            # http://linux.byexamples.com/archives/494/how-can-i-avoid-running-a-python-script-multiple-times-implement-file-locking/
            fp = open(lock_filename, 'r+')
            done = False
            while not done:
                try:
                    fcntl.lockf(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    done = True
                except IOError:
                    pass
            return func(*args, **kwargs)

        return lock_and_run_method

    return decorator 

def threadsafe_fn(lock_filename, fn, *args, **kwargs ):
    lock_method(lock_filename)(fn)(*args, **kwargs)

def save_test_results(filename, seed, data, ):
    d = pd.DataFrame(index=[seed], data=data)
    if os.path.exists(filename):
        test_df = pd.read_csv(filename, sep=",", index_col=0)
        test_df = d.combine_first(test_df)
    else:
        test_df = d
    test_df.to_csv(filename, sep=",")

def threadsafe_save_test_results(lock_filename, filename, seed, data):
    threadsafe_fn(lock_filename, save_test_results, filename=filename, seed=seed, data=data)

def parse_args():

    parser = argparse.ArgumentParser(description='Load Hyperboloid Embeddings and evaluate node classification')
    
    parser.add_argument("--edgelist", dest="edgelist", type=str, 
        help="edgelist to load.")
    parser.add_argument("--features", dest="features", type=str, 
        help="features to load.")
    parser.add_argument("--labels", dest="labels", type=str, 
        help="path to labels")

    parser.add_argument('--directed', action="store_true", help='flag to train on directed graph')

    parser.add_argument("--embedding", dest="embedding_filename",  
        help="path of embedding to load.")

    parser.add_argument("--test-results-dir", dest="test_results_dir",  
        help="path to save results.")

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--dist_fn", dest="dist_fn", type=str,
    choices=["poincare", "hyperboloid", "euclidean"])

    return parser.parse_args()

def main():
    random.seed(0)
    np.random.seed(0)
    args = parse_args()
    node_labels = pd.read_csv(args.labels, index_col=0, sep=" ")
    node_labels=node_labels.values
    print ("Loaded dataset")
    dist_fn = args.dist_fn

    sep = ","
    header = "infer"
    if dist_fn == "euclidean":
        sep = " "
        header = None

    embedding_df = pd.read_csv(args.embedding_filename,
        sep=sep, header=header, index_col=0)
    embedding_df = embedding_df.reindex(sorted(embedding_df.index))
    embedding = embedding_df.values

    # project to a space with straight euclidean lines
    if dist_fn == "poincare":
        embedding = poincare_ball_to_hyperboloid(embedding)



    k_fold_f1_micro, k_fold_f1_macro = \
        evaluate_kfold_label_classification(embedding,node_labels, k=10)

    test_results = {}
    

    test_results.update({"10-fold-f1_micro": k_fold_f1_micro, "10-fold-f1-macro": k_fold_f1_macro})

    test_results_dir = args.test_results_dir
    if not os.path.exists(test_results_dir):
        os.makedirs(test_results_dir)
    test_results_filename = os.path.join(test_results_dir, "test_results.csv")
    test_results_lock_filename = os.path.join(test_results_dir, "test_results.lock")

    touch (test_results_lock_filename)

    print ("saving test results to {}".format(test_results_filename))
    threadsafe_save_test_results(test_results_lock_filename, test_results_filename, args.seed, data=test_results )
    
if __name__ == "__main__":
    main()