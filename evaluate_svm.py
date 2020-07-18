from sklearn.metrics import roc_auc_score
import numpy as np
import random
from ge.classify import read_node_label, Classifier
from ge import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from karateclub import GraphWave
def test(labels,embedding_dict):
    embedding=np.array([(embedding_dict[str(i)]) for i in range(len(labels))])
    print(type(labels))
    print(type(embedding))
    sss = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    f1_micros = []
    f1_macros = []
        
  
#########################
    i = 1
    f1_micros = []
    f1_macros = []
    for split_train, split_test in sss.split(embedding, labels):
        model=SVC(gamma='auto')
        model.fit(embedding[split_train], labels[split_train])        
        predictions = model.predict(embedding[split_test])
        f1_micro = f1_score(labels[split_test], predictions, average="micro")
        f1_macro = f1_score(labels[split_test], predictions, average="macro")
        f1_micros.append(f1_micro)
        f1_macros.append(f1_macro)
        i += 1
        print(f1_macros, file = sample)
    print("No feature: ",np.mean(f1_micros), np.mean(f1_macros), file = sample)
    
    #########################33

    return 
def test_int(labels,embedding_dict):
    embedding=np.array([(embedding_dict[i]) for i in range(len(labels))])
    print(type(labels))
    print(type(embedding))
    sss = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    f1_micros = []
    f1_macros = []
        
  
#########################
    i = 1
    f1_micros = []
    f1_macros = []
    for split_train, split_test in sss.split(embedding, labels):
        model=SVC(gamma='auto')
        model.fit(embedding[split_train], labels[split_train])        
        predictions = model.predict(embedding[split_test])
        f1_micro = f1_score(labels[split_test], predictions, average="micro")
        f1_macro = f1_score(labels[split_test], predictions, average="macro")
        f1_micros.append(f1_micro)
        f1_macros.append(f1_macro)
        i += 1
        print(f1_macros, file = sample)
    print("No feature: ",np.mean(f1_micros), np.mean(f1_macros), file = sample)
    
    #########################33

    return 
if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    graph = nx.read_weighted_edgelist("data/brazil-airports.edgelist", delimiter=" ", nodetype=None,create_using=nx.Graph())
    graph_int = nx.read_weighted_edgelist("data/brazil-airports.edgelist", delimiter=" ", nodetype=int,create_using=nx.Graph())
    labels = pd.read_csv('data/labels-brazil-airports.txt', index_col=0, sep=" ")
    labels=labels.values

    nx.set_edge_attributes(graph, name="weight", values={edge: 1
        for edge, weight in nx.get_edge_attributes(graph, name="weight").items()})

    sample = open('a.out', 'w') 
    for our_size in [128,128,128,128]:
        print("size!!!",our_size, file = sample)
############################################################################################
        model = Node2Vec(graph.to_directed(), walk_length = 10, num_walks = 80,p = 1, q = 1, workers = 1)#init model
        model.train(embed_size=our_size,window_size = 5, iter = 3)# train model
        embeddings = model.get_embeddings()# get embedding vectors
        print("Node2Vec", file = sample)
        test(labels,embeddings)
        sample.flush()
############################################################################################
        model = Struc2Vec(graph.to_directed(), walk_length=10, num_walks=80,workers=8, verbose=40 )
        model.train(embed_size=our_size,window_size = 5, iter = 3)
        embeddings = model.get_embeddings()
        print("Struc2Vec", file = sample)
        test(labels,embeddings)
        sample.flush()

###################################################
        model=GraphWave(mechanism = "exact")
        model.fit(graph_int)
        embeddings=model.get_embedding()
        print("GraphWave", file = sample)
        test_int(labels,embeddings) 
        sample.flush()
