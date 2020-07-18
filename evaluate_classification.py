from sklearn.metrics import roc_auc_score
import numpy as np
import random
from ge.classify import read_node_label, Classifier
from ge import *
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
#from karateclub import GraphWave
def split_edges(edges, graph,pivot_time):

    nodes_in_training=set()
    test_edges=[]
    train_edges=[]
    train_non_edges = []
    for edge in edges:
        if(edge[2]['time']>pivot_time):
            test_edges.append((edge[0],edge[1]))
        else:
            train_edges.append((edge[0],edge[1]))
            nodes_in_training.add(edge[0])
            nodes_in_training.add(edge[1])
    print("before ",len(test_edges))
    test_edges=[edge for edge in test_edges if (edge[0] in nodes_in_training and edge[1] in nodes_in_training)]
    print("after ",len(test_edges))
    random.seed(0)
    if(len(graph.nodes())<20000):
        non_edges = list(nx.non_edges(graph))
        
        used_non=random.sample(non_edges,len(test_edges)+len(train_edges))
        test_non_edges = used_non[:len(test_edges)]
        train_non_edges = used_non[len(test_edges):]
        
       
    else:
        print("Using another method")
        test_non_edges=[]
        train_non_edges=[]
        nodes = graph.nodes()
        N=len(test_edges)
        with tqdm(total=N, desc='False Edges', unit='false_edge') as pbar:
            while len(test_non_edges)<N:
                random_edge = sorted(np.random.choice(nodes, 2, replace=False))
                if random_edge[1] not in graph[random_edge[0]] and random_edge not in test_non_edges:
                    test_non_edges.append(random_edge)
                    pbar.update(1)
                    
        N=len(train_edges)
        with tqdm(total=N, desc='False Edges', unit='false_edge') as pbar:
            while len(train_non_edges)<N:
                random_edge = sorted(np.random.choice(nodes, 2, replace=False))
                if random_edge[1] not in graph[random_edge[0]] and random_edge not in train_non_edges and random_edge not in test_non_edges:
                    train_non_edges.append(random_edge)
                    pbar.update(1)
    print("check split edges:")
    print(len(train_non_edges))
    print(len(train_edges))
    print(len(test_non_edges))
    print(len(test_edges))



    return (train_edges, train_non_edges), (test_edges, test_non_edges)
def multigraph2graph(multi_graph_nx):
    '''
    convert a multi_graph into a graph, where a multi edge becomes a singe weighted edge
    Args:
        multi_graph_nx: networkx - the given multi_graph

    Returns:
        networkx graph
    '''
    if type(multi_graph_nx) == nx.Graph or type(multi_graph_nx) == nx.DiGraph:
        print("No worries, No change")
        return multi_graph_nx
    graph_nx = nx.DiGraph() if multi_graph_nx.is_directed() else nx.Graph()

    if len(multi_graph_nx.nodes()) == 0:
        return graph_nx

    # add edges + attributes
    for u, v, data in multi_graph_nx.edges(data=True):
        data['weight'] = data['weight'] if 'weight' in data else 1.0

        if graph_nx.has_edge(u, v):
            graph_nx[u][v]['weight'] += data['weight']
        else:
            graph_nx.add_edge(u, v, **data)

    # add node attributes
    for node, attr in multi_graph_nx.nodes(data=True):
        if node not in graph_nx:
            continue
        graph_nx.nodes[node].update(attr)

    return graph_nx
def get_graph_T(graph_nx, min_time=-np.inf, max_time=np.inf, return_df=False):
    '''
    Given a graph with a time attribute for each edge, return the subgraph with only edges between an interval.
    Args:
        graph_nx: networkx - the given graph
        min_time: int - the minimum time step that is wanted. Default value -np.inf
        max_time: int - the maximum time step that is wanted. Default value np.inf
        return_df: bool - if True, return a DataFrame of the edges and attributes,
                          else, a networkx object

    Returns:
        sub_graph_nx: networkx - subgraph with only edges between min_time and max_time
    '''
    relevant_edges = []
    attr_keys = []

    if len(graph_nx.nodes()) == 0:
        return graph_nx

    for u, v, attr in graph_nx.edges(data=True):
        if min_time < attr['time'] and attr['time'] <= max_time:
            relevant_edges.append((u, v, *attr.values()))

            if attr_keys != [] and attr_keys != attr.keys():
                raise Exception('attribute keys in \'get_graph_T\' are different')
            attr_keys = attr.keys()

    graph_df = pd.DataFrame(relevant_edges, columns=['from', 'to', *attr_keys])

    if return_df:
        node2label = nx.get_node_attributes(graph_nx, 'label')
        if len(node2label) > 0:
            graph_df['from_class'] = graph_df['from'].map(lambda node: node2label[node])
            graph_df['to_class'] = graph_df['to'].map(lambda node: node2label[node])
        return graph_df
    else:
        sub_graph_nx = nx.from_pandas_edgelist(graph_df, 'from', 'to', list(attr_keys), create_using=type(graph_nx)())

        # add node attributes
        for node, attr in graph_nx.nodes(data=True):
            if node not in sub_graph_nx:
                continue
            sub_graph_nx.nodes[node].update(attr)

        return sub_graph_nx


def get_graph_times(graph_nx):
    '''
    Return all times in the graph edges attributes
    Args:
        graph_nx: networkx - the given graph

    Returns:
        list - ordered list of all times in the graph
    '''
    return np.sort(np.unique(list(nx.get_edge_attributes(graph_nx, 'time').values())))

def get_pivot_time(graph_nx, wanted_ratio=0.2, min_ratio=0.1):
    '''
    Given a graph with 'time' attribute for each edge, calculate the pivot time that gives
    a wanted ratio to the train and test edges
    Args:
        graph_nx: networkx - Graph
        wanted_ratio: float - number between 0 and 1 representing |test|/(|train|+|test|)
        min_ratio: float - number between 0 and 1 representing the minimum value of the expected ratio

    Returns:
        pivot_time: int - the time step that creates such deviation
    '''
    times = get_graph_times(graph_nx)
    if wanted_ratio == 0:
        return times[-1]

    time2dist_from_ratio = {}
    for time in times[int(len(times) / 3):]:
        train_graph_nx = multigraph2graph(get_graph_T(graph_nx, max_time=time))
        num_edges_train = len(train_graph_nx.edges())

        test_graph_nx = get_graph_T(graph_nx, min_time=time)
        print(time," before :",len(test_graph_nx.edges()))
        test_graph_nx.remove_nodes_from([node for node in test_graph_nx if node not in train_graph_nx])
        test_graph_nx = multigraph2graph(test_graph_nx)
        num_edges_test = len(test_graph_nx.edges())
        print(time," after :",len(test_graph_nx.edges()))
        
        current_ratio = num_edges_test / (num_edges_train + num_edges_test)
        print(time,"   ",current_ratio)
        if current_ratio <= min_ratio:
            continue

        time2dist_from_ratio[time] = np.abs(wanted_ratio - current_ratio)

    pivot_time = min(time2dist_from_ratio, key=time2dist_from_ratio.get)

    print(f'pivot time {pivot_time}, is close to the wanted ratio by {round(time2dist_from_ratio[pivot_time], 3)}')

    return pivot_time
def test(embeddings):
    train_x=[]
    train_y=[]
    test_x=[]
    test_y=[]
    length=0
    with open('data/barbell_small/train.tsv') as f:
        for line in f:
            a,b=(int(x) for x in line.split()) 
            train_x.append(embeddings[str(a)])
            train_y.append(b)
    with open('data/barbell_small/test.tsv') as f:
        for line in f:
            a,b=(int(x) for x in line.split()) 
            test_x.append(embeddings[str(a)])
            test_y.append(b)
    
    clf = LogisticRegression(random_state=0).fit(train_x, train_y)
    y_pred=clf.predict(test_x)
    acc=accuracy_score(test_y,y_pred)
    f1_micro = f1_score(test_y,y_pred, average="micro")
    f1_macro = f1_score(test_y,y_pred, average="macro")
    print("ACC:",acc, file = sample)
    print("f1_micro:",f1_micro, file = sample)
    print("f1_macro:",f1_macro, file = sample)
def test_int(embeddings):
    train_x=[]
    train_y=[]
    test_x=[]
    test_y=[]
    length=0
    with open('data/barbell_small/train.tsv') as f:
        for line in f:
            a,b=(int(x) for x in line.split()) 
            train_x.append(embeddings[a])
            train_y.append(b)
    with open('data/barbell_small/test.tsv') as f:
        for line in f:
            a,b=(int(x) for x in line.split()) 
            test_x.append(embeddings[a])
            test_y.append(b)
    
    clf = LogisticRegression(random_state=0).fit(train_x, train_y)
    y_pred=clf.predict(test_x)
    acc=accuracy_score(test_y,y_pred)
    f1_micro = f1_score(test_y,y_pred, average="micro")
    f1_macro = f1_score(test_y,y_pred, average="macro")
    print("ACC:",acc, file = sample)
    print("f1_micro:",f1_micro, file = sample)
    print("f1_macro:",f1_macro, file = sample)
def plot():
    X=[]
    Y=[]
    Label=[]
    for i in range(1,29):
        X.append(embeddings[str(i)][0])
        Y.append(embeddings[str(i)][1])
        if(i in set([1,2,3,4,6,7,8,9])):
            Label.append('Class 1')
        elif(i in set([15,16,17,18,20,21,22,23])):
            Label.append('Class 2')
        else:
            Label.append('Bridge')
    X.append(embeddings[str(0)][0])
    Y.append(embeddings[str(0)][1])
    Label.append('Bridge')
    d = {'X': X, 'Y': Y,'Label':Label}

    wide_df= pd.DataFrame(d)
    ax = sns.scatterplot(x='X', y='Y', hue="Label", s=100, style="Label",data=wide_df)
    return ax
    
def plot_int():
    X=[]
    Y=[]
    Label=[]
    for i in range(1,29):
        X.append(embeddings[i][0])
        Y.append(embeddings[i][1])
        if(i in set([1,2,3,6,7,8])):
            Label.append('Class 1')
            print(X[i-1])
            print(Y[i-1])
        elif(i in set([15,16,17,18,20,21,22,23])):
            Label.append('Class 2')
            print(X[i-1])
            print(Y[i-1])
        else:
            Label.append('Bridge')
    #X.append(embeddings[0][0])
    #Y.append(embeddings[0][1])
    #Label.append('Bridge')
    d = {'X': X, 'Y': Y,'Label':Label}

    wide_df= pd.DataFrame(d)
    ax = sns.scatterplot(x='X', y='Y', hue="Label", s=100, style="Label",data=wide_df)
    return ax
if __name__ == "__main__":
    graph = nx.read_weighted_edgelist("data/barbell_small/edgelist.tsv", delimiter=" ", nodetype=None,create_using=nx.Graph())
    nx.set_edge_attributes(graph, name="time", values={edge: abs(weight) 
        for edge, weight in nx.get_edge_attributes(graph, name="weight").items()})
    nx.set_edge_attributes(graph, name="weight", values={edge: 1
        for edge, weight in nx.get_edge_attributes(graph, name="weight").items()})
    
    
    graph_int = nx.read_weighted_edgelist("data/barbell_small/edgelist.tsv", delimiter=" ", nodetype=int,create_using=nx.Graph())
    nx.set_edge_attributes(graph_int, name="weight", values={edge: 1
        for edge, weight in nx.get_edge_attributes(graph_int, name="weight").items()})
    sample = open('a.out', 'w') 
    
    for history_length in [10]:
        
        
        ########################################################
        model = Struc2Vec(graph.to_directed(), walk_length=10, num_walks=80,opt3_num_layers=1,workers=8, verbose=40 )
        walks=model.return_walk_list()
        print(walks)
        
       