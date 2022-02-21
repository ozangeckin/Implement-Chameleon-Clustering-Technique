# 1801042103
# Ozan GECKIN
import itertools
import pandas as pd
import numpy as np
import networkx as nx
from visualization import plot2d_graph
import metis

def chameleonCluster(dataFrame, k,k_neighbor_number, subCluster):
    graph = k_neighbor_graph(dataFrame, k_neighbor_number)
    plot2d_graph(graph)
    graph = partGraph(graph, subCluster, dataFrame)
    iterm = enumerate(range(subCluster-k))
    for i in iterm:
        merge(graph, dataFrame) 
    copyDataFrame = dataFrame.copy()
    cl = list(pd.DataFrame(dataFrame['cl'].value_counts()).index)
    j=1
    for i in cl:
        copyDataFrame.loc[dataFrame['cl']==i,'cl']=j
        j = j + 1
    return copyDataFrame

def euclidean(xiyi, xjyj):
    return np.linalg.norm(np.array(xiyi) - np.array(xjyj))

def k_neighbor_graph(dataFrame, k):
    points =[]
    for p in dataFrame.itertuples():
        points.append(p[1:])
    grph = nx.Graph()
    for i in range(0, len(points)):
        grph.add_node(i)
    print("Create k neighbor number graph ")
    iterp = enumerate(points)
    for i, p in iterp:
        distances = list(map(lambda temp: euclidean(p, temp), points))
        closests = np.argsort(distances)[1:k+1]  
        for j in closests:
            grph.add_edge(i, j, weight=1.0 / distances[j], similarity=int(1.0 / distances[j] * 1e4))
        grph.nodes[i]['plc'] = p
    grph.graph['edge_attr'] = 'similarity'
    return grph

def helperPartGraph(graph, k, dataFrame=None):
    edgecuts, parts = metis.part_graph(graph, 2, objtype='cut', ufactor=250)
    for i, p in enumerate(graph.nodes()):
        graph.nodes[p]['cl'] = parts[i]
    if dataFrame is not None:
        dataFrame['cl'] = nx.get_node_attributes(graph, 'cl').values()
    return graph

def partGraph(graph, k, dataFrame):
   print("Stat Clustering...")
   clstrs = 0
   for i, p in enumerate(graph.nodes()):
       graph.nodes[p]['cl'] = 0
   connect = {}
   connect[0] = len(graph.nodes())

   while clstrs < k - 1:
        mc = -1
        mcnt = 0
        for ke, v in connect.items():
            if v > mcnt:
                mcnt = v
                mc = ke
        s_nodes=[]
      
        for n in graph.nodes:
            if graph.nodes[n]['cl']==mc:
                s_nodes.append(n)
        
        s_graph = graph.subgraph(s_nodes)
        edgecuts, parts = metis.part_graph(s_graph, 2, objtype='cut', ufactor=250)
        newPartCnt = 0
        for i, p in enumerate(s_graph.nodes()):
            if parts[i] == 1:
                graph.nodes[p]['cl'] = clstrs + 1
                newPartCnt = newPartCnt + 1
        connect[mc] = connect[mc] - newPartCnt
        connect[clstrs + 1] = newPartCnt
        clstrs = clstrs + 1

   edgecuts, parts = metis.part_graph(graph, k)
   if dataFrame is not None:
       dataFrame['cl'] = nx.get_node_attributes(graph, 'cl').values()
   return graph

def relativeInterconnectivity(graph, clusterX, clusterY):
    edges = connectingEdges((clusterX, clusterY), graph)
    edgeCut = np.sum(getWeights(graph, edges))
    edgeCutcx = np.sum(bisectionWeights(graph,clusterX))
    edgeCutcy = np.sum(bisectionWeights(graph,clusterY))
    return edgeCut / ((edgeCutcx + edgeCutcy) / 2.0)

def internalCloseness(graph, cluster):
    cluster = graph.subgraph(cluster)
    edges = cluster.edges()
    weights = getWeights(cluster, edges)
    return np.sum(weights)

def relative_closeness(graph, clusterX, clusterY):
    edges = connectingEdges((clusterX, clusterY), graph)
    if not edges:
        return 0.0
    else:
        SEC = np.mean(getWeights(graph, edges))
    Ci =internalCloseness(graph, clusterX)
    Cj =internalCloseness(graph, clusterY)
    SECci= np.mean(bisectionWeights(graph, clusterX))
    SECcj =np.mean(bisectionWeights(graph, clusterY))
    return SEC / ((Ci / (Ci + Cj) * SECci) + (Cj / (Ci + Cj) * SECcj))

def merge(graph, dataFrame):
    clusters = np.unique(dataFrame['cl'])
    max_score = 0
    ci, cj = -1, -1
    for combination in itertools.combinations(clusters, 2):
        i, j = combination
        if i != j:
            gi = getCluster(graph, [i])
            gj = getCluster(graph, [j])
            edges = connectingEdges((gi, gj), graph)
            if not edges:
                continue
            ms =relativeInterconnectivity(graph, gi, gj) * (relative_closeness(graph, gi, gj))
            if ms > max_score:
                max_score = ms
                ci = i
                cj = j
    if max_score > 0:
        dataFrame.loc[dataFrame['cl'] == cj, 'cl'] = ci
        for i, p in enumerate(graph.nodes()):
            if graph.nodes[p]['cl'] == cj:
                graph.nodes[p]['cl'] = ci
    return max_score > 0

def getCluster(graph, clusters):
    nodes=[]
    for n in graph.nodes:
        if graph.nodes[n]['cl'] in clusters:
            nodes.append(n)
    return nodes
            
def connectingEdges(partitions, graph):
    cut_set = []
    for a in partitions[0]:
        for b in partitions[1]:
            if a in graph:
                if b in graph[a]:
                    cut_set.append((a, b))
    return cut_set

def getWeights(graph, edges):
    weights=[]
    for edge in edges:
        weights.append(graph[edge[0]][edge[1]]['weight'])
    return weights

def bisectionWeights(graph, cluster):
    cluster = graph.subgraph(cluster)
    clustert = cluster.copy()
    clustert = helperPartGraph(clustert, 2)
    partitions = getCluster(clustert, [0]), getCluster(clustert, [1])
    edges = connectingEdges(partitions,clustert)
    weights = getWeights(cluster, edges)
    return weights
