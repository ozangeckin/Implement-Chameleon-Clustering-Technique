# 1801042103
# Ozan GECKIN
import networkx as nx
import matplotlib.pyplot as plt

def plot2d_graph(graph):
    plc = nx.get_node_attributes(graph, 'plc')
    color_map=[]
    color_map.append('black')
    if color_map:  
        nx.draw(graph, plc, node_color=color_map, node_size=0.20)
    else:
        nx.draw(graph, plc, node_size=0.35)
    plt.show(block=False)


def plot2d_data(df):
    df.plot(kind='scatter', c=df['cl'], cmap='gist_rainbow', x=0, y=1)
    plt.show(block=False)
