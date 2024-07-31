import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(edges, source_colname, target_colname, file_name):
    """
        Draw a graph from a pandas dataframe given the column names of the source and target of the edges.
        :param source_colname: name of column that includes source of edge
        :param target_colname: name of column that includes target of edge
        :param file_name: name of file in which image of graph is stored (in `output` folder)
    """
    G = nx.from_pandas_edgelist(edges, source=source_colname, target=target_colname)
    
    nx.draw(G, with_labels=True, node_size=800, font_size=6)
    
    plt.savefig(file_name)
    plt.clf()