"""
    Edited by Rosa
    Source python3 update: https://github.com/PPerdomoQ/rare-disease-explainer/blob/main/edge2vec3.py
    Source: https://github.com/RoyZhengGao/edge2vec/blob/master/edge2vec.py
"""

import random
import numpy as np   
from gensim.models import Word2Vec

def simulate_walks_2(G, num_walks, walk_length, matrix, p, q, seed=None):
    """
        Generate random walk paths constrained by transition matrix for each node in given graph
        :param G: digraph generated with networkx
        :param num_walks: number of walks per node 
        :param walk_length: allowed length of walks
        :param matrix: edge type transition matrix
        :param p: the greater p, the lower the probability of returning to previous node
        :param q: the greater q, the lower the probability of moving to another node than the previous and current node
        :return list of paths containing nodes visited during these paths
    """
    random.seed(seed)
    
    walks = []
    nodes = list(G.nodes())
    
    print('Walk iteration:')
    for walk_iter in range(num_walks):
        print(str(walk_iter+1), '/', str(num_walks))
        random.shuffle(nodes) 
        for node in nodes:
            walks.append(edge2vec_walk_2(G, walk_length, node, matrix, p, q, seed))  
    return walks

def edge2vec_walk_2(G, walk_length, start_node, matrix, p, q, seed=None):
    """
        Return a random walk path constrained by edge type transition matrix and parameters p and q
        :param G: digraph generated with networkx
        :param walk_length: allowed length of walks
        :param start_node: first node visited
        :param matrix: edge type transition matrix
        :param is_directed: specifies whether graph has directed edges
        :param p: the greater p, the lower the probability of returning to previous node
        :param q: the greater q, the lower the probability of moving to another node than the previous and current node
        :return list of nodes encountered during the walk
    """
    random.seed(seed)
    np.random.seed(seed)
    
    walk = [start_node]  
    while len(walk) < walk_length:# here we may need to consider some dead end issues
        cur = walk[-1]
        cur_nbrs =sorted(G.neighbors(cur))
        random.shuffle(cur_nbrs)
        if len(cur_nbrs) > 0:
            if len(walk) == 1:
                rand = int(np.random.rand()*len(cur_nbrs))
                next =  cur_nbrs[rand]
                walk.append(next) 
            else:
                prev = walk[-2]
                pre_edge_type = G[prev][cur]['type']
                distance_sum = 0
                for neighbor in cur_nbrs:
                    neighbor_link = G[cur][neighbor] 
                    neighbor_link_type = neighbor_link['type']
                    neighbor_link_weight = neighbor_link['weight']
                    trans_weight = matrix[pre_edge_type-1][neighbor_link_type-1]
                    
                    if G.has_edge(neighbor,prev) or G.has_edge(prev,neighbor): #undirected graph
                        distance_sum += trans_weight*neighbor_link_weight/p #+1 normalization
                    elif neighbor == prev: #decide whether it can random walk back
                        distance_sum += trans_weight*neighbor_link_weight
                    else:
                        distance_sum += trans_weight*neighbor_link_weight/q

                '''
                pick up the next step link
                ''' 

                rand = np.random.rand() * distance_sum
                threshold = 0 
                for neighbor in cur_nbrs:
                    neighbor_link = G[cur][neighbor] 
                    neighbor_link_type = neighbor_link['type']
                    neighbor_link_weight = neighbor_link['weight']
                    trans_weight = matrix[pre_edge_type-1][neighbor_link_type-1]
                    
                    if G.has_edge(neighbor,prev)or G.has_edge(prev,neighbor):#undirected graph
                        
                        threshold += trans_weight*neighbor_link_weight/p 
                        if threshold >= rand:
                            next = neighbor
                            break;
                    elif neighbor == prev:
                        threshold += trans_weight*neighbor_link_weight
                        if threshold >= rand:
                            next = neighbor
                            break;        
                    else:
                        threshold += trans_weight*neighbor_link_weight/q
                        if threshold >= rand:
                            next = neighbor
                            break;
		 
                walk.append(next) 
        else:
            break #if only has 1 neighbour 
 
    return walk  