"""
    Edited by Rosa
    Source python3 update: https://github.com/PPerdomoQ/rare-disease-explainer/blob/main/transition3.py
    Source original (python2) version: https://github.com/RoyZhengGao/edge2vec/blob/master/transition.py
"""

import networkx as nx
import random
import numpy as np   
import math
from scipy import stats
from scipy import spatial
 
def initialize_edge_type_matrix(type_num):
    """
        Initialize an edge type transition matrix with equal values.
        :param type_num: number of edge types
        :return list of lists representing matrix
    """
    initialized_val = 1.0/(type_num*type_num)
    matrix = [ [ initialized_val for _ in range(type_num) ] for _ in range(type_num) ]
    return matrix

def simulate_walks_1(G: nx.DiGraph, num_walks, walk_length, matrix, is_directed, p, q, seed=None):
    """
        Generate random walk paths that are constrained by edge type transition matrix.
        :param G: digraph generated with networkx
        :param num_walks: number of walks per node (of a maximum of 1000 nodes)
        :param walk_length: allowed length of walks
        :param matrix: edge type transition matrix
        :param is_directed: specifies whether graph has directed edges
        :param p: the greater p, the lower the probability of returning to previous node
        :param q: the greater q, the lower the probability of moving to another node than the previous and current node
        :return list of paths containing edge types encountered during these paths
    """
    random.seed(seed)
        
    walks = []
    links = list(G.edges(data = True))

    print('Walk iteration:')
    for walk_iter in range(num_walks):
        print(str(walk_iter+1), '/', str(num_walks))
        random.shuffle(links)
        count = 1000
        for link in links:
            walks.append(edge2vec_walk(G, walk_length, link, matrix, is_directed, p, q, seed)) 
            count = count - 1
            if count == 0 and len(links)>1000:  # control the pairwise list length
                break
    return walks

def edge2vec_walk(G, walk_length, start_link, matrix, is_directed, p, q, seed=None): 
    """
        Return a random walk path constrained by edge type transition matrix and parameters p and q
        :param G: digraph generated with networkx
        :param walk_length: allowed length of walks
        :param start_link: first edge traversed
        :param matrix: edge type transition matrix
        :param is_directed: specifies whether graph has directed edges
        :param p: the greater p, the lower the probability of returning to previous node
        :param q: the greater q, the lower the probability of moving to another node than the previous and current node
        :return list of edge types encountered during the walk
    """
    random.seed(seed)
    np.random.seed(seed)
    
    walk = [start_link] 
    result = [str(start_link[2]['type'])]
    
    while len(walk) < walk_length:# here we may need to consider some dead end issues
        cur = walk[-1]
        start_node = cur[0]
        end_node = cur[1]
        cur_edge_type = cur[2]['type']

        '''
        find the direction of link to go. If a node degree is 1, it means if go that direction, there is no other links to go further
        if the link are the only link for both nodes, the link will have no neighbours (need to have teleportation later)
        '''
        '''
        consider the hub nodes and reduce the hub influence
        '''
        if is_directed: # directed graph has random walk direction already
            direction_node = end_node
            left_node = start_node
        else: # for undirected graph, first consider the random walk direction by choosing the start node
            start_direction = 1.0/G.degree(start_node)
            end_direction = 1.0/G.degree(end_node)
            prob = start_direction/(start_direction+end_direction)

            rand = np.random.rand() 

            if prob >= rand:
                direction_node = start_node
                left_node = end_node
            else:
                direction_node = end_node
                left_node = start_node
                
        '''
        here to choose which link goes to. There are three conditions for the link based on node distance. 0,1,2
        '''
        
        neighbors = G.neighbors(direction_node) 
        
        '''
        calculate sum of distance, with +1 normalization
        '''
        
        distance_sum = 0
        for neighbor in neighbors:
            neighbor_link = G[direction_node][neighbor] # get candidate link's type
            neighbor_link_type = neighbor_link['type']
            neighbor_link_weight = neighbor_link['weight']
            trans_weight = matrix[cur_edge_type-1][neighbor_link_type-1]
            if G.has_edge(neighbor,left_node) or G.has_edge(left_node,neighbor): 
                distance_sum += trans_weight*neighbor_link_weight/p  
            elif neighbor == left_node: # decide whether it can random walk back
                distance_sum += trans_weight*neighbor_link_weight
            else:
                distance_sum += trans_weight*neighbor_link_weight/q

        '''
        pick up the next step link
        '''
        rand = np.random.rand() * distance_sum
        threshold = 0
        neighbors2 = G.neighbors(direction_node) 
        for neighbor in neighbors2:
            neighbor_link = G[direction_node][neighbor] # get candidate link's type
            neighbor_link_type = neighbor_link['type']
            neighbor_link_weight = neighbor_link['weight']
            trans_weight = matrix[cur_edge_type-1][neighbor_link_type-1]
            if G.has_edge(neighbor,left_node) or G.has_edge(left_node,neighbor): 
                threshold += trans_weight*neighbor_link_weight/p
                if threshold >= rand:
                    next_link_end_node = neighbor
                    break;
            elif neighbor == left_node:
                threshold += trans_weight*neighbor_link_weight
                if threshold >= rand:
                    next_link_end_node = neighbor
                    break;
            else:
                threshold += trans_weight*neighbor_link_weight/q
                if threshold >= rand:
                    next_link_end_node = neighbor
                    break;

        if distance_sum > 0: # the direction_node has next_link_end_node
            next_link = G[direction_node][next_link_end_node]
            
            next_link_tuple = tuple()
            next_link_tuple += (direction_node,)
            next_link_tuple += (next_link_end_node,)
            next_link_tuple += (next_link,)
            
            walk.append(next_link_tuple)
            result.append(str(next_link_tuple[2]['type']))
        else:
            break

    return result  


def update_trans_matrix(walks, type_size, evaluation_metric):
    """
        This is the E-step of the EM framework, during which the edge type transition matrix is updated using an evaluation metric
        :param walks: list of edge types encountered during the walks
        :param type_size: number of edge types
        :param evaluation_metric: allowed values are 1 (wilcoxon test), 2 (entroy test), 3 (spearmanr test), 4 (pearsonr test)
        :return updated matrix
    """
    '''
    E step, update transition matrix
    '''
    #here need to use list of list to store all edge type numbers and use KL divergence to update
    matrix = [ [ 0 for i in range(type_size) ] for j in range(type_size) ]
    repo = dict()
    for i in range(type_size):#initialize empty list to hold edge type vectors
        repo[i] = []

    for walk in walks:
        curr_repo = dict()#store each type number in current walk
        for edge in walk:
            edge_id = int(edge) - 1 
            if edge_id in curr_repo:
                curr_repo[edge_id] = curr_repo[edge_id]+1
            else:
                curr_repo[edge_id] = 1

        for i in range(type_size):
            if i in curr_repo:
                repo[i].append(curr_repo[i]) 
            else:
                repo[i].append(0) 
    
    for i in range(type_size):
        for j in range(type_size):  
            if evaluation_metric == 1:
                sim_score = wilcoxon_test(repo[i],repo[j])  
                matrix[i][j] = sim_score
            elif evaluation_metric == 2:
                sim_score = entroy_test(repo[i],repo[j])  
                matrix[i][j] = sim_score
            elif evaluation_metric == 3:
                sim_score = spearmanr_test(repo[i],repo[j])  
                matrix[i][j] = sim_score
            elif evaluation_metric == 4:
                sim_score = pearsonr_test(repo[i],repo[j])  
                matrix[i][j] = sim_score 
            else:
                raise ValueError('not correct evaluation metric! You need to choose from 1-4')  

    return matrix

'''
different ways to calculate correlation between edge-types
'''
#pairwised judgement
def wilcoxon_test(v1,v2):# original metric: the smaller the more similar 
    check = int(sum([a_i - b_i for a_i, b_i in zip(v1, v2)]))
    if check == 0:
        result = 0
    else: 
        result = stats.wilcoxon(v1, v2).statistic

    return 1/(math.sqrt(result)+1)

def entroy_test(v1,v2):#original metric: the smaller the more similar
    check = int(sum([a_i - b_i for a_i, b_i in zip(v1, v2)]))
    if check == 0:
        result = 0
    else: 
        result = stats.wilcoxon(v1, v2).statistic

    return result

def spearmanr_test(v1,v2):#original metric: the larger the more similar 
    if v1 == v2:
        result = 0.0
    else: 
        result = stats.wilcoxon(v1, v2).statistic

    return sigmoid(result)

def pearsonr_test(v1,v2):#original metric: the larger the more similar
    check = int(sum([a_i - b_i for a_i, b_i in zip(v1, v2)]))
    if check == 0:
        result = 0
    else: 
        result = stats.wilcoxon(v1, v2).statistic

    return sigmoid(result)

def cos_test(v1,v2): 
    return 1 - spatial.distance.cosine(v1, v2)

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def standardization(x):
    return (x+1)/2

def relu(x):
    return (abs(x) + x) / 2
    