# IMPORTANT: Run with `xaifo` environment
# TODO: error with `xaifognn` environment

import pandas as pd
import networkx as nx

import torch
from torch.utils.data import DataLoader

from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch

from ray import tune
from ray.tune.schedulers import ASHAScheduler

import predictor.edge2vec.transition3 as transitions
import predictor.edge2vec.edge2vec3 as edge2vec

from predictor.gnn.linkpred_model import LinkPredModel, train, test


def optim(args):
    # Node embedding using Edge2Vec
    trans_matrix = transitions.initialize_edge_type_matrix(args['type_size'])
    
    G1 = nx.from_pandas_edgelist(args['df'], 'index_head', 'index_tail', 'type', create_using=nx.DiGraph(), edge_key= (('type', int),('id', int)))
    G1 = G1.to_undirected()
    for edge in G1.edges():
        G1[edge[0]][edge[1]]['weight'] = 1.0
    
    for i in range(args['epoch_e2v']):
        walks = transitions.simulate_walks_1(G1, args['num_walks'], args['walk_length'], trans_matrix, True, args['p'], args['q'])
        trans_matrix = transitions.update_trans_matrix(walks, args['type_size'], 3)
    
    walks = edge2vec.simulate_walks_2(G1, args['num_walks'], args['walk_length'], trans_matrix, args['p'], args['q'])
    w2v_model = edge2vec.Word2Vec(walks, vector_size=args['dimensions_e2v'], window=args['walk_length']-1, min_count=0, sg=1, workers=8, epochs=args['epoch_e2v'])
    
    # Create a graph with all edges and nodes including the obtained embeddings for each node
    e2v_embedding = pd.DataFrame(columns = ['Node', 'Embedding'])
    for idx, key in enumerate(w2v_model.wv.index_to_key):
        e2v_embedding.loc[int(key)] = pd.Series({'Node':int(key), 'Embedding':list(w2v_model.wv[key])})
    e2v_embedding = e2v_embedding.sort_values('Node')
    
    # Build graph with nodes and their embedding as node feature
    G2 = nx.DiGraph()   # TODO: changed from Graph
    for ind, node in e2v_embedding.iterrows(): 
        G2.add_node(int(node['Node']), node_feature=torch.Tensor(node['Embedding']))
    for ind, edge in args['df'].iterrows(): 
        G2.add_edge(int(edge['index_head']), int(edge['index_tail']))
        
    # Split graph dataset into train, test and validation sets
    dataset = GraphDataset(G2, task='link_pred', edge_train_mode="all", edge_negative_sampling_ratio=args['edge_neg_sampl'])
    
    datasets = {}
    datasets['train'], datasets['val'], datasets['test']= dataset.split(transductive=True, split_ratio=[0.8, 0.1, 0.1])
    
    # Set up link prediction model
    input_dim = datasets['train'].num_node_features
    
    model = LinkPredModel(input_dim, args['hidden_dim'], args['output_dim'], args['layers'], args['aggr'], args['dropout'], args['device']).to(args['device'])
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9, weight_decay=5e-4)
    
    # Generate dataloaders
    dataloaders = {split: DataLoader(ds, collate_fn=Batch.collate([]), batch_size=1, shuffle=(split=='train')) for split, ds in datasets.items()}
    
    best_model, best_x, perform = train(model, dataloaders, optimizer, args, ho = False)
    
    best_train_roc = test(best_model, dataloaders['train'], args)
    best_val_roc = test(best_model, dataloaders['val'], args)
    best_test_roc = test(best_model, dataloaders['test'], args)
    
    log = "Train: {:.4f}, Val: {:.4f}, Test: {:.4f}"
    print(log.format(best_train_roc, best_val_roc, best_test_roc))
    
    tune.report(val_auc=best_val_roc, train_auc = best_train_roc, test_auc = best_test_roc)

if __name__ == "__main__":
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', torch_device)
    
    # Load data
    dataset_nr = input('Enter dataset number (1 or 2):')
    assert dataset_nr == 1 or 2

    if dataset_nr == 1:
        dataset_prefix = 'prev'
    else:
        dataset_prefix = 'restr'
    
    disease_prefix = input('Enter disease prefix (dmd, hd, oi):')
    assert disease_prefix == 'dmd' or 'hd' or 'oi'
    
    edge_df = pd.read_csv(f'output/{disease_prefix}/{dataset_prefix}_{disease_prefix}_indexed_edges.csv')
    
    search_args = {
        'device': torch_device, 
        "hidden_dim" : tune.choice([64, 128, 256]),
        'output_dim': tune.choice([64, 128, 256]),
        "epochs" : tune.choice([100, 150, 200]),
        'type_size' : len(set(edge_df['type'])),
        'epoch_e2v' : tune.choice([5, 10]),
        'num_walks' : tune.choice([2, 4, 6]),
        'walk_length' : tune.choice([3, 5, 7]),
        'p' : tune.choice([0.5, 0.75, 1]),
        'q' : tune.choice([0.5, 0.75, 1]),
        'dimensions_e2v' : tune.choice([32, 64, 128]),
        'df': edge_df, 
        'lr': tune.loguniform(1e-4, 1e-1), 
        'aggr': tune.choice(['mean', 'sum']), 
        'dropout': tune.choice([0, 0.1, 0.2]), 
        'layers': tune.choice([2, 4, 6]),
        'edge_neg_sampl': tune.choice([0.5, 1.0, 1.5])
    }

    scheduler = ASHAScheduler(
        max_t=10,
        grace_period=1,
        reduction_factor=2)

    result = tune.run(
        tune.with_parameters(optim),
        resources_per_trial = {"cpu": 8}, #change this value according to the gpu units you would like to use
        config = search_args,
        metric = "val_auc",
        mode = "max",
        num_samples = 30, #select the maximum number of models you would like to test
        scheduler = scheduler, 
        resume = False, 
        local_dir = "output")
    
    best_trial = result.get_best_trial("val_auc")
    print("Best trial config: {}".format(best_trial.config))