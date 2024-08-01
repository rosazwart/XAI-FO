"""
    Copy of source code in TORCH_GEOMETRIC.NN.MODELS.GNN_EXPLAINER from torch_geometric version 2.0.4 
    with adaptations https://github.com/PPerdomoQ/rare-disease-explainer/blob/main/3_Predictions_and_explanations.ipynb
"""

from math import sqrt
from typing import Optional
import copy

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import networkx as nx

import torch
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx
from inspect import signature

from torch_geometric.nn.models.explainer import (
    Explainer,
    clear_masks,
    set_masks,
)

EPS = 1e-15


class GNNExplainer(Explainer):
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and small subsets node features that play a crucial role in a
    GNN’s node-predictions.

    .. note::

        For an example of using GNN-Explainer, see `examples/gnn_explainer.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        gnn_explainer.py>`_.

    Args:
        model (torch.nn.Module): The GNN module to explain.
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        num_hops (int, optional): The number of hops the :obj:`model` is
            aggregating information from.
            If set to :obj:`None`, will automatically try to detect this
            information based on the number of
            :class:`~torch_geometric.nn.conv.message_passing.MessagePassing`
            layers inside :obj:`model`. (default: :obj:`None`)
        return_type (str, optional): Denotes the type of output from
            :obj:`model`. Valid inputs are :obj:`"log_prob"` (the model
            returns the logarithm of probabilities), :obj:`"prob"` (the
            model returns probabilities), :obj:`"raw"` (the model returns raw
            scores) and :obj:`"regression"` (the model returns scalars).
            (default: :obj:`"log_prob"`)
        feat_mask_type (str, optional): Denotes the type of feature mask
            that will be learned. Valid inputs are :obj:`"feature"` (a single
            feature-level mask for all nodes), :obj:`"individual_feature"`
            (individual feature-level masks for each node), and :obj:`"scalar"`
            (scalar mask for each each node). (default: :obj:`"feature"`)
        allow_edge_mask (boolean, optional): If set to :obj:`False`, the edge
            mask will not be optimized. (default: :obj:`True`)
        log (bool, optional): If set to :obj:`False`, will not log any learning
            progress. (default: :obj:`True`)
        **kwargs (optional): Additional hyper-parameters to override default
            settings in :attr:`~torch_geometric.nn.models.GNNExplainer.coeffs`.
    """

    coeffs = {
        'edge_size': 0.00005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
    }

    def __init__(self, model, epochs: int = 100, lr: float = 0.01, num_hops: Optional[int] = None, 
                 return_type: str = 'log_prob', feat_mask_type: str = 'feature', allow_edge_mask: bool = True,
                 log: bool = True, **kwargs):
        
        super().__init__(model, lr, epochs, num_hops, return_type, log)
        
        assert feat_mask_type in ['feature', 'individual_feature', 'scalar']
        
        self.allow_edge_mask = allow_edge_mask
        self.feat_mask_type = feat_mask_type
        self.coeffs.update(kwargs)

    def _initialize_masks(self, x, edge_index, init="normal"):
        (N, F), E = x.size(), edge_index.size(1)
        std = 0.1

        if self.feat_mask_type == 'individual_feature':
            self.node_feat_mask = torch.nn.Parameter(torch.randn(N, F) * std)   # mask applied to each feature per node
        elif self.feat_mask_type == 'scalar':
            self.node_feat_mask = torch.nn.Parameter(torch.randn(N, 1) * std)   # mask applied to each node
        else:
            self.node_feat_mask = torch.nn.Parameter(torch.randn(1, F) * std)   # mask applied to each feature

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))

        if self.allow_edge_mask:
            self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)   # mask applied to each edge

    def _clear_masks(self):
        clear_masks(self.model)
        self.node_feat_masks = None
        self.edge_mask = None

    def _loss(self, log_logits, prediction, node_idx: Optional[int] = None):
        if self.return_type == 'regression':
            if node_idx is not None and node_idx >= 0:
                loss = torch.cdist(log_logits[node_idx], prediction[node_idx])
            else:
                loss = torch.cdist(log_logits, prediction)
        else:
            if node_idx is not None and node_idx >= 0:
                loss = -log_logits[node_idx, prediction[node_idx]]
            else:
                loss = -log_logits[0, prediction[0]]

        if self.allow_edge_mask:
            m = self.edge_mask.sigmoid()
            edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
            loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
            ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
            loss = loss + self.coeffs['edge_ent'] * ent.mean()

        m = self.node_feat_mask.sigmoid()
        node_feat_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
        loss = loss + self.coeffs['node_feat_size'] * node_feat_reduce(m)
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def explain_graph(self, x, edge_index, **kwargs):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for a graph.

        Args:
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """

        self.model.eval()
        self._clear_masks()

        # all nodes belong to same graph
        batch = torch.zeros(x.shape[0], dtype=int, device=x.device)

        # Get the initial prediction.
        prediction = self.get_initial_prediction(x, edge_index, batch=batch,
                                                 **kwargs)

        self._initialize_masks(x, edge_index)
        self.to(x.device)
        if self.allow_edge_mask:
            set_masks(self.model, self.edge_mask, edge_index,
                      apply_sigmoid=True)
            parameters = [self.node_feat_mask, self.edge_mask]
        else:
            parameters = [self.node_feat_mask]
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        if self.log:  # pragma: no cover
            pbar = tqdm(total=self.epochs)
            pbar.set_description('Explain graph')

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            h = x * self.node_feat_mask.sigmoid()
            out = self.model(x=h, edge_index=edge_index, batch=batch, **kwargs)
            loss = self.get_loss(out, prediction, None)
            loss.backward()
            optimizer.step()

            if self.log:  # pragma: no cover
                pbar.update(1)

        if self.log:  # pragma: no cover
            pbar.close()

        node_feat_mask = self.node_feat_mask.detach().sigmoid().squeeze()
        if self.allow_edge_mask:
            edge_mask = self.edge_mask.detach().sigmoid()
        else:
            edge_mask = torch.ones(edge_index.size(1))

        self._clear_masks()
        return node_feat_mask, edge_mask


    def explain_node(self, node_idx, x, edge_index, **kwargs):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for node
        :attr:`node_idx`.

        Args:
            node_idx (int): The node to explain.
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """

        self.model.eval()
        self._clear_masks()

        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        # Only operate on a k-hop subgraph around `node_idx`.
        x, edge_index, mapping, hard_edge_mask, subset, kwargs = \
            self.subgraph(node_idx, -x, edge_index, **kwargs)
        print('Edge Index 1:', edge_index)
        # Get the initial prediction.
        prediction = self.get_initial_prediction(x, edge_index, **kwargs)
        print('Prediction:', prediction)

        self._initialize_masks(x, edge_index)
        self.to(x.device)

        if self.allow_edge_mask:
            set_masks(self.model, self.edge_mask, edge_index,
                      apply_sigmoid=True)
            parameters = [self.node_feat_mask, self.edge_mask]
        else:
            parameters = [self.node_feat_mask]
        print('Edge Index 2:', edge_index)
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        if self.log:  # pragma: no cover
            pbar = tqdm(total=self.epochs)
            pbar.set_description(f'Explain node {node_idx}')

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            h = x * self.node_feat_mask.sigmoid()
            print('X:', x)
            print('H:', h)
            print('Mask:', self.node_feat_mask.sigmoid())
            print('Edge Index 3:', edge_index)
            out = self.model(x=h, edge_index=edge_index, **kwargs)
            print('Out:', out)
            print('Prediction:', prediction)
            loss = self.get_loss(out, prediction, mapping)
            print('Loss:', loss)
            loss.backward()
            optimizer.step()

            if self.log:  # pragma: no cover
                pbar.update(1)
        print('Out:', out)
        print('Mapping:', mapping )
        if self.log:  # pragma: no cover
            pbar.close()

        node_feat_mask = self.node_feat_mask.detach().sigmoid()
        if self.feat_mask_type == 'individual_feature':
            new_mask = x.new_zeros(num_nodes, x.size(-1))
            new_mask[subset] = node_feat_mask
            node_feat_mask = new_mask
        elif self.feat_mask_type == 'scalar':
            new_mask = x.new_zeros(num_nodes, 1)
            new_mask[subset] = node_feat_mask
            node_feat_mask = new_mask
        node_feat_mask = node_feat_mask.squeeze()

        if self.allow_edge_mask:
            edge_mask = self.edge_mask.new_zeros(num_edges)
            print('Hard Edges:', hard_edge_mask)
            print('Len Hard Edges:', len(hard_edge_mask))
            print('Edges:', edge_mask)
            print('Len Edges:', len(edge_mask))      
            edge_mask[hard_edge_mask] = self.edge_mask.detach().sigmoid()
        else:
            edge_mask = torch.zeros(num_edges)
            edge_mask[hard_edge_mask] = 1

        self._clear_masks()

        return node_feat_mask, edge_mask

    def __repr__(self):
        return f'{self.__class__.__name__}()'
    
    def explain_link(self, node_idx1, node_idx2, x, edge_index, **kwargs): 
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for the edge
        connecting :attr:`node_idx1` and :attr:`node_idx2`.

        Args:
            node_idx1 (int): One node of the edge to explain.
            node_idx2 (int): The other node of the edge to explain.
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """

        self.model.eval()   # set to model evaluation
        self._clear_masks()

        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        _, emb = self.model(x, edge_index, edge_index)  # yielded node embeddings used to calculate edge existing or not

        # Only operate on a k-hop subgraph around `node_idx1` and `node_idx2`.
        x1, edge_index1, mapping1, hard_edge_mask1, subset1, kwargs1 = subgraph(node_idx=[node_idx1, node_idx2], 
                                                                                x=x, edge_index=edge_index, 
                                                                                flow='source_to_target', num_hops = self.num_hops, **kwargs)

        # x1                -> subgraph nodes with features
        # edge_index1       -> subgraph edges expressed in node pairs
        # mapping1          -> indices of node_idx1 and node_idx2 in subgraph node list
        # hard_edge_mask1   -> complete graph edges mask for subgraph edges

        # Initial embedding of nodes from relevant edge
        node_embedding_1 = emb[node_idx1]
        node_embedding_2 = emb[node_idx2]

        # Get the initial prediction from trained model on complete computation graph       
        prediction = torch.Tensor([torch.sum(node_embedding_1 * node_embedding_2, dim=-1)]).requires_grad_()
        print('Prediction from trained model:', prediction.sigmoid())

        # Initialize GNNExplainer masks for subgraph
        self._initialize_masks(x1, edge_index1)
        self.to(x1.device)

        if self.allow_edge_mask:
            # Apply mask on edges in each layer of model
            set_masks(self.model, self.edge_mask, edge_index1,
                      apply_sigmoid=True)
            parameters = [self.node_feat_mask, self.edge_mask]
        else:
            parameters = [self.node_feat_mask]
        
        # Optimize values of node and edge masks
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        if self.log:  # pragma: no cover
            pbar = tqdm(total=self.epochs)
            pbar.set_description(f'Explain edge between nodes {node_idx1} and {node_idx2}')
        
        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()   # Remove gradient from previous epoch
            criterion = torch.nn.MSELoss(reduction='sum')
            h = x1  # all subgraph nodes with their features
            
            pred1, emb1 = self.model(h, edge_index1, edge_index1)
            out = torch.tensor([torch.sum(emb1[mapping1[1]]* emb1[mapping1[0]], dim=-1)]).requires_grad_()

            loss = criterion(out, prediction)   # calculate loss given prediction of trained model and current model's prediction

            m = self.edge_mask.sigmoid()    # ranging between 0 and 1
            edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
            loss = loss + self.coeffs['edge_size'] * edge_reduce(m) # partition of sum over all mask values
            ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
            loss = loss + self.coeffs['edge_ent'] * ent.mean()

            loss.backward()     # Compute the gradients
            optimizer.step()    # Iterate over all parameter tensors that need to be updated which are the masks

            if self.log:  # pragma: no cover
                pbar.update(1)

        if self.log:  # pragma: no cover
            pbar.close()

        node_feat_mask = self.node_feat_mask.detach().sigmoid()
        if self.feat_mask_type == 'individual_feature':
            new_mask = x1.new_zeros(num_nodes, x1.size(-1))
            new_mask[subset1] = node_feat_mask
            node_feat_mask = new_mask
        elif self.feat_mask_type == 'scalar':
            new_mask = x1.new_zeros(num_nodes, 1)
            new_mask[subset1] = node_feat_mask
            node_feat_mask = new_mask
        node_feat_mask = node_feat_mask.squeeze()

        if self.allow_edge_mask:
            edge_mask = self.edge_mask.new_zeros(num_edges)
            edge_mask[hard_edge_mask1] = self.edge_mask.detach().sigmoid()
        else:
            edge_mask = torch.zeros(num_edges)
            edge_mask[hard_edge_mask1] = 1

        self._clear_masks()

        return node_feat_mask, edge_mask
    
def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes

def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target'):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.

    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        
        node_mask_index = subsets[-1].long()
        node_mask[node_mask_index] = True
        
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask

def subgraph(node_idx, x, edge_index, num_hops, **kwargs):
    r"""Returns the subgraph of the given node.

    Args:
        node_idx (int): The node to explain.
        x (Tensor): The node feature matrix.
        edge_index (LongTensor): The edge indices.
        **kwargs (optional): Additional arguments passed to the GNN module.
    :rtype: (Tensor, Tensor, LongTensor, LongTensor, LongTensor, dict)
    """
    num_nodes, num_edges = x.size(0), edge_index.size(0)
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=True, num_nodes=num_nodes, flow = 'source_to_target')

    x = x[subset]
    kwargs_new = {}
    for key, value in kwargs.items():
        if torch.is_tensor(value) and value.size(0) == num_nodes:
            kwargs_new[key] = value[subset]
        elif torch.is_tensor(value) and value.size(0) == num_edges:
            kwargs_new[key] = value[edge_mask]
        kwargs_new[key] = value  # TODO: this is not in PGExplainer
        
    return x, edge_index, mapping, edge_mask, subset, kwargs_new

def visualize_subgraph(node_idx, edge_index_full, edge_mask, nodes, node_labels_dict, y = None,
                       threshold = None,
                       edge_y = None,
                       node_alpha = None, 
                       seed= 10,
                       flow = 'source_to_target', 
                       num_hops = 1,
                       node_label = 'preflabel',
                       edge_labels = None,
                       show_inactive = False,
                       remove_unconnected = False,
                       **kwargs):
    r"""Visualizes the subgraph given an edge mask :attr:`edge_mask`.

    Args:
        node_idx (int, list, tuple, Tensor): The node id(s) to explain.
        edge_index_full (LongTensor): The indices for each existing edge in graph.
        edge_mask (Tensor): The edge mask.
        y (Tensor, optional): The ground-truth node-prediction labels used
            as node colorings. All nodes will have the same color
            if :attr:`node_idx` is :obj:`-1`.(default: :obj:`None`).
        threshold (float, optional): Sets a threshold for visualizing
            important edges. If set to :obj:`None`, will visualize all
            edges with transparancy indicating the importance of edges.
            (default: :obj:`None`)
        edge_y (Tensor, optional): The edge labels used as edge colorings.
        node_alpha (Tensor, optional): Tensor of floats (0 - 1) indicating
            transparency of each node.
        seed (int, optional): Random seed of the :obj:`networkx` node
            placement algorithm. (default: :obj:`10`)
        **kwargs (optional): Additional arguments passed to
            :func:`nx.draw`.

    :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
    """

    assert edge_mask.size(0) == edge_index_full.size(1) # check whether there is an equal number of edges in the mask and set of all edges

    # Only operate on a k-hop subgraph around node(s) given by `node_idx`
    subset, edge_index, mapping_sub, hard_edge_mask = k_hop_subgraph(
        node_idx, num_hops, edge_index_full, relabel_nodes=True,
        num_nodes=None, flow=flow)

    edge_mask = edge_mask[hard_edge_mask]   # only get GNNExplainer mask values from edges in subgraph

    if threshold is not None:
        edge_mask = (edge_mask >= threshold).to(torch.float)    # edge mask value needs to exceed given threshold to be included

    if y is None:
        y = torch.zeros(edge_index.max().item() + 1,
                        device=edge_index.device)
        y2 = torch.zeros(edge_index.max().item() + 1,
                        device=edge_index.device)
    else:
        y2 = y[subset].to(torch.int)    # maintain node class identifier as node attribute
        y = y[subset].to(torch.float) / y.max().item()
        
    if edge_y is None:
        edge_color = ['black'] * edge_index.size(1) # all edges are represented with color black
    else:
        colors = list(plt.rcParams['axes.prop_cycle'])
        edge_color = [
            colors[i % len(colors)]['color']
            for i in edge_y[hard_edge_mask]
        ]

    data = Data(edge_index=edge_index, att=edge_mask,
                edge_color=edge_color, y=y, y2=y2, num_nodes=y.size(0)).to('cpu')  # store subgraph in data object describing homogeneous graph

    G = to_networkx(data, node_attrs=['y', 'y2'],
                    edge_attrs=['att', 'edge_color'])   # convert to networkx graph object

    G2 = copy.deepcopy(G)
    if num_hops >= 1 and not show_inactive: 
      for indx, edge in enumerate(G.edges): 
        if edge_mask[indx] < threshold:
          G2.remove_edge(edge[0], edge[1])  # remove all edges that do not exceed threshold
      removed_nodes = list(nx.isolates(G2))
      G2.remove_nodes_from(removed_nodes)   # remove all nodes that have become isolates due to edge removal
    
    if remove_unconnected:
        G2 = G2.to_undirected()
        for component in list(nx.connected_components(G2)):
            if mapping_sub[0].item() in component or mapping_sub[1].item() in component:
                continue
            else:
                for node in component:
                    G2.remove_node(node)
                    

    active = torch.tensor(list(G2.nodes())).long()
    
    G3 = copy.deepcopy(G2)

    mapping = {k: str(nodes.iloc[i][node_label]) + ' ' + str(i) for k, i in enumerate(subset.tolist())}
    mapping2 = {str(nodes.iloc[i][node_label]) + ' ' + str(i): i for k, i in enumerate(subset.tolist())}
    G3 = nx.relabel_nodes(G3, mapping)

    node_args = set(signature(nx.draw_networkx_nodes).parameters.keys())
    node_kwargs = {k: v for k, v in kwargs.items() if k in node_args}
    node_kwargs['node_size'] = kwargs.get('node_size') or 800

    label_args = set(signature(nx.draw_networkx_labels).parameters.keys())
    label_kwargs = {k: v for k, v in kwargs.items() if k in label_args}
    label_kwargs['font_size'] = kwargs.get('font_size') or 10

    pos = nx.spring_layout(G3, seed=seed)
    ax = plt.gca()
    for source, target, data in G3.edges(data=True):
        ax.annotate(
            '', xy=pos[target], xycoords='data', xytext=pos[source],
            textcoords='data', arrowprops={
                'arrowstyle': "->",
                'alpha': max(data['att'], 0.1),
                'color': data['edge_color'],
                'shrinkA': sqrt(node_kwargs['node_size']) / 2.0,
                'shrinkB': sqrt(node_kwargs['node_size']) / 2.0,
                'connectionstyle': "arc3,rad=0.1",
            })

    if node_alpha is None:
        node_label_color_dict = {node: (node_labels_dict[label], color) for node, label, color in zip(G3.nodes(), y2[active].tolist(), y[active].tolist())}
        
        # Create legend entries for each color
        unique_colors = {}
        for node, (label, value) in node_label_color_dict.items():
            unique_colors[value] = label
        
        # Create color mapper
        color_values = [value for _, value in node_label_color_dict.values()]
        norm = Normalize(vmin=min(color_values), vmax=max(color_values))
        colormap = plt.get_cmap('cool')
        
        # Draw nodes with color
        for node, (label, color) in node_label_color_dict.items():
            color_mapped = colormap(norm(color))
            nx.draw_networkx_nodes(G3, pos, nodelist=[node], node_color=color_mapped, label=label, **node_kwargs)
        
        # Draw legend 
        legend_handles = []
        for value, label in unique_colors.items():
            color = colormap(norm(value))
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=label,
                                            markerfacecolor=color, markersize=10))

        plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1), title='Node Types')
        
    else:
        node_alpha_subset = node_alpha[subset]
        assert ((node_alpha_subset >= 0) & (node_alpha_subset <= 1)).all()
        nx.draw_networkx_nodes(G3, pos, alpha=node_alpha_subset.tolist(),
                                node_color=y.tolist(), **node_kwargs)

    nx.draw_networkx_labels(G3, pos, **label_kwargs)

    if edge_labels is not None: 
        edge_labels_sub = {}
        edge_labels_sub_attr = {}
        for (n1,n2) in G3.edges():
            edge_labels_sub[(n1, n2)] = edge_labels[(mapping2[n1], mapping2[n2])]
            edge_labels_sub_attr[(n1, n2)] = {'label': edge_labels[(mapping2[n1], mapping2[n2])]}
      
        nx.draw_networkx_edge_labels(
            G3, pos,
            edge_labels=edge_labels_sub,
            font_color='red', 
            font_size = 9
        )
        
        # Store relation label for each edge
        nx.set_edge_attributes(G3, edge_labels_sub_attr)

    return ax, G3