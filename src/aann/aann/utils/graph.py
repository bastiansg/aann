import networkx as nx
import graph_tool.all as gt

from uuid import uuid1

from torch.nn import Module
from torch.nn.functional import sigmoid


def get_model_graph(model: Module, min_weight: float = 1.0) -> nx.DiGraph:
    model_params = (
        sigmoid(param).detach().swapaxes(0, 1).numpy()
        for param in model.model.parameters()
    )

    nodes = []
    edges = []
    prev_layer_nodes = []
    for l_idx, params in enumerate(model_params):
        src_dim, tgt_dim = params.shape
        if not prev_layer_nodes:
            src_nodes = (
                [
                    (
                        uuid1().hex,
                        {
                            "layer": l_idx,
                            "neuron": idx,
                        },
                    )
                    for idx in range(src_dim)
                ]
                if not prev_layer_nodes
                else prev_layer_nodes
            )

            nodes.extend(src_nodes)

        else:
            src_nodes = prev_layer_nodes

        tgt_layer = src_nodes[0][1]["layer"] + 1
        tgt_nodes = [
            (
                uuid1().hex,
                {
                    "layer": tgt_layer,
                    "neuron": idx,
                },
            )
            for idx in range(tgt_dim)
        ]

        nodes.extend(tgt_nodes)

        for src, src_data in src_nodes:
            for tgt, tgt_data in tgt_nodes:
                weight = params[src_data["neuron"]][tgt_data["neuron"]]
                if weight >= min_weight:
                    edge = (src, tgt, {"weight": weight})
                    edges.append(edge)

        prev_layer_nodes = tgt_nodes

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    return graph


def nx2gt(nx_graph: nx.DiGraph) -> gt.Graph:
    gt_graph = gt.Graph()

    layer = gt_graph.new_vertex_property("int")
    gt_graph.vertex_properties.layer = layer

    neuron = gt_graph.new_vertex_property("int")
    gt_graph.vertex_properties.neuron = neuron

    weight = gt_graph.new_edge_property("float")
    gt_graph.edge_properties.weight = weight

    node_prop_map = nx_graph.nodes(data=True)
    node_id_map = {
        node_id: gt_graph.add_vertex() for node_id in nx_graph.nodes
    }

    edge_prop_map = {
        (src, tgt): data for src, tgt, data in nx_graph.edges(data=True)
    }

    for node_id, v in node_id_map.items():
        gt_graph.vp.layer[v] = node_prop_map[node_id]["layer"]
        gt_graph.vp.neuron[v] = node_prop_map[node_id]["neuron"]

    for src, tgt in nx_graph.edges():
        e = gt_graph.add_edge(node_id_map[src], node_id_map[tgt])
        gt_graph.ep.weight[e] = edge_prop_map[(src, tgt)]["weight"]

    return gt_graph
