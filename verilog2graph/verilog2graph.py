import networkx as nx
import json
from parser import parse_verilog
from graph import Graph

### convert to networkx graphs
def network_to_networkx(network):
    """method to export a pathpy Network to a networkx compatible graph

    Parameters
    ----------
    network: Network

    Returns
    -------
    networkx Graph or DiGraph
    """
    # keys to exclude since they are handled differently in networkx
    excluded_node_props = {"degree", "inweight", "outweight", "indegree", "outdegree"}
    try:
        import networkx as nx
    except ImportError:
        raise PathpyError("To export a network to networkx it must be installed")

    directed = network.directed
    if directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()

    for node_id, node_props in network.nodes.items():
        valid_props = {k: v for k, v in node_props.items() if k not in excluded_node_props}
        graph.add_node(node_id, **valid_props)

    for edge, edge_props in network.edges.items():
        graph.add_edge(*edge, **edge_props)

    return graph


### remove wire nodes
def graph_optimize(G):
    wire_nodes=[]
    for n in G.nodes():
        if n.startswith('n'):
            wire_nodes.append(n)

            # get in edges of wire nodes
            in_edge=G.in_edges(n)
            # get the parent of wire nodes
            for e in in_edge:
                source_node=e[0]

            # get out edges of wire ndoes
            out_edge=G.out_edges(n)
            # get the children of wire nodes
            target_node=[]
            for e in out_edge:
                target_node.append(e[1])
            
            # add new edges
            for target in target_node:
                G.add_edges_from([(source_node, target)])

    # remove wire nets
    for n in wire_nodes:
        G.remove_node(n)

    #print(len(wire_nodes))
    return G


### save the graph into json
def json_save(G, fname):
    f = open(fname + '.json', 'w')
    G_dict = dict(nodes=[[n, G.nodes[n]] for n in G.nodes()], \
                  edges=[(e[0], e[1], G.edges[e]) for e in G.edges()])
    json.dump(G_dict, f)
    f.close()

### load the graph from json
def json_load(fname):
    f = open(fname + '.json', 'r')
    G = nx.DiGraph()
    d = json.load(f)
    f.close()
    G.add_nodes_from(d['nodes'])
    G.add_edges_from(d['edges'])
    return G


### 

path = 'epfl_new/'
filename = ['adder.v', 'arbiter.v', 'bar.v', 'div.v', 'log2.v', 'max.v', 'multiplier.v', 'sin.v', 'sqrt.v', 'square.v', 'voter.v']

graph_path='epfl_graph/'

for fname in filename:
    top = parse_verilog(open(path + 'new_' + fname).read())
    top_module = top.modules[0]

    graph_oper = Graph()
    graph_oper.generate_verilog_graph(top_module)

    # convert to networkx graph
    G = network_to_networkx(graph_oper._graph)
    G_new = graph_optimize(G)

    # save graphs into json
    json_save(G_new, graph_path+fname.split('.')[0])
    # print(fname)
           