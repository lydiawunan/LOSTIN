import json
import pandas as pd

from os import listdir
from os.path import isfile, join



allowable_features = {
    'node_type'    : ['input', 'intermediate', 'output'], 
    'command_type' : ['b', 'rf', 'rfz', 'rw', 'rwz', 'resub', 'resub -z'],
    'op_type'      : ['and_oper', 'or_oper', 'not_oper', 'misc'],
    
}

node_list = ['input', 'output', 'and_oper', 'or_oper', 'not_oper']



def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def node_type(opcode):
    if opcode == 'input' or opcode == 'output':
        return opcode
    if opcode in {'and_oper', 'or_oper', 'not_oper'}:
        t='intermediate'
    return t


def node_to_feature_vector(node):
    """
    Converts node object to feature list of indices
    :return: list
    """
    node_feature = [
            safe_index(allowable_features['node_type'], node_type(node['node_attributes']['node_type'])),
            safe_index(allowable_features['op_type'], node['node_attributes']['node_type']),
            0, 0, 0, 0, 0, 0
    ]
   
    return node_feature


def get_node_feature_dims():

    dim_list = list(map(len, [allowable_features['node_type'], allowable_features['op_type']]))
    last_dim = dim_list[-1]
    for i in range(6):
        dim_list.append(last_dim)

    return dim_list


def edge_to_feature_vector(source, sink):
    """
    Converts edge to feature list of indices
    :return: list
    """
    bond_feature = [node_list.index(source), node_list.index(sink)]

    return bond_feature


def get_edge_feature_dims():
    return list(map(len, [
        node_list,
        node_list
        ]))


def get_command_idx(cmd):
    if cmd in allowable_features['command_type']:
        return allowable_features['command_type'].index(cmd) + 1
    else:
        raise NotImplementedError



if __name__ == '__main__':
    ff_10 = pd.read_csv('flow_10.csv',header=None)
    ff_15 = pd.read_csv('flow_15.csv',header=None)
    ff_20 = pd.read_csv('flow_20.csv',header=None)
    ff_25 = pd.read_csv('flow_25.csv',header=None)

    keyword = 'delay'  # area or delay
    label_dir = 'dataset'
    label_list = [f for f in listdir(label_dir) if isfile(join(label_dir, f))]

    read_dir = 'epfl_graph/'
    vgraphs = ['adder', 'arbiter', 'bar', 'div', 'log2', 'max', 'multiplier', 'sin', 'sqrt', 'square', 'voter']
    
    node_feat     = []
    super_nodes   = []
    edge_list     = []
    edge_feat     = []
    graph_label   = []
    graph_choice  = []
    num_node_list = []
    num_edge_list = []

    count_10 = 0
    count_15 = 0
    count_20 = 0
    count_25 = 0

    for idx, vgraph in enumerate(vgraphs):
        f = open(read_dir + vgraph + '.json', 'r')
        d = json.load(f)
        f.close()

        label_file_10 = None
        label_file_15 = None
        label_file_20 = None
        label_file_25 = None

        for f in label_list:
            if (keyword in f) and (vgraph in f) and ('25' in f):
                label_file_25 = f
            elif (keyword in f) and (vgraph in f) and ('20' in f):
                label_file_20 = f
            elif (keyword in f) and (vgraph in f) and ('15' in f):
                label_file_15 = f
            elif (keyword in f) and (vgraph in f) and ('10' in f):
                label_file_10 = f

        label_10 = pd.read_csv(f'{label_dir}/{label_file_25}', header=None)
        label_15 = pd.read_csv(f'{label_dir}/{label_file_25}', header=None)
        label_20 = pd.read_csv(f'{label_dir}/{label_file_25}', header=None)
        label_25 = pd.read_csv(f'{label_dir}/{label_file_25}', header=None)

        nodes = d['nodes']
        edges = d['edges']
        node_index_map = dict() # map the node name to the index
            
        for index, n in enumerate(nodes):
            if n[0] not in node_index_map:
                node_index_map[n[0]] = index

            current_node_feat = node_to_feature_vector(n[1])
            node_feat.append(current_node_feat)
                        
        for e in edges:
            source = node_index_map[e[0]]
            sink = node_index_map[e[1]]
            edge_list.append([source,sink])
            
            source_type = nodes[source][1]['node_attributes']['node_type']
            sink_type = nodes[sink][1]['node_attributes']['node_type']

            edge_feat.append(edge_to_feature_vector(source_type, sink_type))

        num_node_list.append(len(nodes))
        num_edge_list.append(len(edges))

        # Processing Length 10 Flow
        for i in range(50000):
            commands = ff_10[0][count_10+i].split(';')
            super_node = []

            # Embed super node
            for j in range(10):
                if commands[j] == 'b':
                    super_node.append(1)
                elif commands[j] == 'rf':
                    super_node.append(2)
                elif commands[j] == 'rfz':
                    super_node.append(3)
                elif commands[j] == 'rw':
                    super_node.append(4)
                elif commands[j] == 'rwz':
                    super_node.append(5)
                elif commands[j] == 'resub':
                    super_node.append(6)
                elif commands[j] == 'resub -z':
                    super_node.append(7)
                else:
                    raise NotImplementedError

            for j in range(15):
                super_node.append(0)

            graph_choice.append(idx)
            graph_label.append([label_10[0][count_10+i]])
            super_nodes.append(super_node)

        # Processing Length 15 Flow
        for i in range(50000):
            commands = ff_15[0][count_15+i].split(';')
            super_node = []

            # Embed super node
            for j in range(15):
                if commands[j] == 'b':
                    super_node.append(1)
                elif commands[j] == 'rf':
                    super_node.append(2)
                elif commands[j] == 'rfz':
                    super_node.append(3)
                elif commands[j] == 'rw':
                    super_node.append(4)
                elif commands[j] == 'rwz':
                    super_node.append(5)
                elif commands[j] == 'resub':
                    super_node.append(6)
                elif commands[j] == 'resub -z':
                    super_node.append(7)
                else:
                    raise NotImplementedError

            for j in range(10):
                super_node.append(0)

            graph_choice.append(idx)
            graph_label.append([label_15[0][count_15+i]])
            super_nodes.append(super_node)

        # Processing Length 20 Flow
        for i in range(100000):
            commands = ff_20[0][count_20+i].split(';')
            super_node = []

            # Embed super node
            for j in range(20):
                if commands[j] == 'b':
                    super_node.append(1)
                elif commands[j] == 'rf':
                    super_node.append(2)
                elif commands[j] == 'rfz':
                    super_node.append(3)
                elif commands[j] == 'rw':
                    super_node.append(4)
                elif commands[j] == 'rwz':
                    super_node.append(5)
                elif commands[j] == 'resub':
                    super_node.append(6)
                elif commands[j] == 'resub -z':
                    super_node.append(7)
                else:
                    raise NotImplementedError

            for j in range(5):
                super_node.append(0)

            graph_choice.append(idx)
            graph_label.append([label_20[0][count_20+i]])
            super_nodes.append(super_node)

        # Processing Length 25 Flow
        for i in range(100000):
            commands = ff_25[0][count_25+i].split(';')
            super_node = []

            # Embed super node
            for j in range(25):
                if commands[j] == 'b':
                    super_node.append(1)
                elif commands[j] == 'rf':
                    super_node.append(2)
                elif commands[j] == 'rfz':
                    super_node.append(3)
                elif commands[j] == 'rw':
                    super_node.append(4)
                elif commands[j] == 'rwz':
                    super_node.append(5)
                elif commands[j] == 'resub':
                    super_node.append(6)
                elif commands[j] == 'resub -z':
                    super_node.append(7)
                else:
                    raise NotImplementedError

            graph_choice.append(idx)
            graph_label.append([label_25[0][count_25+i]])
            super_nodes.append(super_node)



    NODE = pd.DataFrame(node_feat)
    SUPER_NODE = pd.DataFrame(super_nodes)
    GRAPH = pd.DataFrame(graph_choice)
    EDGE_list = pd.DataFrame(edge_list)
    EDGE_feat = pd.DataFrame(edge_feat)
    node_num = pd.DataFrame(num_node_list)
    edge_num = pd.DataFrame(num_edge_list)
    labels = pd.DataFrame(graph_label)

    NODE.to_csv('node-feat.csv', index=False, header=False)
    SUPER_NODE.to_csv('node-super.csv', index=False, header=False)
    GRAPH.to_csv('graph-choice.csv', index=False, header=False)
    EDGE_list.to_csv('edge.csv', index=False, header=False)
    EDGE_feat.to_csv('edge-feat.csv', index=False, header=False)
    node_num.to_csv('num-node-list.csv', index=False, header=False)
    edge_num.to_csv('num-edge-list.csv', index=False, header=False)   
    labels.to_csv('graph-label.csv', index=False, header=False)

