import json
import pandas as pd

allowable_features = {
    'node_type': ['input', 'intermediate', 'output'], 
    'op_type': ['and_oper', 'or_oper', 'not_oper', 'misc'],
    
}

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
            ]
   
    return node_feature


def get_node_feature_dims():
    return list(map(len, [
        allowable_features['node_type'],
        allowable_features['op_type']
        ]))

def edge_to_feature_vector(edge):
    """
    Converts edge to feature list of indices
    :return: list
    """
    bond_feature = [0]
    return bond_feature

def get_edge_feature_dims():
    return [1]


'''
num_graph=11
read_dir='epfl_graph/'
vgraphs = ['adder', 'arbiter', 'bar', 'div', 'log2', 'max', 'multiplier', 'sin', 'sqrt', 'square', 'voter']

node_feat=[]
edge_list=[]
edge_feat=[]
num_node_list=[]
num_edge_list=[]

for i in range(num_graph):
    f = open(read_dir+vgraphs[i]+'.json', 'r')
    d = json.load(f)
    f.close()
    nodes=d['nodes']
    edges=d['edges']

    node_index_map=dict() # map the node name to the index
    index=0

    for n in nodes:
        if n[0] not in node_index_map:
            node_index_map[n[0]]=index
        node_feat.append(node_to_feature_vector(n[1]))
        index=index+1
    
    for e in edges:
        source=node_index_map[e[0]]
        sink=node_index_map[e[1]]
        edge_list.append([source,sink])
        edge_feat.append(edge_to_feature_vector(e[2]))

    num_node_list.append(len(nodes))
    num_edge_list.append(len(edges))

NODE=pd.DataFrame(node_feat)
EDGE_list=pd.DataFrame(edge_list)
EDGE_feat=pd.DataFrame(edge_feat)
node_num = pd.DataFrame(num_node_list)
edge_num = pd.DataFrame(num_edge_list)

NODE.to_csv('node-feat.csv',index=False,header=False)
EDGE_list.to_csv('edge.csv',index=False,header=False)
EDGE_feat.to_csv('edge-feat.csv',index=False,header=False)
node_num.to_csv('num-node-list.csv',index=False,header=False)
edge_num.to_csv('num-edge-list.csv',index=False,header=False)   
'''
