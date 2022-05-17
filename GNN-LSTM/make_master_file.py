### script for writing meta information of datasets into master.csv
### for graph property prediction datasets.
import pandas as pd

dataset_list = []
dataset_dict = {}


### add cdfg_lut
name = 'vgraph'
dataset_dict[name] = {'eval metric': 'rmse'}
dataset_dict[name]['download_name'] = 'vgraph'
dataset_dict[name]['version'] = 1
dataset_dict[name]['add_inverse_edge'] = False 
dataset_dict[name]['split'] = 'scaffold'
dataset_dict[name]['num tasks'] = 1
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = False
dataset_dict[name]['task type'] = 'regression'
dataset_dict[name]['num classes'] = -1
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['binary'] = False


df = pd.DataFrame(dataset_dict)
# saving the dataframe 
df.to_csv('master.csv')