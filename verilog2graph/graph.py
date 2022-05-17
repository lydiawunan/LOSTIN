import pathpy as pp
import igraph
import lark

from collections import deque


# Graph class
class Graph:
    def __init__(self):
        self._graph = pp.Network(directed=True)
        self._num_ANDs = 0
        self._num_ORs = 0
        self._num_NOTs = 0

    def create_node(self, node_name, node_type):
        node_dict = {
            'node_type': node_type, 
            'neighbors': []
        }
        self._graph.add_node(node_name, node_attributes=node_dict)

    def remove_node(self, node_name):
        self._graph.remove_node(node_name)

    # Assume unweighted edge at the moment...
    def connect(self, node_l, node_r):
        self._graph.add_edge(v=node_l, w=node_r)

    def disconnect(self, node_l, node_r):
        self._graph.remove_edge(v=node_l, w=node_r)

    def _get_operater_node(self, oper_type):
        if oper_type == 'and_oper':
            node_type = f'&_{self._num_ANDs}'
            self._num_ANDs += 1
        elif oper_type == 'or_oper':
            node_type = f'|_{self._num_ORs}'
            self._num_ORs += 1
        elif oper_type == 'not_oper':
            node_type = f'~_{self._num_NOTs}'
            self._num_NOTs += 1
        else:
            raise NotImplementedError

        return node_type

    def generate_verilog_graph(self, module, sub_modules=None):
        # Initialization
        inputs, outputs, netlist, table = {}, {}, {}, {}
        self._name = module.module_name

        # Create temporary input, output, net dictionaries
        for v_in in module.input_declarations:
            if isinstance(v_in.net_name, lark.tree.Tree):
                for node in v_in.net_name.children:
                    inputs[node] = []
                    self.create_node(node, 'input')
            else:
                inputs[v_in.net_name] = []
                self.create_node(v_in.net_name, 'input')

        for v_out in module.output_declarations:
            if isinstance(v_out.net_name, lark.tree.Tree):
                for node in v_out.net_name.children:
                    outputs[node] = []
                    self.create_node(node, 'output')
            else:
                outputs[v_out.net_name] = []
                self.create_node(v_out.net_name, 'output')

        for v_net in module.net_declarations:
            if isinstance(v_net.net_name, lark.tree.Tree):
                for node in v_net.net_name.children:
                    netlist[node] = []
            else:
                netlist[v_net.net_name] = []

        # Create component look-up table if sub-modules are there
        if sub_modules:
            for sub_module in sub_modules:
                table[sub_module.module_name] = {}

                for in_port in sub_module.input_declarations:
                    table[sub_module.module_name][in_port.net_name] = 'in'

                for out_port in sub_module.output_declarations:
                    table[sub_module.module_name][out_port.net_name] = 'out'

            # Create instance nodes + update dictionaries
            for v_inst in module.module_instances:
                self.create_node(v_inst.instance_name, v_inst.module_name)
                
                for port, net in v_inst.ports.items():
                    port_polarity = table[v_inst.module_name][port] 

                    if net in netlist.keys():
                        netlist[net].append((v_inst.instance_name, port_polarity))
                        continue    

                    if net in inputs.keys():
                        inputs[net].append(v_inst.instance_name)
                        continue    

                    if net in outputs.keys():
                        outputs[net].append(v_inst.instance_name)
                        continue

        # Operator based 
        for assignment in module.assignments:
            unpacked = assignment.assignments[0]
            assigned_node = unpacked[0]
            expression = unpacked[1]
            tree_q = deque()

            tree_q.append((assigned_node, expression))

            while tree_q:
                parent_node, tree = tree_q.popleft()
                operator_node = self._get_operater_node(tree.data)
                self.create_node(operator_node, tree.data)
                self.connect(operator_node, parent_node)
                children_nodes = tree.children;
                for children_node in children_nodes:
                    if isinstance(children_node, lark.tree.Tree):
                        tree_q.append((operator_node, children_node))
                    else:
                        self.connect(children_node, operator_node)


        # Complete edge connections using temporary dictionaries
        if sub_modules:
            for net, item_list in netlist.items():
                item_len = len(item_list)
                for current_idx, (current_node, current_polarity) in enumerate(item_list):
                    dynamic_idx = current_idx+1
                    while dynamic_idx < item_len:
                        neighbor_node, _ = item_list[dynamic_idx]
                        if current_polarity == 'out':
                            self.connect(current_node, neighbor_node)
                        else:
                            self.connect(neighbor_node, current_node)
                        dynamic_idx += 1    

            for in_net, item_list in inputs.items():
                for node in item_list:
                    self.connect(in_net, node)

            for out_net, item_list in outputs.items():
                for node in item_list:
                    self.connect(node, out_net)

    def visualize_graph(self, image, visual_style):
        g = igraph.Graph(directed=True)

        for e in self._graph.edges:
            if g.vcount()== 0 or e[0] not in g.vs()["name"]:
                g.add_vertex(e[0])
            if g.vcount()== 0 or e[1] not in g.vs()["name"]:
                g.add_vertex(e[1])
            g.add_edge(e[0], e[1])

        visual_style["vertex_label"] = g.vs["name"]

        igraph.plot(g, f'{image}.png', **visual_style)
