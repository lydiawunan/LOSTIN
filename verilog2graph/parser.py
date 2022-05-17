from lark import Lark, Transformer, v_args

from typing import Dict, List, Tuple, Optional, Union

verilog_netlist_grammar = r"""
    start: description*
    
    ?description: module
    
    ?module: "module" identifier list_of_ports? ";" module_item* "endmodule"
    
    list_of_ports: "(" port ("," port)* ")"
    ?port: identifier
        | named_port_connection
    
    ?module_item: input_declaration
        | output_declaration
        | net_declaration
        | continuous_assign
        | module_instantiation
        
    input_declaration: "input" range? list_of_variables ";"
    
    output_declaration: "output" range? list_of_variables ";"
    
    net_declaration: "wire" range? list_of_variables ";"


    
    continuous_assign: "assign" list_of_assignments ";"
    
    list_of_assignments: assignment ("," assignment)*
    
    assignment: lvalue "=" expression
    
    ?lvalue: identifier
        | identifier_indexed
        | identifier_sliced
        | concatenation
        
    concatenation: "{" expression ("," expression)* "}"

    grouping: "(" lvalue ")"

    and_oper: expression "&" expression

    or_oper: expression "|" expression
    
    not_oper: "~" lvalue

    
    ?expression: identifier
        | identifier_indexed
        | identifier_sliced
        | concatenation
        | number
        | and_oper
        | or_oper
        | not_oper
        | grouping
    
    identifier_indexed: identifier "[" number "]"
    identifier_sliced: identifier range
    
    module_instantiation: identifier module_instance ("," module_instance)* ";"
    
    module_instance: identifier "(" list_of_module_connections? ")"
    
    list_of_module_connections: module_port_connection ("," module_port_connection)*
        | named_port_connection ("," named_port_connection)*
        
    module_port_connection: expression
    
    named_port_connection: "." identifier "(" expression ")"
    
    identifier: CNAME
    
    ?range: "[" number ":" number "]"
    
    ?list_of_variables: identifier ("," identifier)*

    string: ESCAPED_STRING

    // FIXME TODO: Use INT
    unsigned_hex_str: HEXDIGIT+
    signed_hex_str: ( "-" | "+" ) unsigned_hex_str
    
    number: 
        | unsigned_hex_str -> number
        | signed_hex_str -> number
        | unsigned_hex_str base unsigned_hex_str -> number_explicit_length
        | base unsigned_hex_str -> number_implicit_length
    
    base: BASE
    BASE: "'b" | "'B" | "'h" | "'H" | "'o" | "'O'" | "'d" | "'D"

    COMMENT_SLASH: /\/\*(\*(?!\/)|[^*])*\*\//
    COMMENT_BRACE: /\(\*(\*(?!\))|[^*])*\*\)/
    
    NEWLINE: /\\?\r?\n/

    %import common.WORD
    %import common.ESCAPED_STRING
    %import common.CNAME
    //%import common.SIGNED_NUMBER
    //%import common.INT
    //%import common.SIGNED_INT
    %import common.WS
    %import common.HEXDIGIT

    %ignore WS
    %ignore COMMENT_SLASH
    %ignore COMMENT_BRACE
    %ignore NEWLINE
"""


class Number:
    def __init__(self, length: Optional[int], base: Optional[str], mantissa: str):
        assert isinstance(mantissa, str), "Mantissa is expected to be a string."
        assert length is None or isinstance(length, int)
        self.length = length
        self.base = base
        self.mantissa = mantissa

    def as_integer(self):
        base_map = {
            'h': 16,
            'b': 2,
            'd': 10,
            'o': 8
        }

        if self.base is None:
            int_base = 10
        else:
            base = self.base.lower()
            assert base in base_map, "Unknown base: '{}'".format(base)
            int_base = base_map[base]

        return int(self.mantissa, base=int_base)

    def __int__(self):
        return self.as_integer()

    def as_bits_lsb_first(self):
        """
        Get integer value as a list of bits.
        If the length of the Number is not None then the list is either extended or truncated to the given length.
        Extension is sign extended.
        :return:
        """
        value = self.as_integer()
        x = value
        bits = []
        while x != 0:
            bits.append(x & 1)
            x //= 2

        if self.length is not None:
            if len(bits) < self.length:
                sign = 1 if value < 0 else 0
                # Extend.
                bits.extend([sign] * (self.length - len(bits)))
            elif len(bits) > self.length:
                # Truncate
                bits = bits[0:self.length]

        return bits

    def as_bits_msb_first(self):
        return list(reversed(self.as_bits_lsb_first()))

    def __repr__(self):
        if self.base is None:
            return "{}".format(self.as_integer())
        elif self.length is None:
            return "'{}{}".format(self.base, self.mantissa)
        else:
            return "{}'{}{}".format(self.length, self.base, self.mantissa)


def test_class_number():
    assert Number(None, None, '12').as_bits_lsb_first() == [0, 0, 1, 1]
    assert Number(None, None, '12').as_bits_msb_first() == [1, 1, 0, 0]
    assert Number(5, None, '12').as_bits_msb_first() == [0, 1, 1, 0, 0]
    assert Number(3, None, '12').as_bits_msb_first() == [1, 0, 0]
    assert Number(3, 'h', 'c').as_bits_msb_first() == [1, 0, 0]


class Range:

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def to_indices(self):
        """
        Convert to list of indices in the range.
        :return:
        """
        return list(reversed(range(self.end.as_integer(), self.start.as_integer() + 1)))

    def __repr__(self):
        return "[{}:{}]".format(self.start, self.end)


class Vec:

    def __init__(self, name: str, range: Range):
        self.name = name
        self.range = range

    def __repr__(self):
        return "{}{}".format(self.name, self.range)


# class PortConnection:
#
#     def __init__(self, port_name: str, signal_name: str):
#         self.port_name = port_name
#         self.signal_name = signal_name
#
#     def __repr__(self):
#         return ".{}({})".format(self.port_name, self.signal_name)


class Identifier:

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return self.name


class IdentifierIndexed:

    def __init__(self, name: str, index):
        self.name = name
        self.index = index

    def __repr__(self):
        return "{}[{}]".format(self.name, self.index)


class IdentifierSliced:

    def __init__(self, name: str, range: Range):
        self.name = name
        self.range = range

    def __repr__(self):
        return "{}{}".format(self.name, self.range)


class Concatenation:

    def __init__(self, elements: List[Union[Identifier, IdentifierIndexed, IdentifierSliced]]):
        self.elements = elements

    def __repr__(self):
        return "Concatenation()".format(", ".join([str(e) for e in self.elements]))


class ModuleInstance:

    def __init__(self, module_name: str, instance_name: str, ports: Dict[str, str]):
        self.module_name = module_name
        self.instance_name = instance_name
        self.ports = ports

    def __repr__(self):
        return "ModuleInstance({}, {}, {})".format(self.module_name, self.instance_name, self.ports)


class NetDeclaration:
    def __init__(self, net_name: str, range: Range):
        self.net_name = net_name
        self.range = range

    def __repr__(self):
        if self.range is not None:
            return "NetDeclaration({} {})".format(self.net_name, self.range)
        else:
            return "NetDeclaration({})".format(self.net_name)


class OutputDeclaration(NetDeclaration):
    def __repr__(self):
        if self.range is not None:
            return "OutputDeclaration({} {})".format(self.net_name, self.range)
        else:
            return "OutputDeclaration({})".format(self.net_name)


class InputDeclaration(NetDeclaration):
    def __repr__(self):
        if self.range is not None:
            return "InputDeclaration({} {})".format(self.net_name, self.range)
        else:
            return "InputDeclaration({})".format(self.net_name)


class ContinuousAssign:
    def __init__(self, assignments: List[Tuple[str, str]]):
        assert isinstance(assignments, list)
        self.assignments = assignments

    def __repr__(self):
        return "ContinuousAssign({})" \
            .format(", ".join(("{} = {}".format(l, r) for l, r in self.assignments)))


class Module:

    def __init__(self, module_name: str, port_list: List[str], module_items: List):
        self.module_name = module_name
        self.port_list = port_list

        self.module_items = module_items

        self.net_declarations = []
        self.output_declarations = []
        self.input_declarations = []
        self.module_instances = []
        self.assignments = []
        self.sub_modules = []

        for it in module_items:
            if isinstance(it, OutputDeclaration):
                self.output_declarations.append(it)
            elif isinstance(it, InputDeclaration):
                self.input_declarations.append(it)
            elif isinstance(it, NetDeclaration):
                self.net_declarations.append(it)
            elif isinstance(it, ModuleInstance):
                self.module_instances.append(it)
            elif isinstance(it, ContinuousAssign):
                self.assignments.append(it)
            elif isinstance(it, Module):
                self.sub_modules.append(it)

    def __repr__(self):
        return "Module({}, {}, {})".format(self.module_name, self.port_list, self.module_items)


class Netlist:
    def __init__(self, modules: List[Module]):
        self.modules = modules

    def __repr__(self):
        return "Netlist({})".format(self.modules)


class VerilogTransformer(Transformer):
    list_of_ports = list

    def unsigned_hex_str(self, hexstr):
        return "".join((str(h) for h in hexstr))

    @v_args(inline=True)
    def signed_hex_str(self, sign, hexstr):
        return sign + hexstr

    @v_args(inline=True)
    def identifier(self, identifier):
        return str(identifier)

    @v_args(inline=True)
    def base(self, base):
        return str(base)[1]

    @v_args(inline=True)
    def identifier_sliced(self, name, range: Range):
        return IdentifierSliced(name, range)

    @v_args(inline=True)
    def identifier_indexed(self, name, index):
        return IdentifierIndexed(name, index)

    @v_args(inline=True)
    def named_port_connection(self, port_name: str, expression):
        return {port_name: expression}

    @v_args(inline=True)
    def assignment(self, left, right):
        return left, right

    def list_of_assignments(self, args) -> List:
        return list(args[0])

    def continuous_assign(self, assignments):
        return ContinuousAssign(assignments)

    @v_args(inline=True)
    def module(self, module_name, list_of_ports, *module_items):
        # TODO: What happens if list_of_ports is not present?
        items = []
        for it in module_items:
            if isinstance(it, list):
                items.extend(it)
            else:
                items.append(it)

        return Module(module_name, list_of_ports, items)

    @v_args(inline=True)
    def module_instantiation(self, module_name, *module_instances) -> List[ModuleInstance]:
        instances = []
        for module_instance in module_instances:
            instance_name, ports = module_instance
            instances.append(ModuleInstance(module_name, instance_name, ports))

        return instances

    def net_declaration(self, args) -> List[NetDeclaration]:

        if len(args) > 0 and isinstance(args[0], Range):
            _range = args[0]
            variable_names = args[1:]
        else:
            _range = None
            variable_names = args

        declarations = []
        for name in variable_names:
            declarations.append(NetDeclaration(name, _range))
        return declarations

    def output_declaration(self, args) -> List[OutputDeclaration]:

        if len(args) > 0 and isinstance(args[0], Range):
            _range = args[0]
            variable_names = args[1:]
        else:
            _range = None
            variable_names = args

        declarations = []
        for name in variable_names:
            declarations.append(OutputDeclaration(name, _range))
        return declarations

    def input_declaration(self, args) -> List[InputDeclaration]:

        if len(args) > 0 and isinstance(args[0], Range):
            _range = args[0]
            variable_names = args[1:]
        else:
            _range = None
            variable_names = args

        declarations = []
        for name in variable_names:
            declarations.append(InputDeclaration(name, _range))
        return declarations

    def list_of_module_connections(self, module_connections):
        connections = dict()
        for conn in module_connections:
            connections.update(**conn)
        return connections

    @v_args(inline=True)
    def module_instance(self, instance_name, module_connections):
        return (instance_name, module_connections)

    @v_args(inline=True)
    def range(self, start, end):
        return Range(start, end)

    @v_args(inline=True)
    def number(self, string):
        return Number(None, None, string)

    @v_args(inline=True)
    def number_explicit_length(self, length, base, mantissa):
        length = int(length)
        return Number(length, base, mantissa)

    @v_args(inline=True)
    def number_implicit_length(self, base, mantissa):
        return Number(None, base, mantissa)

    def concatenation(self, l) -> Concatenation:
        result = []
        for x in l:
            if isinstance(x, Concatenation):
                result.extend(x.elements)
            else:
                result.append(x)
        return Concatenation(result)

    def start(self, description):
        if isinstance(description, list):
            return Netlist(description)
        else:
            return Netlist([description])


def parse_verilog(data: str) -> Netlist:
    """
    Parse a string containing data of a verilog file.
    :param data: Raw verilog string.
    :return:
    """
    verilog_parser = Lark(verilog_netlist_grammar,
                          parser='lalr',
                          lexer='standard',
                          transformer=VerilogTransformer()
                          )
    netlist = verilog_parser.parse(data)

    assert isinstance(netlist.modules, list)

    return netlist


def test_parse_verilog1():
    data = r"""
module blabla(port1, port_2);
    input [0:1234] asdf;
    output [1:3] qwer;
    wire [1234:45] mywire;

    assign a = b;

    assign {a, b[1], c[0: 39]} = {x, y[5], z[1:40]};
    assign {a, b[1], c[0: 39]} = {x, y[5], 1'h0 };
    (* asdjfasld ajsewkea 3903na ;lds *)
    wire zero_set;
    OR _blabla_ ( .A(netname), .B (qwer) );
    OR blabla2 ( .A(netname), .B (1'b0) );

wire zero_res;
  (* src = "alu_shift.v:23" *)
  wire zero_set;
  NOT _072_ (
    .A(func_i[2]),
    .Y(_008_)
  );

endmodule
"""

    netlist = parse_verilog(data)
    # print(netlist.pretty())


def test_parse_verilog2():
    from . import test_data

    data = test_data.verilog_netlist()

    netlist = parse_verilog(data)
    # print(netlist)
    # print(netlist.pretty())
