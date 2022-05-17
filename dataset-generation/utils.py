import os
import re
import datetime
import numpy as np
from subprocess import check_output

#abc_binary='./abc'
#abc_command='read adder.v; strash;rw; rw; rf; rfz; b; rf; b; rw; rfz;  read OCL.lib;map -v;ps;'
#proc = check_output([abc_binary, '-c', abc_command])


def run_abc(input_file, command):
    abc_binary='./abc'
    abc_command='read '+input_file+'; strash; '+command+' read 7nm_lvt_ff.lib; map -v; ps;'
    try:
        proc = check_output([abc_binary, '-c', abc_command])
        return proc
    except Exception as e:
        return None


# parse delay and area from the stats command of ABC
def get_metrics(stats):
    lines = stats.decode("utf-8").split('\n')
    for i in range(len(lines)):
        if len(lines[i])>5 and lines[i][:5]=='Area ':
            break
    line=lines[i+1].split(':')[-1].strip()
    #print(line)
    ob = re.search(r'Delay *= *[0-9]+.?[0-9]*', line)
    delay = float(ob.group().split('=')[1].strip())
        
    ob = re.search(r'Area *= *[0-9]+.?[0-9]*', line)
    area = float(ob.group().split('=')[1].strip())

    return delay, area

# parse delay, area, and more stats from the stats command of ABC
def get_cnn_metrics(stats):
    lines = stats.decode("utf-8").split('\n')
    
    line = lines[-2]

    ob = re.search(r'delay *= *[0-9]+.?[0-9]*', line)
    delay = float(ob.group().split('=')[1].strip())
        
    ob = re.search(r'area *= *[0-9]+.?[0-9]*', line)
    area = float(ob.group().split('=')[1].strip())

    ob = re.search(r'nd *= *[0-9]*', line)
    nd = int(ob.group().split('=')[1].strip())

    ob = re.search(r'edge *= *[0-9]*', line)
    edge = int(ob.group().split('=')[1].strip())

    ob = re.search(r'lev *= *[0-9]*', line)
    lev = int(ob.group().split('=')[1].strip())

    ob = re.search(r'i/o *= *[0-9]* */ *[0-9]*', line)
    io = ob.group().split('=')[1].strip().split('/')
    i = int(io[0])
    o = int(io[1])

    return delay, area, edge, nd, lev, i, o
