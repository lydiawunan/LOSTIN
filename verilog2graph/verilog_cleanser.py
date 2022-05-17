### remove / and [] in verilog files, so as to be compatible with the parser
path='epfl/'
new_path='epfl_new/'
filename=['adder.v', 'arbiter.v', 'bar.v', 'div.v', 'log2.v', 'max.v', 'multiplier.v', 'sin.v', 'sqrt.v', 'square.v', 'voter.v']

for fname in filename: 
    # read verilog file
    f = open(path + fname, "r")
    design = f.readlines()

    # remove / and [] in file
    new_design = []
    for i in range(len(design)):
        line = design[i].replace('[','').replace(']','').replace('\\','')
        new_design.append(line)

    # save modified designs
    new_fname = 'new_'+fname
    f = open(new_path+new_fname, "w")
    f.writelines(new_design)
