import utils
import pandas as pd
import numpy as np
import argparse
import pprint as pp



def main(args):
    if int(args['flow_length'])==10:
        ff=pd.read_csv('flow_10.csv',header=None)
    elif int(args['flow_length'])==15:
        ff=pd.read_csv('flow_15.csv',header=None)
    elif int(args['flow_length'])==20:
        ff=pd.read_csv('flow_20.csv',header=None)
    else:
        ff=pd.read_csv('flow_25.csv',header=None)

    ff=np.array(ff)
    Delays=[0 for i in range(len(ff))]
    Areas=[0 for i in range(len(ff))]
    # mark failure flows
    e=[]
    for i in range(len(ff)):
        stat=utils.run_abc(str(args['input']),str(ff[i][0]))
        if stat == None:
            e.append(i)
            continue
        delay, area = utils.get_metrics(stat)
        Delays[i]=delay
        Areas[i]=area

        if np.mod(i,100)==0:
            print(i)

        if np.mod(i+1,10000)==0 and i!=len(ff):
            result_area=pd.DataFrame(Areas)
            result_delay=pd.DataFrame(Delays)

            filename=str(args['input']).split('/')[-1].split('.')[0]
            result_area.to_csv('area_ground_truth_'+filename+'_flow_'+str(args['flow_length'])+'_part_'+str(int(i/10000))+'.csv',index=False,header=False)
            result_delay.to_csv('delay_ground_truth_'+filename+'_flow_'+str(args['flow_length'])+'_part_'+str(int(i/10000))+'.csv',index=False,header=False)
    
    result_area=pd.DataFrame(Areas)
    result_delay=pd.DataFrame(Delays)

    filename=str(args['input']).split('/')[-1].split('.')[0]
    result_area.to_csv('area_ground_truth_'+filename+'_flow_'+str(args['flow_length'])+'.csv',index=False,header=False)
    result_delay.to_csv('delay_ground_truth_'+filename+'_flow_'+str(args['flow_length'])+'.csv',index=False,header=False)
    if len(e)>0:
        error=pd.DataFrame(e)
        error.to_csv('failures_'+filename+'_flow_'+str(args['flow_length'])+'.csv',index=False,header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for the actor critic')

    parser.add_argument('--flow-length', help='the number of optimizations in each synthesis flow', default=10)
    parser.add_argument('--input', help='the name of input verilog file', default='epfl/div.v')
    
    args = vars(parser.parse_args())
    pp.pprint(args)
    main(args)
