from mpi4py import MPI


import numpy as np
import core
import glob
files = glob.glob("./pairscount/pairs/*")
file_l = np.array(files)[np.array(["_j.pkl" in file for file in files])]
files_run = np.array(files)[~np.array(["_j.pkl" in file for file in files])]
file_to_run = file_l[~np.in1d(np.array([file.split("_j.pkl")[0] for file in file_l]),np.array([file.split(".pkl")[0] for file in files_run]))]


import pickle
def save_obj( name,obj ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

number_of_works=len(file_to_run)
print number_of_works
if 1==1:
        list_run= np.array(number_of_works)
        run_count=0
        while run_count<(number_of_works):
            comm = MPI.COMM_WORLD

            print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
            name= file_to_run[run_count+comm.rank].split(".pkl")[0]
            J = load_obj(name)
            
            pairs = J.NNCorrelation()
                    
            save_obj(name.split("_j")[0],pairs)
          
      
            run_count+=comm.size
            comm.bcast(run_count,root = 0)
            comm.Barrier() 
