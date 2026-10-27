################################################################################
# IMPORT LIBRARIES
################################################################################

#import os
import sys
import time
import pygad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Increase the recursion limit to avoid: "RecursionError: maximum recursion depth exceeded in comparison"
rec_lim=50000
if sys.getrecursionlimit()<rec_lim:
    sys.setrecursionlimit(rec_lim)

### Suppress some warnings
# import warnings
# warnings.filterwarnings("ignore", message="p-value capped")
# warnings.filterwarnings("ignore", message="p-value floored")

### Plots
#import seaborn as sns
#sns.set_style('darkgrid')
#from matplotlib import rc
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('text', usetex=True)
save_plot=0

random_seed=None
#random_seed=42
#np.random.seed(random_seed)

print_time=True

################################################################################
# SET PATHS
################################################################################

### Set the username for the path of the files:
user={'bach', 'gravity'}
#user='LB'
#user='AF'
#user='bach'
#user='gravity'
#user='pleiadi'

user = user if isinstance(user, set) else {user}
if 'bach' in user:
    # library paths
    sys.path.append('/home/bazzanini/PYTHON/genetic/lc_pulse_avalanche/statistical_test')
    sys.path.append('/home/bazzanini/PYTHON/genetic/lc_pulse_avalanche/lc_pulse_avalanche')
    # real data
    batse_path = '/astrodata/guidorzi/CGRO_BATSE/'
    swift_path = '/astrodata/guidorzi/Swift_BAT/'
    sax_path   = '/astrodata/guidorzi/BeppoSAX_GRBM/'
if 'gravity' in user:
    # library paths
    sys.path.append('/home/bazzanini/PYTHON/genetic3/statistical_test')
    sys.path.append('/home/bazzanini/PYTHON/genetic3/lc_pulse_avalanche')
    # real data
    batse_path = '/astrodata/guidorzi/CGRO_BATSE/'
    swift_path = '/astrodata/guidorzi/Swift_BAT/'
    sax_path   = '/astrodata/guidorzi/BeppoSAX_GRBM/'
if 'pleiadi' in user:
    # library paths
    sys.path.append('/beegfs/mbulla/genetic_grbs/genetic/lc_pulse_avalanche/statistical_test')
    sys.path.append('/beegfs/mbulla/genetic_grbs/genetic/lc_pulse_avalanche/lc_pulse_avalanche')
    # real data
    batse_path = '/beegfs/mbulla/genetic_grbs/CGRO_BATSE/'
    swift_path = '/beegfs/mbulla/genetic_grbs/Swift_BAT/'
    sax_path   = '/beegfs/mbulla/genetic_grbs/BeppoSAX_GRBM/'
if 'LB' in user:
    # library paths
    sys.path.append('/Users/lorenzo/Documents/UNIVERSITA/Astrophysics/PYTHON/GRBs/lc_pulse_avalanche/statistical_test')
    sys.path.append('/Users/lorenzo/Documents/UNIVERSITA/Astrophysics/PYTHON/GRBs/lc_pulse_avalanche/lc_pulse_avalanche')
    # real data
    batse_path = '/Users/lorenzo/Documents/UNIVERSITA/Astrophysics/PYTHON/DATA/CGRO_BATSE/'
    swift_path = '/Users/lorenzo/Documents/UNIVERSITA/Astrophysics/PYTHON/DATA/Swift_BAT/'
    sax_path   = '/Users/lorenzo/Documents/UNIVERSITA/Astrophysics/PYTHON/DATA/BeppoSAX_GRBM/'
if 'AF' in user:
    # libraries
    sys.path.append('C:/Users/Lisa/Documents/GitHub/lc_pulse_avalanche/statistical_test')
    sys.path.append('C:/Users/Lisa/Documents/GitHub/lc_pulse_avalanche/lc_pulse_avalanche')
    # real data
    batse_path = 'E:/grb_to_test/CGRO_BATSE/'
    swift_path = 'E:/grb_to_test/Swift_BAT/'
    sax_path   = 'E:/grb_to_test/BeppoSAX_GRBM/'
if not user & {'bach', 'gravity', 'pleiadi', 'LB', 'AF'}:
    raise ValueError('Assign to the variable "user" a correct username!')

# Always include local relative paths as fallback
import os as _os
sys.path.append(_os.path.abspath('../statistical_test'))
sys.path.append(_os.path.abspath('../lc_pulse_avalanche'))

from statistical_test import *
from avalanche import LC


################################################################################
# SET PARAMETERS
################################################################################

### Choose the instrument
#instrument = 'batse'
instrument = 'swift'
#instrument = 'sax'
#instrument = 'fermi'

#------------------------------------------------------------------------------#

if instrument=='batse':
    t_i           = 0                            # [s]
    t_f           = 150                          # [s]
    eff_area      = instr_batse['eff_area']      # 2025  # effective area of instrument [cm2]
    bg_level      = instr_batse['bg_level']      # 2.8   # background level [cnt/cm2/s]
    t90_threshold = instr_batse['t90_threshold'] # 2     # [s] --> used to select only _long_ GRBs
    t90_frac      = 15
    sn_threshold  = instr_batse['sn_threshold']  # 70    # signal-to-noise ratio
    bin_time      = instr_batse['res']           # 0.064 # [s] temporal bins for BATSE (time resolution)
    test_times    = np.linspace(t_i, t_f, int((t_f-t_i)/bin_time))
elif instrument=='swift':
    t_i           = 0                            # [s]
    t_f           = 150                          # [s]
    eff_area      = instr_swift['eff_area']      # 1400 # effective area of instrument [cm2]
    bg_level      = instr_swift['bg_level']      # (10000/eff_area) # background level [cnt/cm2/s]
    t90_threshold = instr_swift['t90_threshold'] # 2 # [s] --> used to select only _long_ GRBs
    t90_frac      = 15
    sn_threshold  = instr_swift['sn_threshold']  # 15 # signal-to-noise ratio
    bin_time      = instr_swift['res']           # 0.064 # [s] temporal bins for Swift (time resolution)
    test_times    = np.linspace(t_i, t_f, int((t_f-t_i)/bin_time))
elif instrument=='sax':
    t_i           = 0                          # [s]
    t_f           = 50                         # [s] (HR)
    eff_area      = instr_sax['eff_area']      # 420 # effective area of instrument [cm2]
    bg_level      = instr_sax['bg_level']      # (1000/eff_area) # background level [cnt/cm2/s]
    t90_threshold = instr_sax['t90_threshold'] # 2 # [s] --> used to select only _long_ GRBs
    t90_frac      = 15
    sn_threshold  = instr_sax['sn_threshold']  # 10 # signal-to-noise ratio
    bin_time      = instr_sax['res']           # 0.0078125 # [s] temporal bins for BeppoSAX (HR)
    test_times    = np.linspace(t_i, t_f, int((t_f-t_i)/bin_time))
elif instrument=='sax_lr':
    t_i           = 0                              # [s]
    t_f           = 150                            # [s] (LR)
    eff_area      = instr_sax_lr['eff_area']       # 420 # effective area of instrument [cm2]
    bg_level      = instr_sax_lr['bg_level']       # (1000/eff_area) # background level [cnt/cm2/s]
    t90_threshold = instr_sax_lr['t90_threshold']  # 2 # [s] --> used to select only _long_ GRBs
    t90_frac      = 15
    sn_threshold  = instr_sax_lr['sn_threshold']   # 10   # signal-to-noise ratio
    bin_time      = instr_sax_lr['res']            # 1.0  # [s] temporal bins for BeppoSAX (LR)
    test_times    = np.linspace(t_i, t_f, int((t_f-t_i)/bin_time))
elif instrument=='fermi':
    t_i           = 0                            # [s]
    t_f           = 150                          # [s]
    eff_area      = instr_fermi['eff_area']      # 100 # effective area of instrument [cm2]
    bg_level      = instr_fermi['bg_level']      # (400/eff_area_fermi) # background level [cnt/cm2/s]
    t90_threshold = instr_fermi['t90_threshold'] # 2 # [s] --> used to select only _long_ GRBs
    t90_frac      = 15
    sn_threshold  = instr_fermi['sn_threshold']  # 5 # signal-to-noise ratio
    bin_time      = instr_fermi['res']           # 0.064 # [s] temporal bins for Fermi
    test_times    = np.linspace(t_i, t_f, int((t_f-t_i)/bin_time))
# elif instrument=='batse_old': # it is actually for 'galileo'
#     t_i           = 0     # [s]
#     t_f           = 150   # [s]
#     eff_area      = 3600  # effective area of instrument [cm2]
#     bg_level      = 10.67 # background level [cnt/cm2/s]
#     t90_threshold = 2     # [s] --> used to select only _long_ GRBs
#     t90_frac      = 15
#     sn_threshold  = 70    # signal-to-noise ratio
#     bin_time      = 0.064 # [s] temporal bins for BATSE (time resolution)
#     test_times    = np.linspace(t_i, t_f, int((t_f-t_i)/bin_time))
else:
    raise NameError('Variable "instrument" not defined properly; choose between: "batse", "swift", "sax", "sax_lr", and "fermi".')

#------------------------------------------------------------------------------#

# Genetic Algorithm parameters
parent_selection_type = "tournament" 
crossover_probability = 1                      # 'None' means couples parent k with parent k+1, otherwise it selects from the parents candidate list each one of them with probability 'crossover_probability', and then it takes two of them at random
initial_population    = None                   # if 'None', the initial population is randomly chosen using the 'sol_per_pop; and 'num_genes' parameters
mutation_type         = "random"
crossover_type        = "scattered"
num_generations       = 15                     # Number of generations.
sol_per_pop           = 2000                   # Number of solutions in the population (i.e., number of different sets per generation).
num_parents_mating    = int(0.15*sol_per_pop)  # Number of solutions to be selected as parents in the mating pool.
keep_parents          = 0                      # if 0, keep NO parents (the ones selected for mating in the current population) in the next population
keep_elitism          = 0                      # keep in the next generation the best N solution of the current generation
mutation_probability  = 0.04                   # by default is 'None', otherwise it selects a value randomly from the current gene's space (each gene is changed with probability 'mutation_probability')

# Other parameters
N_grb            = 10   # number of simulated GRBs to produce per set of parameters
n_cut            = 2500   # maximum number of pulses to consider in the avalanche model
test_pulse_distr = False  # add a fifth metric regarding the distribution of number of pulses per GRB (set False by default)

# Options for parallelization
if user=='pleiadi':
    n_processes = int(os.environ['OMP_NUM_THREADS'])
else:
    n_processes = 100
parallel_processing  = ["process", n_processes]  # USE THIS ONE!  
#parallel_processing = ["thread", n_processes]   # this is slower
#parallel_processing = None                      # single thread

# Name of the pkl file where to save the GA instance at the end of the run
filename_model = 'geneticGRB'

# We impose constraints on the range of values that the 7 parameter can assume
range_mu      = {"low": 0.80,            "high": 1.7}
range_mu0     = {"low": 0.80,            "high": 1.7} 
range_alpha   = {"low": 1,               "high": 15} 
range_delta1  = {"low": -1.5,            "high": -0.30-1.e-6} 
range_delta2  = {"low": 0,               "high": 0.30}
range_tau_min = {"low": np.log10(1.e-2), "high": np.log10(bin_time-1.e-6)}  # sample `tau_min` uniformly in log    space
#range_tau_min = {"low": 1.e-2,          "high": bin_time-1.e-6}            # sample `tau_min` uniformly in linear space
range_tau_max = {"low": 1,               "high": 60}
# The values of the 7 parameters from the paper [Stern & Svensson, 1996] are:
# mu=1.2
# mu0=1
# alpha=4
# delta1=-0.5
# delta2=0
# tau_min=0.02
# tau_max=26

range_constraints = [range_mu, 
                     range_mu0,
                     range_alpha,
                     range_delta1, 
                     range_delta2, 
                     range_tau_min, 
                     range_tau_max]

num_genes = len(range_constraints) # 7

save_model = True

print('\n\n')
print('################################################################################')
print('START')
print('################################################################################')
print('\n\n')


################################################################################
# LOAD REAL DATA
################################################################################

init_load_time = time.perf_counter()

### Load the BATSE GRBs
if instrument=='batse':
    # load all data
    grb_list_real = load_lc_batse(path=batse_path)
    # apply constraints
    grb_list_real = apply_constraints(grb_list=grb_list_real,
                                      bin_time=bin_time,
                                      t90_threshold=t90_threshold,
                                      t90_frac=t90_frac,
                                      sn_threshold=sn_threshold,
                                      t_f=t_f)
    # Load MEPSA results on BATSE (ONLY those that satisfy the constraint!)
    mepsa_out_file_list_temp = []
    for i in range(len(grb_list_real)):
        name = grb_list_real[i].name
        mepsa_out_file_list_temp.append(name)
    reb_factor          = np.inf
    peak_sn_level       = 10
    mepsa_out_file_list = [ batse_path+'PEAKS_ALL/peaks_'+el+'_all_bs_2.txt' for el in mepsa_out_file_list_temp ]
    if test_pulse_distr:
        n_of_pulses_real = readMEPSAres(mepsa_out_file_list=mepsa_out_file_list, # mepsa results on BATSE data
                                        maximum_reb_factor=reb_factor, 
                                        sn_level=peak_sn_level)
    else:
        n_of_pulses_real = None

### Load the Swift GRBs
elif instrument=='swift': 
    # load all data
    grb_list_real = load_lc_swift(path=swift_path)
    # apply constraints
    grb_list_real = apply_constraints(grb_list=grb_list_real, 
                                      bin_time=bin_time, 
                                      t90_threshold=t90_threshold,
                                      t90_frac=t90_frac, 
                                      sn_threshold=sn_threshold, 
                                      t_f=t_f)
    n_of_pulses_real = None

### Load the BeppoSAX GRBs
elif instrument=='sax': 
    # load all (HR) data
    grb_list_real = load_lc_sax_hr(path=sax_path) 
    # apply constraints
    grb_list_real = apply_constraints(grb_list=grb_list_real, 
                                      bin_time=bin_time, 
                                      t90_threshold=t90_threshold, 
                                      t90_frac=t90_frac,
                                      sn_threshold=sn_threshold, 
                                      t_f=t_f)
    n_of_pulses_real = None

else:
    raise NameError('Variable "instrument" not defined properly; choose between: "batse", "swift", "sax".')

end_load_time = time.perf_counter()
print('\n')
print('--------------------------------------------------------------------------------')
print('* {} data loaded in {:0.0f} sec'.format(instrument,(end_load_time-init_load_time)))
print('--------------------------------------------------------------------------------')


################################################################################
# COMPUTE AVERAGE QUANTITIES OF REAL DATA
################################################################################

### TEST 1&2: Average Peak-Aligned Profiles
averaged_fluxes_real, \
averaged_fluxes_cube_real, \
averaged_fluxes_rms_real = compute_average_quantities(grb_list=grb_list_real,
                                                      t_f=t_f,
                                                      bin_time=bin_time,
                                                      filter=True)
### TEST 3: Autocorrelation
# For the REAL LCs we use the Link+93 formula to compute the autocorrelation,
# whereas for the simulated LCs instead we use the scipy.signal.correlate()
# function on the model curve, i.e., the one before adding the Poisson noise.
# N_lim = np.min( [N_grb, len(grb_list_real)] )
steps_real, acf_real = compute_autocorrelation(grb_list=grb_list_real,
                                               N_lim=len(grb_list_real),
                                               t_max=t_f,
                                               bin_time=bin_time,
                                               mode='link93',
                                               compute_rms=False)

### TEST 4: Duration
duration_real = [ evaluateDuration20(times=grb.times,
                                     counts=grb.counts,
                                     filter=True,
                                     t90=grb.t90,
                                     bin_time=bin_time)[0] for grb in grb_list_real ]
duration_distr_real = compute_kde_log_duration(duration_list=duration_real)


################################################################################
# DEFINE FITNESS FUNCTION OF THE GENETIC ALGORITHM
################################################################################

# pygad 3.X
def fitness_func(ga_instance, solution, solution_idx=None):
# pygad 2.X
# def fitness_func(solution, solution_idx=None):
    #--------------------------------------------------------------------------#
    # Generate the GRBs
    #--------------------------------------------------------------------------#
    grb_list_sim = generate_GRBs(# number of simulated GRBs to produce:
                                 N_grb=N_grb,
                                 # 7 parameters:
                                 mu=solution[0],
                                 mu0=solution[1],
                                 alpha=solution[2],
                                 delta1=solution[3],
                                 delta2=solution[4],
                                 tau_min=10**solution[5], # sample `tau_min` uniformly in log    space
                                 #tau_min=solution[5],    # sample `tau_min` uniformly in linear space
                                 tau_max=solution[6],
                                 # instrument parameters:
                                 instrument=instrument,
                                 bin_time=bin_time,
                                 eff_area=eff_area,
                                 bg_level=bg_level,
                                 # constraint parameters:
                                 sn_threshold=sn_threshold,
                                 t90_threshold=t90_threshold,
                                 t90_frac=t90_frac,
                                 t_f=t_f,
                                 filter=True,
                                 # other parameters:
                                 export_files=False,
                                 n_cut=n_cut,
                                 with_bg=False,
                                 test_pulse_distr=test_pulse_distr)
    if test_pulse_distr:
        n_of_pulses_sim = [ grb.num_of_sig_pulses for grb in grb_list_sim ]
    else:
        n_of_pulses_sim = None

    #--------------------------------------------------------------------------#
    # COMPUTE AVERAGE QUANTITIES OF SIMULATED DATA
    #--------------------------------------------------------------------------#
    ### TEST 1&2: Average Peak-Aligned Profiles
    averaged_fluxes_sim, \
    averaged_fluxes_cube_sim, \
    averaged_fluxes_rms_sim = compute_average_quantities(grb_list=grb_list_sim,
                                                         t_f=t_f,
                                                         bin_time=bin_time,
                                                         filter=True)
    ### TEST 3: Autocorrelation
    # For the REAL LCs we use the Link+93 formula to compute the autocorrelation,
    # whereas for the simulated LCs instead we use the scipy.signal.correlate
    # function on the model curve, i.e., the one before adding the Poisson noise.
    steps_sim, acf_sim = compute_autocorrelation(grb_list=grb_list_sim,
                                                 N_lim=N_grb,
                                                 t_max=t_f,
                                                 bin_time=bin_time,
                                                 mode='scipy',
                                                 compute_rms=False)
    ### TEST 4: Duration
    #duration_sim = [ evaluateDuration20(times=grb.times, 
    #                                    counts=grb.counts,
    #                                    filter=True,
    #                                    t90=grb.t90,
    #                                    bin_time=bin_time)[0] for grb in grb_list_sim ]
    duration_sim       = np.array( [ grb.t20 for grb in grb_list_sim ] )
    duration_distr_sim = compute_kde_log_duration(duration_list=duration_sim)

    #--------------------------------------------------------------------------#
    # COMPUTE LOSS
    #--------------------------------------------------------------------------#
    l2_loss = compute_loss(averaged_fluxes=averaged_fluxes_real,
                           averaged_fluxes_sim=averaged_fluxes_sim,
                           averaged_fluxes_cube=averaged_fluxes_cube_real,
                           averaged_fluxes_cube_sim=averaged_fluxes_cube_sim,
                           acf=acf_real,
                           acf_sim=acf_sim,
                           duration=duration_distr_real,
                           duration_sim=duration_distr_sim,
                           n_of_pulses=n_of_pulses_real,
                           n_of_pulses_sim=n_of_pulses_sim,
                           test_pulse_distr=test_pulse_distr)
    fitness = 1.0 / (l2_loss + 1.e-9)
    return fitness


################################################################################
# DEFINE AUXILIARY FUNCTION
################################################################################

def write_best_par_per_epoch(solution, filename='best_par_per_epoch.txt'):
    """
    Function to write the best parameters of each generation in a file. The file
    is opened in append mode, so that we can append the results of each generation
    at the end of the file at each epoch.
    Parameters:
    - solution: array containing the best solution (7 params) of a generation.
    - filename: The name of the file to open in append mode. Default is 'output.txt'.
    """
    with open(filename, 'a') as file:
        file.write("mu      = {solution}".format(solution=solution[0])+'\n')
        file.write("mu0     = {solution}".format(solution=solution[1])+'\n')
        file.write("alpha   = {solution}".format(solution=solution[2])+'\n')
        file.write("delta1  = {solution}".format(solution=solution[3])+'\n')
        file.write("delta2  = {solution}".format(solution=solution[4])+'\n')
        file.write("tau_min = {solution}".format(solution=10**(solution[5]))+'\n') # sample `tau_min` uniformly in log    space
        #file.write("tau_min = {solution}".format(solution=solution[5])+'\n')      # sample `tau_min` uniformly in linear space
        file.write("tau_max = {solution}".format(solution=solution[6])+'\n')
        file.write('\n')


last_fitness, last_loss, current_fitness, current_loss = 0, 0, 0, 0
def on_generation(ga_instance):
    """
    This function is executed after each generation. It prints useful 
    information of the current epoch, in particular, the details of the best
    solution in the current generation.
    """
    global last_fitness, last_loss, current_fitness, current_loss
    print('--------------------------------------------------------------------------------')
    print("Generation     = {generation}".format(generation=ga_instance.generations_completed))
    current_fitness       = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
    current_loss          = current_fitness**(-1)                
    print("Best Loss      = {solution_loss}".format(solution_loss=current_loss))
    print("Best Fitness   = {fitness}".format(fitness=current_fitness))
    print("Fitness Change = {change}".format(change=current_fitness-last_fitness))
    last_fitness          = current_fitness
    last_loss             = current_loss
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    # Print the best solution of the current generation on TERMINAL
    print("Parameters of the best solution in the current generation:")
    print("    - mu      = {solution}".format(solution=solution[0]))
    print("    - mu0     = {solution}".format(solution=solution[1]))
    print("    - alpha   = {solution}".format(solution=solution[2]))
    print("    - delta1  = {solution}".format(solution=solution[3]))
    print("    - delta2  = {solution}".format(solution=solution[4]))
    print("    - tau_min = {solution}".format(solution=10**(solution[5]))) # sample `tau_min` uniformly in log    space
    #print("    - tau_min = {solution}".format(solution=solution[5]))      # sample `tau_min` uniformly in linear space
    print("    - tau_max = {solution}".format(solution=solution[6]))      
    # Print the best solution of the current generation on FILE
    write_best_par_per_epoch(solution)


################################################################################
# INSTANTIATE THE 'GENETIC ALGORITHM' CLASS
################################################################################

if __name__ == '__main__':
    # Usage: 
    #     - to run it the first time, type in the terminal: `python testGA.py`
    #     - to continue the run, type in the terminal:      `python testGA.py -c`


    # print(f"Arguments count: {len(sys.argv)}")
    # for i, arg in enumerate(sys.argv):
    #     print(f"Argument {i:>6}: {arg}")

    if len(sys.argv)==1: # python testGA.py
        print(f"\n[INFO] Running '{sys.argv[0]}'...\n")
        MODE = 'first'
    elif '-c' in sys.argv: # python testGA.py -c
        print(f"\n[INFO] Resuming '{sys.argv[0]}' from checkpoint...\n")
        MODE = 'resume'
    else:
        raise ValueError(f"Unrecognized command line argument(s): {sys.argv[1:]}")
    
    if MODE=='first':
        ga_GRB = pygad.GA(num_generations=num_generations,
                          num_parents_mating=num_parents_mating,
                          sol_per_pop=sol_per_pop,
                          num_genes=num_genes,
                          gene_type=float,
                          initial_population=initial_population,
                          on_generation=on_generation,
                          ### fitness function:
                          fitness_func=fitness_func,
                          ### parent selection:
                          parent_selection_type=parent_selection_type,
                          keep_parents=keep_parents,           
                          keep_elitism=keep_elitism,           
                          ### crossover:
                          crossover_probability=crossover_probability,
                          crossover_type=crossover_type,
                          ### mutation:
                          mutation_type=mutation_type,
                          mutation_probability=mutation_probability,     
                          ### set range of parameters:
                          gene_space=range_constraints,
                          ### other stuff:
                          save_best_solutions=True,
                          save_solutions=True,
                          parallel_processing=parallel_processing,
                          random_seed=random_seed)
    
    elif MODE=='resume':
        # Load the saved GA instance
        ga_GRB = pygad.load(filename=filename_model)

        # Reload the fitness function (otherwise it will raise an error, I don't know why...)
        # pygad 3.X
        def fitness_func_reloaded(ga_instance, solution, solution_idx=None):
        # pygad 2.X
        # def fitness_func_reloaded(solution, solution_idx=None):
            # global loss_list
            #--------------------------------------------------------------------------#
            # Generate the GRBs
            #--------------------------------------------------------------------------#
            grb_list_sim = generate_GRBs(# number of simulated GRBs to produce:
                                         N_grb=N_grb,
                                         # 7 parameters:
                                         mu=solution[0],
                                         mu0=solution[1],
                                         alpha=solution[2],
                                         delta1=solution[3],
                                         delta2=solution[4],
                                         tau_min=10**solution[5],  # sample `tau_min` uniformly in log    space
                                         #tau_min=solution[5],     # sample `tau_min` uniformly in linear space
                                         tau_max=solution[6],
                                         # instrument parameters:
                                         instrument=instrument,
                                         bin_time=bin_time,
                                         eff_area=eff_area,
                                         bg_level=bg_level,
                                         # constraint parameters:
                                         sn_threshold=sn_threshold,
                                         t90_threshold=t90_threshold,
                                         t90_frac=t90_frac,
                                         t_f=t_f,
                                         filter=True,
                                         # other parameters:
                                         export_files=False,
                                         n_cut=n_cut,
                                         with_bg=False,
                                         test_pulse_distr=test_pulse_distr)
            if test_pulse_distr:
                n_of_pulses_sim = [ grb.num_of_sig_pulses for grb in grb_list_sim ]
            else:
                n_of_pulses_sim = None

            #--------------------------------------------------------------------------#
            # Compute average quantities of simulated data needed for the loss function
            #--------------------------------------------------------------------------#
            ### TEST 1&2: Average Peak-Aligned Profiles
            averaged_fluxes_sim, \
            averaged_fluxes_cube_sim, \
            averaged_fluxes_rms_sim = compute_average_quantities(grb_list=grb_list_sim,
                                                                 t_f=t_f,
                                                                 bin_time=bin_time,
                                                                 filter=True)
            ### TEST 3: Autocorrelation
            # For the REAL LCs we use the Link+93 formula to compute the autocorrelation,
            # whereas for the simulated LCs instead we use the scipy.signal.correlate
            # function on the model curve, i.e., the one before adding the Poisson noise.
            steps_sim, acf_sim = compute_autocorrelation(grb_list=grb_list_sim,
                                                         N_lim=N_grb,
                                                         t_max=t_f,
                                                         bin_time=bin_time,
                                                         mode='scipy',
                                                         compute_rms=False)
            ### TEST 4: Duration
            #duration_sim = [ evaluateDuration20(times=grb.times, 
            #                                    counts=grb.counts,
            #                                    filter=True,
            #                                    t90=grb.t90,
            #                                    bin_time=bin_time)[0] for grb in grb_list_sim ]
            duration_sim       = np.array( [ grb.t20 for grb in grb_list_sim ] )
            duration_distr_sim = compute_kde_log_duration(duration_list=duration_sim)

            #--------------------------------------------------------------------------#
            # Compute loss
            #--------------------------------------------------------------------------#
            l2_loss = compute_loss(averaged_fluxes=averaged_fluxes_real,
                                   averaged_fluxes_sim=averaged_fluxes_sim,
                                   averaged_fluxes_cube=averaged_fluxes_cube_real,
                                   averaged_fluxes_cube_sim=averaged_fluxes_cube_sim,
                                   acf=acf_real,
                                   acf_sim=acf_sim,
                                   duration=duration_distr_real,
                                   duration_sim=duration_distr_sim,
                                   n_of_pulses=n_of_pulses_real,
                                   n_of_pulses_sim=n_of_pulses_sim,
                                   test_pulse_distr=test_pulse_distr)
            fitness = 1.0 / (l2_loss + 1.e-9)
            return fitness


        ga_GRB.fitness_func = fitness_func_reloaded

    # print summary of the GA
    ga_GRB.summary() 

    ############################################################################
    # RUN THE GENETIC ALGORITHM
    ############################################################################

    init_run_time = time.perf_counter()
    print('\nStarting the GA...\n')
    ga_GRB.run()
    #ga_GRB.plot_fitness()
    end_run_time = time.perf_counter()

    print('\n')
    print('--------------------------------------------------------------------------------')
    print('* Model run in {:0.0f} sec'.format((end_run_time-init_run_time)))
    print('--------------------------------------------------------------------------------')


    ############################################################################
    # SAVE THE MODEL
    ############################################################################

    ### Save the GA instance
    if save_model:
        ga_GRB.save(filename=filename_model)


    ############################################################################
    # PRINT FINAL RESULTS
    ############################################################################

    #--------------------------------------------------------------------------#
    # Print on terminal
    #--------------------------------------------------------------------------#
    solution, solution_fitness, solution_idx = ga_GRB.best_solution(ga_GRB.last_generation_fitness)
    print('\n################################################################################')
    print('################################################################################')
    print("* Parameters of the BEST solution:")
    print("    - mu      = {solution}".format(solution=solution[0]))
    print("    - mu0     = {solution}".format(solution=solution[1]))
    print("    - alpha   = {solution}".format(solution=solution[2]))
    print("    - delta1  = {solution}".format(solution=solution[3]))
    print("    - delta2  = {solution}".format(solution=solution[4]))
    print("    - tau_min = {solution}".format(solution=10**(solution[5]))) # sample `tau_min` uniformly in log    space
    #print("    - tau_min = {solution}".format(solution=solution[5]))      # sample `tau_min` uniformly in linear space
    print("    - tau_max = {solution}".format(solution=solution[6]))
    print("* Loss value of the best solution    : {solution_loss}".format(solution_loss=solution_fitness**(-1)))
    print("* Fitness value of the best solution : {solution_fitness}".format(solution_fitness=solution_fitness))
    #print("Index of the best solution          : {solution_idx}".format(solution_idx=solution_idx))
    if ga_GRB.best_solution_generation != -1:
        print("* Best fitness value reached after N={best_solution_generation} generations.".format(best_solution_generation=ga_GRB.best_solution_generation))
    print('################################################################################')
    print('################################################################################')
    #--------------------------------------------------------------------------#
    # Print on file
    #--------------------------------------------------------------------------#
    file = open("./simulation_info.txt", "w")
    file.write('################################################################################')
    file.write('\n')
    file.write("INPUT")
    file.write('\n')
    file.write('################################################################################')
    file.write('\n')
    file.write('\n')
    file.write('N_GRBs_per_set       = {}'.format(N_grb))
    file.write('\n')
    file.write('num_generations      = {}'.format(num_generations))
    file.write('\n')
    file.write('sol_per_pop          = {}'.format(sol_per_pop))
    file.write('\n')
    file.write('num_parents_mating   = {}'.format(num_parents_mating))
    file.write('\n')
    file.write('keep_parents         = {}'.format(keep_parents))
    file.write('\n')
    file.write('keep_elitism         = {}'.format(keep_elitism))
    file.write('\n')
    file.write('mutation_probability = {}'.format(mutation_probability))
    file.write('\n')
    file.write('\n')
    file.write('range_mu             = {}'.format(range_mu))
    file.write('\n')
    file.write('range_mu0            = {}'.format(range_mu0))
    file.write('\n')
    file.write('range_alpha          = {}'.format(range_alpha))
    file.write('\n')
    file.write('range_delta1         = {}'.format(range_delta1))
    file.write('\n')
    file.write('range_delta2         = {}'.format(range_delta2))
    file.write('\n')
    file.write('range_tau_min        = {}'.format(range_tau_min))
    file.write('\n')
    file.write('range_tau_max        = {}'.format(range_tau_max))
    file.write('\n')
    file.write('\n')
    file.write('################################################################################')
    file.write('\n')
    file.write("OUTPUT")
    file.write('\n')
    file.write('################################################################################')
    file.write('\n')
    file.write('\n')
    file.write("* Parameters of the BEST solution:")
    file.write('\n')
    file.write("    - mu      = {solution}".format(solution=solution[0]))
    file.write('\n')
    file.write("    - mu0     = {solution}".format(solution=solution[1]))
    file.write('\n')
    file.write("    - alpha   = {solution}".format(solution=solution[2]))
    file.write('\n')
    file.write("    - delta1  = {solution}".format(solution=solution[3]))
    file.write('\n')
    file.write("    - delta2  = {solution}".format(solution=solution[4]))
    file.write('\n')
    file.write("    - tau_min = {solution}".format(solution=10**(solution[5])))  # sample `tau_min` uniformly in log    space
    #file.write("    - tau_min = {solution}".format(solution=solution[5]))       # sample `tau_min` uniformly in linear space
    file.write('\n')
    file.write("    - tau_max = {solution}".format(solution=solution[6]))
    file.write('\n')
    file.write("* Loss value of the best solution    : {solution_loss}".format(solution_loss=solution_fitness**(-1)))
    file.write('\n')
    file.write("* Fitness value of the best solution : {solution_fitness}".format(solution_fitness=solution_fitness))
    file.write('\n')
    #print("Index of the best solution          : {solution_idx}".format(solution_idx=solution_idx))
    if ga_GRB.best_solution_generation != -1:
        file.write("* Best fitness value reached after N = {best_solution_generation} generations.".format(best_solution_generation=ga_GRB.best_solution_generation))
    file.write('\n')
    file.write('\n')
    file.write('################################################################################')
    file.write('\n')
    file.write('################################################################################')
    file.close()
    #--------------------------------------------------------------------------#


    ############################################################################
    # EXPORT DATA FOR THE PLOT 1
    ############################################################################

    if MODE=='first':
        best_loss = np.array(ga_GRB.best_solutions_fitness)**(-1)
        loss_list = np.array(ga_GRB.solutions_fitness)**(-1)
        avg_loss  = np.zeros(len(best_loss))
        std_loss  = np.zeros(len(best_loss))
        for i in range(len(best_loss)):
            avg_loss[i] = np.mean( loss_list[i*sol_per_pop:(i+1)*sol_per_pop] )
            std_loss[i] = np.std(  loss_list[i*sol_per_pop:(i+1)*sol_per_pop] )
        #print('best_loss[-1] =', best_loss[-1])

        datafile = './datafile.txt'
        file = open(datafile, 'w')
        file.write('# generation\t best_loss\t avg_loss\t std_loss\t std_loss/sqrt(sol_per_pop)\n')
        for i in range(len(best_loss)):
            file.write('{0} {1} {2} {3} {4}\n'.format(i, best_loss[i], avg_loss[i], std_loss[i], std_loss[i]/np.sqrt(sol_per_pop)))
        file.close()

    elif MODE=='resume':
        best_loss = np.array(ga_GRB.best_solutions_fitness)**(-1)
        loss_list = np.array(ga_GRB.solutions_fitness)**(-1)
        avg_loss  = np.zeros(len(best_loss))
        std_loss  = np.zeros(len(best_loss))
        best_loss_print = []
        avg_loss_print  = []
        std_loss_print  = []
        for i in range(len(best_loss)):
            avg_loss[i] = np.mean( loss_list[i*sol_per_pop:(i+1)*sol_per_pop] )
            std_loss[i] = np.std(  loss_list[i*sol_per_pop:(i+1)*sol_per_pop] )
        #print('best_loss[-1] =', best_loss[-1])
        for i in range(len(best_loss)):
            if i%(num_generations+1)==0 and i!=0:
                pass
            else:
                best_loss_print.append(best_loss[i])
                avg_loss_print.append(avg_loss[i])
                std_loss_print.append(std_loss[i])              

        datafile = './datafile.txt'
        file = open(datafile, 'w')
        file.write('# generation\t best_loss\t avg_loss\t std_loss\t std_loss/sqrt(sol_per_pop)\n')
        for i in range(len(best_loss_print)):
            file.write('{0} {1} {2} {3} {4}\n'.format(i, best_loss_print[i], avg_loss_print[i], std_loss_print[i], std_loss_print[i]/np.sqrt(sol_per_pop)))
        file.close()

    ############################################################################
    # PLOT THE RESULTS
    ############################################################################

    if save_plot:
        if MODE=='first':
            plt.plot(best_loss, ls='-', lw=2, c='b')
            #plt.yscale('log')
            plt.xlabel(r'Generation')
            plt.ylabel(r'Best Loss')
            plt.savefig('fig01.pdf')
            plt.clf()

            plt.errorbar(np.arange(len(best_loss)), avg_loss, yerr=std_loss/np.sqrt(sol_per_pop), ls='-', lw=2, c='b')
            #plt.yscale('log')
            plt.xlabel(r'Generation')
            plt.ylabel(r'Average Loss')
            plt.savefig('fig02.pdf')
            plt.clf()

            plt.plot(std_loss, ls='-', lw=2, c='b')
            plt.xlabel(r'Generation')
            plt.ylabel(r'Standard Deviation of the loss')
            plt.savefig('fig03.pdf')
            plt.clf()

        elif MODE=='resume':
            plt.plot(np.array(best_loss_print), ls='-', lw=2, c='b')
            #plt.yscale('log')
            plt.xlabel(r'Generation')
            plt.ylabel(r'Best Loss')
            plt.savefig('fig01.pdf')
            plt.clf()

            plt.errorbar(np.arange(len(best_loss_print)), np.array(avg_loss_print), yerr=np.array(std_loss_print)/np.sqrt(sol_per_pop), ls='-', lw=2, c='b')
            #plt.yscale('log')
            plt.xlabel(r'Generation')
            plt.ylabel(r'Average Loss')
            plt.savefig('fig02.pdf')
            plt.clf()

            plt.plot(np.array(std_loss_print), ls='-', lw=2, c='b')
            plt.xlabel(r'Generation')
            plt.ylabel(r'Standard Deviation of the loss')
            plt.savefig('fig03.pdf')
            plt.clf()
    

    ############################################################################
    # EXPORT DATA FOR THE PLOT 2
    ############################################################################
    # Here we save the parameters of ALL the individuals in ALL generations,
    # along with their associated fitness.

    # all fitness values in the ALL epochs:
    all_gen_fitness = np.array(ga_GRB.solutions_fitness[:])

    # all solutions in the ALL epochs:
    all_gen_sol     = np.array(ga_GRB.solutions[:])
    all_gen_mu      = np.array(all_gen_sol[:,0])       # array with all the mu      of the ALL generations 
    all_gen_mu0     = np.array(all_gen_sol[:,1])       # array with all the mu0     of the ALL generations
    all_gen_alpha   = np.array(all_gen_sol[:,2])       # array with all the alpha   of the ALL generations
    all_gen_delta1  = np.array(all_gen_sol[:,3])       # array with all the delta1  of the ALL generations
    all_gen_delta2  = np.array(all_gen_sol[:,4])       # array with all the delta1  of the ALL generations
    all_gen_tau_min = 10**(np.array(all_gen_sol[:,5])) # array with all the tau_min of the ALL generations  # sample `tau_min` uniformly in log    space
    #all_gen_tau_min = np.array(all_gen_sol[:,5])      # array with all the tau_min of the ALL generations  # sample `tau_min` uniformly in linear space
    all_gen_tau_max = np.array(all_gen_sol[:,6])       # array with all the tau_max of the ALL generations

    data_all_gen = {
        'mu':      all_gen_mu,
        'mu0':     all_gen_mu0,
        'alpha':   all_gen_alpha,
        'delta1':  all_gen_delta1,
        'delta2':  all_gen_delta2,
        'tau_min': all_gen_tau_min,
        'tau_max': all_gen_tau_max,
        'fitness': all_gen_fitness
    }
    df_all_gen = pd.DataFrame(data_all_gen)
    df_all_gen.to_csv('./df_all_gen.csv', index=False)    
    
    ############################################################################
    ############################################################################

    print('\n')
    print('################################################################################')
    print('END')
    print('################################################################################')