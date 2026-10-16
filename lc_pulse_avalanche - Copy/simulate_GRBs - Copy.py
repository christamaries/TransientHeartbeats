################################################################################
# IMPORT LIBRARIES
################################################################################
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt

# Make stdout line-buffered so progress prints (including the periodic
# "[simulate] N/N_grb generated" lines in generate_GRBs()) appear live in a
# redirected/background log instead of only flushing at process exit.
sys.stdout.reconfigure(line_buffering=True)

import seaborn as sns
sns.set_style('darkgrid')

from matplotlib import rc


class LiveGenerationProgress:
    """
    Small on-screen window showing a filling progress bar as light curves
    are generated. Opt-in (config.ini: gui_progress = true), since it needs
    an actual display -- construction is wrapped in try/except by the
    caller so a missing display degrades to a printed warning, not a crash.
    """
    COLOR_FILL = '#2a78d6'
    COLOR_TRACK = '#e1e0d9'
    COLOR_INK = '#0b0b0b'
    COLOR_SURFACE = '#fcfcfb'

    def __init__(self, n_total):
        self.n_total = n_total
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 1.4), facecolor=self.COLOR_SURFACE)
        try:
            self.fig.canvas.manager.set_window_title('Generating light curves')
        except Exception:
            pass
        self.ax.set_facecolor(self.COLOR_SURFACE)
        self.ax.barh([0], [n_total], color=self.COLOR_TRACK, height=0.6)
        self.bar = self.ax.barh([0], [0], color=self.COLOR_FILL, height=0.6)[0]
        self.ax.set_xlim(0, n_total)
        self.ax.set_ylim(-0.6, 0.6)
        self.ax.set_yticks([])
        self.ax.set_xticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.text = self.ax.text(n_total / 2, 0, f'0/{n_total}', ha='center', va='center',
                                  color=self.COLOR_INK, fontweight='bold')
        self.fig.tight_layout()
        self.fig.show()
        plt.pause(0.01)

    def update(self, cnt, n_total):
        self.bar.set_width(cnt)
        self.text.set_text(f'{cnt}/{n_total}')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('text', usetex=True)

# Increase the recursion limit to avoid: 
# "RecursionError: maximum recursion depth exceeded in comparison"
rec_lim=50000
if sys.getrecursionlimit()<rec_lim:
    sys.setrecursionlimit(rec_lim)

# seed=42
# np.random.seed(SEED)

#------------------------------------------------------------------------------#
# Set the username for the path of the files:
#------------------------------------------------------------------------------#
user='external_user'
#user='LB'
#user='AF'
#user='bach'
#user='gravity'
#user='pleiadi'
if user=='bach':
    sys.path.append('/home/bazzanini/PYTHON/genetic/lc_pulse_avalanche/statistical_test')
    sys.path.append('/home/bazzanini/PYTHON/genetic/lc_pulse_avalanche/lc_pulse_avalanche')
elif user=='gravity':
    sys.path.append('/home/bazzanini/PYTHON/genetic3/statistical_test')
    sys.path.append('/home/bazzanini/PYTHON/genetic3/lc_pulse_avalanche')
elif user=='pleiadi':
    sys.path.append('/beegfs/mbulla/genetic_grbs/genetic/lc_pulse_avalanche/statistical_test')
    sys.path.append('/beegfs/mbulla/genetic_grbs/genetic/lc_pulse_avalanche/lc_pulse_avalanche')
elif user=='LB':
    sys.path.append('/Users/lorenzo/Documents/UNIVERSITA/Astrophysics/PYTHON/GRBs/lc_pulse_avalanche/statistical_test')
    sys.path.append('/Users/lorenzo/Documents/UNIVERSITA/Astrophysics/PYTHON/GRBs/lc_pulse_avalanche/lc_pulse_avalanche')
    export_path='../simulations/'
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
elif user=='AF':
    sys.path.append('C:/Users/lisaf/Desktop/GitHub/lc_pulse_avalanche/statistical_test')
    sys.path.append('C:/Users/lisaf/Desktop/GitHub/lc_pulse_avalanche/lc_pulse_avalanche')
    export_path='C:/Users/lisaf/Desktop/'
elif user=='external_user':
    sys.path.append('../statistical_test')
    sys.path.append('../lc_pulse_avalanche')
    export_path=''
else:
    raise ValueError('Assign to the variable "user" a correct username!')

from statistical_test import *
from avalanche import LC 
    
if __name__ == '__main__':

    #--------------------------------------------------------------------------#
    # SET PARAMETERS External user: read params from config file
    #--------------------------------------------------------------------------#
    if user=='external_user':
        if len(sys.argv) != 2:
            print("\nERROR\n")
            print("Usage: python simulate_GRBs.py <config_file>")
            sys.exit(1)

        remove_instrument_path = True
        # Read argument from command line, and create the dictionary
        config_file = sys.argv[1]
        variables   = read_values(filename=config_file)
        # Create the folder where to store the simulated GRB LCs
        sim_dir, lcs_dir, timestamp = create_dir(variables=variables)
        variables['dir'] = str(lcs_dir)
        # Assign the variables to the corresponding values
        instrument  = variables['instrument']
        N_grb       = variables['N_grb']
        mu          = variables['mu']
        mu0         = variables['mu0']
        alpha       = variables['alpha']
        delta1      = variables['delta1']
        delta2      = variables['delta2']
        tau_min     = variables['tau_min']
        tau_max     = variables['tau_max']
        export_path = variables['dir']
        gui_progress = str(variables.get('gui_progress', 'false')).strip().lower() == 'true'
        # Print the variables and their values
        print('==============================================')
        print('Generating LCs with the following properties:')
        print('==============================================')
        save_config_path = str(sim_dir)+'/config_'+timestamp+'.txt'
        save_config(variables=variables, file_name=save_config_path)
        print('==============================================')

    #--------------------------------------------------------------------------#
    # SET PARAMETERS Read params from down below
    #--------------------------------------------------------------------------#
    else:
        remove_instrument_path = False
        #----------------------------------------------------------------------#
        #instrument = 'batse'
        instrument = 'swift'
        #instrument = 'sax'
        #instrument = 'sax_lr'
        #instrument = 'fermi'
        #----------------------------------------------------------------------#
        N_grb = 5000 # number of simulated GRBs to produce per set of parameters
        #----------------------------------------------------------------------#
        # The 7 values obtained from BATSE v35 optimization are
        # (4 loss, Poisson, equal weights, keep_elitism=0, corrected noise+bkg,  
        # corrected ACF, corrected sampling of the individual peaks), 
        # corrected T90 estimate, fixed subcritical value, 30 gen
        # MEDIAN VALUES OF THE PARAMETERS IN THE LAST GENERATION
        if instrument=='batse':
            mu      = 1.10 
            mu0     = 0.91
            alpha   = 2.57
            delta1  = -1.28
            delta2  = 0.28
            tau_min = 0.02
            tau_max = 40.2

        #----------------------------------------------------------------------#
        # The 7 values obtained from Swift v36 optimization are
        # (4 loss, Poisson, equal weights, keep_elitism=0, corrected noise+bkg, 
        # corrected ACF, corrected sampling of the individual peaks), 
        # corrected T90 estimate, fixed subcritical value, 30 gen
        # MEDIAN VALUES OF THE PARAMETERS IN THE LAST GENERATION
        elif instrument=='swift':
            mu      = 1.34
            mu0     = 1.16
            alpha   = 2.53
            delta1  = -0.75
            delta2  = 0.27
            tau_min = 0.03
            tau_max = 56.8

        #----------------------------------------------------------------------#
        else:
            raise ValueError('Assign to the variable "instrument" a correct name!')

    # other parameters
    t_i   = 0    # [s]
    t_f   = 150  # [s]
    n_cut = 2500 # maximum number of pulses to consider in the avalanche model

    if instrument=='batse':
        res           = instr_batse['res']
        eff_area      = instr_batse['eff_area']
        bg_level      = instr_batse['bg_level']
        t90_threshold = instr_batse['t90_threshold']
        sn_threshold  = instr_batse['sn_threshold']
    elif instrument=='swift':
        res           = instr_swift['res']
        eff_area      = instr_swift['eff_area']
        bg_level      = instr_swift['bg_level']
        t90_threshold = instr_swift['t90_threshold']
        sn_threshold  = instr_swift['sn_threshold']
    # elif instrument=='sax':
    #     res           = instr_sax['res']
    #     eff_area      = instr_sax['eff_area']
    #     bg_level      = instr_sax['bg_level']
    #     t90_threshold = instr_sax['t90_threshold']
    #     sn_threshold  = instr_sax['sn_threshold']
    #     t_f           = 50 # s
    # elif instrument=='sax_lr':
    #     res           = instr_sax_lr['res']
    #     eff_area      = instr_sax_lr['eff_area']
    #     bg_level      = instr_sax_lr['bg_level']
    #     t90_threshold = instr_sax_lr['t90_threshold']
    #     sn_threshold  = instr_sax_lr['sn_threshold']
    # elif instrument=='fermi':
    #     res           = instr_fermi['res']
    #     eff_area      = instr_fermi['eff_area']
    #     bg_level      = instr_fermi['bg_level']
    #     t90_threshold = instr_fermi['t90_threshold']
    #     sn_threshold  = instr_fermi['sn_threshold']
    #     t_f           = 50 # s
    else:
        raise NameError('Variable "instrument" not defined properly; choose between: "batse" or "swift".')


    ################################################################################
    ################################################################################
    from datetime import datetime
    start = datetime.now()

    test_pulse_distr = False # True: currently broken upstream (count_significative_pulses_ver2
                              # references lc._ampl, which is dead/unset in avalanche.py).
                              # We use n_parent_pulses (mu_s) as the peak-count label instead.

    live_progress = None
    if user == 'external_user' and gui_progress:
        try:
            live_progress = LiveGenerationProgress(N_grb)
        except Exception as e:
            print(f"gui_progress=true but could not open a display ({e}); continuing without it.")

    test  = generate_GRBs(# number of simulated GRBs to produce
                          N_grb=N_grb,
                          # 7 parameters
                          mu=mu,
                          mu0=mu0,
                          alpha=alpha,
                          delta1=delta1,
                          delta2=delta2,
                          tau_min=tau_min,
                          tau_max=tau_max,
                          # instrument parameters
                          instrument=instrument,
                          bin_time=res,
                          eff_area=eff_area,
                          bg_level=bg_level,
                          # constraint parameters
                          t90_threshold=t90_threshold,
                          sn_threshold=sn_threshold,
                          t_f=t_f,
                          filter=True,
                          # other parameters
                          export_files=True,
                          export_path=export_path,
                          n_cut=n_cut,
                          with_bg=False,
                          remove_instrument_path=remove_instrument_path,
                          test_pulse_distr=test_pulse_distr,
                          progress_callback=(live_progress.update if live_progress is not None else None))

    if live_progress is not None:
        print("Close the generation-progress window to continue...")
        plt.ioff()
        plt.show()

    if test_pulse_distr:
        pulse_out_file=open('./n_of_pulses.txt', 'w')
        for grb in test:
            pulse_out_file.write('{0}\n'.format(grb.num_of_sig_pulses))
        pulse_out_file.close()

    if test_pulse_distr:
        n_of_pulses = [ grb.num_of_sig_pulses for grb in test ]

    print('Time elapsed: ', (datetime.now() - start).seconds)

################################################################################
################################################################################

# The 7 values obtained from v1 optimization are
# (3 loss)
# mu      = 1.3712230777324108
# mu0     = 1.292056879500315
# alpha   = 6.238631180118012
# delta1  = -0.5895371604462968
# delta2  = 0.21749228991192124
# tau_min = 0.06234108759332604
# tau_max = 23.443421866972386

# The 7 values obtained from v2 optimization are
# (4 loss)
# mu      = 1.3824946258409123
# mu0     = 1.15547634120758
# alpha   = 5.240511090395332
# delta1  = -0.45579705811174676
# delta2  = 0.1341616114704469
# tau_min = 0.003487215483012309
# tau_max = 32.858056193896196

# The 7 values obtained from v3 optimization are
# (4 loss)
# mu      = 1.3980875041410008
# mu0     = 1.5997385641936739
# alpha   = 3.8373579048667117
# delta1  = -0.5497159353657516
# delta2  = 0.12206808487464499
# tau_min = 0.00047431784713861797
# tau_max = 39.313297221735766

# The 7 values obtained from v4 optimization are
# (4 loss)
# mu      = 1.7377495777582268
# mu0     = 1.2674137674116688
# alpha   = 6.56892665444723
# delta1  = -0.5989803252226719
# delta2  = 0.02306881143876948
# tau_min = 6.478038929262871e-06
# tau_max = 45.936383095147605

# The 7 values obtained from v5 optimization are
# (4 loss)
# mu      = 1.8642165398675894
# mu0     = 0.9460684332226531
# alpha   = 6.539055496753974
# delta1  = -0.7805636907606287
# delta2  = 0.07414591188731365
# tau_min = 6.350848178629759e-06
# tau_max = 52.41492789344243

# The 7 values obtained from v6 optimization are
# (4 loss)
# mu      = 1.5355877552761932
# mu0     = 1.534168123065679
# alpha   = 3.1200524011794863
# delta1  = -0.7655182486991188
# delta2  = 0.2206237762670341
# tau_min = 0.0018477209878527603
# tau_max = 50.124910976218175

# The 7 values obtained from v7 optimization are
# (5 loss)
# mu      = 1.5197492009322398
# mu0     = 1.5588763589949317
# alpha   = 2.7027204695213194
# delta1  = -0.7741267250062283
# delta2  = 0.20809088491524874
# tau_min = 0.025098559904990592
# tau_max = 53.18239761751395

# The 7 values obtained from v8 optimization are
# (5 loss, Poisson)
# mu      = 1.264930172400689
# mu0     = 1.9545413768529967
# alpha   = 1.9791552702076287
# delta1  = -0.5633113305426156
# delta2  = 0.21883657826929145
# tau_min = 0.023468723791192015
# tau_max = 37.80588105257772

# The 7 values obtained from v9 optimization are
# (4 loss, Poisson)
# mu      = 1.3329447024643284
# mu0     = 1.2263029893911657
# alpha   = 3.273629312229965
# delta1  = -0.49675439969345714
# delta2  = 0.12694034684654393
# tau_min = 0.00011096965026213441
# tau_max = 41.172294302312366

# The 7 values obtained from v10 optimization are
# (one loss (<F/F_p>), Poisson)
# mu      = 0.9475251603474704
# mu0     = 0.9798380943908988
# alpha   = 3.5671437449135976
# delta1  = -0.7198120480484348
# delta2  = 0.14634415982397875
# tau_min = 5.535018240950338e-06
# tau_max = 27.408501454687663

# The 7 values obtained from v11 optimization are
# (one loss (<(F/F_p)**3>), Poisson)
# mu      = 1.275006111708082
# mu0     = 0.9241332593921731
# alpha   = 3.9560667766550814
# delta1  = -0.6039275951949501
# delta2  = 0.04714747706984365
# tau_min = 3.1160832088232976e-06
# tau_max = 27.685109594537607

# The 7 values obtained from v12 optimization are
# (one loss (<ACF>), Poisson)
# mu      = 0.9446841808077004
# mu0     = 0.9681754054935662
# alpha   = 1.757960546967845
# delta1  = -0.32164512051516003
# delta2  = 0.09051446260540383
# tau_min = 0.0009609765738533678
# tau_max = 31.1900145774892

# The 7 values obtained from v13 optimization are
# (one loss (duration distribution), Poisson)
# mu      = 1.423219221170363
# mu0     = 1.8486292768207997
# alpha   = 2.607690092406438
# delta1  = -0.6637684243758021
# delta2  = 0.20456507843804805
# tau_min = 0.007777873403883386
# tau_max = 47.147406633040376

# The 7 values obtained from v14 optimization are
# (4 loss, Poisson, correct weight, keep_elitism=0)
# mu      = 1.3400110297200563
# mu0     = 1.890238938249485
# alpha   = 3.1950527492978287
# delta1  = -0.4600048148857563
# delta2  = 0.12234358572102152
# tau_min = 5.563858305836977e-05
# tau_max = 40.49163583715167

# The 7 values obtained from v15 optimization are
# (4 loss, Poisson, correct weight, keep_elitism=0, long run)
# mu      = 1.4081142187106597
# mu0     = 1.6184509962933409
# alpha   = 2.77910446583413
# delta1  = -0.621068787091892
# delta2  = 0.17772316530635301
# tau_min = 0.0027027975256713383
# tau_max = 53.92090393324469

# The 7 values obtained from v16 optimization are
# (4 loss, Poisson, equal weights, keep_elitism=0, corrected noise+bkg)
#mu      = 1.4678732974780715
#mu0     = 1.0225493407693491
#alpha   = 3.4225313614068393
#delta1  = -1.39120506763176
#delta2  = 0.24613420719741125
#tau_min = 1.635473454362247e-06
#tau_max = 20.154342298134676

# The 7 values obtained from v17 optimization are
# (1 loss (ACF), only 5 epochs, Poisson, keep_elitism=0, corrected noise+bkg, 2000sol/pop)
# mu      = 0.8813384397086275
# mu0     = 1.1694147630314133
# alpha   = 6.978449701941005
# delta1  = -1.42008244756357
# delta2  = 0.05597302855361258
# tau_min = 0.006233072521668988
# tau_max = 31.825617004821087

# The 7 values obtained from v18 optimization are
# (1 loss (<F/F_p>), only 5 epochs, Poisson, keep_elitism=0, corrected noise+bkg, 2000sol/pop)
# mu      = 1.0615698644125227
# mu0     = 1.881290733838656s
# alpha   = 4.438031556734086
# delta1  = -0.735786216779466
# delta2  = 0.14969222298849683
# tau_min = 2.0214344096142374e-05
# tau_max = 12.924266336471554

# The 7 values obtained from v19 optimization are
# (1 loss (<(F/F_p)**3>), only 5 epochs, Poisson, keep_elitism=0, corrected noise+bkg, 2000sol/pop)
# mu      = 1.395734421965432
# mu0     = 1.5675268339692208
# alpha   = 2.0611645049040073
# delta1  = -0.7651818839706319
# delta2  = 0.14212243115759166
# tau_min = 6.570153404289466e-05
# tau_max = 18.695600016465686

# The 7 values obtained from v20 optimization are
# (1 loss (duration-distr), only 5 epochs, Poisson, keep_elitism=0, corrected noise+bkg, 2000sol/pop)
# mu      = 0.9747038698791641
# mu0     = 1.4428453243271493
# alpha   = 6.404184757187495
# delta1  = -1.314982098429236
# delta2  = 0.23869731701958827
# tau_min = 0.0016902949925141984
# tau_max = 10.544462322977166

# The 7 values obtained from v21 optimization are
# (3 loss (all but <(F/F_p)**3>), only 5 epochs, Poisson, keep_elitism=0, corrected noise+bkg, 2000sol/pop)
# mu      = 1.0247085207638207
# mu0     = 1.3703937771297046
# alpha   = 2.992874185558952
# delta1  = -0.5913798763882866
# delta2  = 0.2050623764263088
# tau_min = 5.870199320373939e-05
# tau_max = 11.676444516246468

# The 7 values obtained from Swift v22 optimization are
# (4 loss, Poisson, equal weights, keep_elitism=0, corrected noise+bkg)
# mu      = 1.0650454588484382
# mu0     = 1.5516431424585888
# alpha   = 3.8889811365200426
# delta1  = -1.39182619673643
# delta2  = 0.10639085827033429
# tau_min = 0.015446264618348984
# tau_max = 41.55450235716483

# The 7 values obtained from Swift v24 optimization are
# (4 loss, Poisson, equal weights, keep_elitism=0, corrected noise+bkg, corrected ACF)
# mu      = 1.2821767278364844
# mu0     = 1.5371224933120717
# alpha   = 3.3222629941943387
# delta1  = -0.9767114418094832
# delta2  = 0.1608353510843784
# tau_min = 5.225403089711409e-05
# tau_max = 45.998967578827624

# The 7 values obtained from BATSE v25 optimization are
# (4 loss, Poisson, equal weights, keep_elitism=0, corrected noise+bkg, corrected ACF)
# mu      = 1.2879173930592351
# mu0     = 0.8919030666746223
# alpha   = 3.752197999884041
# delta1  = -0.96938196988164
# delta2  = 0.2451901980912974
# tau_min = 6.5224264304019425e-06
# tau_max = 14.816904711205176

# The 7 values obtained from BATSE v30 optimization are
# (4 loss, Poisson, equal weights, keep_elitism=0, corrected noise+bkg, corrected ACF, 
# corrected sampling of the individual peaks)
# MEDIAN VALUES OF THE PARAMETERS IN THE LAST GENERATION
# mu      = 1.08
# mu0     = 1.18
# alpha   = 9.78
# delta1  = -0.81
# delta2  = 0.10
# tau_min = 0.04
# tau_max = 32.20

# The 7 values obtained from BATSE v31 optimization are
# (4 loss, Poisson, equal weights, keep_elitism=0, corrected noise+bkg, corrected ACF, 
# corrected sampling of the individual peaks), corrected T90 estimate 
# MEDIAN VALUES OF THE PARAMETERS IN THE LAST GENERATION
# mu      = 1.02
# mu0     = 0.96
# alpha   = 2.84
# delta1  = -1.32
# delta2  = 0.28
# tau_min = 0.02
# tau_max = 34.8

# The 7 values obtained from Swift v32 optimization are
# (4 loss, Poisson, equal weights, keep_elitism=0, corrected noise+bkg, corrected ACF, 
# corrected sampling of the individual peaks), corrected T90 estimate 
# MEDIAN VALUES OF THE PARAMETERS IN THE LAST GENERATION
# mu      = 1.26
# mu0     = 1.17
# alpha   = 2.96
# delta1  = -0.69
# delta2  = 0.26
# tau_min = 0.02
# tau_max = 47.8

# The 7 values obtained from BATSE v33 optimization are
# (4 loss, Poisson, equal weights, keep_elitism=0, corrected noise+bkg, corrected ACF, 
# corrected sampling of the individual peaks), corrected T90 estimate, fixed subcritical value
# MEDIAN VALUES OF THE PARAMETERS IN THE LAST GENERATION
# mu      = 1.09
# mu0     = 0.96
# alpha   = 2.10
# delta1  = -1.27
# delta2  = 0.24
# tau_min = 0.02
# tau_max = 41.2
    
# The 7 values obtained from Swift v34 optimization are
# (4 loss, Poisson, equal weights, keep_elitism=0, corrected noise+bkg, corrected ACF, 
# corrected sampling of the individual peaks), corrected T90 estimate, fixed subcritical value
# MEDIAN VALUES OF THE PARAMETERS IN THE LAST GENERATION
# mu      = 1.26
# mu0     = 1.29
# alpha   = 3.18
# delta1  = -0.93
# delta2  = 0.25
# tau_min = 0.02
# tau_max = 48.2

################################################################################
################################################################################