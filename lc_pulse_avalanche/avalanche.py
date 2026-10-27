import inspect
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import exponential, lognormal, normal, uniform
from scipy.stats import poisson
import os, h5py

user={'bach','gravity'}

SEED=None
#SEED=42
#np.random.seed(SEED)

#==============================================================================#
#==============================================================================#

### See `statistical_tests.ipynb`
def generate_rand_from_pdf(pdf, x_grid, N=1):
    """
    Generates `N` random numbers from a given probability distribution function
    `pdf`, by taking values on the x-axis on a grid `x_grid`.
    """
    cdf             = np.cumsum(pdf)
    cdf             = cdf / cdf[-1]
    values          = np.random.rand(N)
    value_bins      = np.searchsorted(cdf, values)
    random_from_cdf = x_grid[value_bins]
    return random_from_cdf

### Load the pdf of peak count rates of each instrument, with which we
# will sample the amplitude A of each pulse (we'll not sample A anymore 
# from U[0,1])
peak_count_rates_batse  = '../lc_pulse_avalanche/kde_pdf_BATSE_peak_count_rates.txt'                                                # LB, AF, external_user
#peak_count_rates_batse = '/home/bazzanini/PYTHON/genetic/lc_pulse_avalanche/lc_pulse_avalanche/kde_pdf_BATSE_peak_count_rates.txt' # bach
#peak_count_rates_batse = '/home/bazzanini/PYTHON/genetic3/lc_pulse_avalanche/kde_pdf_BATSE_peak_count_rates.txt'                   # gravity
peak_count_rates_swift  = '../lc_pulse_avalanche/kde_pdf_Swift_peak_count_rates.txt'                                                # LB, AF, external_user
#peak_count_rates_swift = '/home/bazzanini/PYTHON/genetic/lc_pulse_avalanche/lc_pulse_avalanche/kde_pdf_Swift_peak_count_rates.txt' # bach
#peak_count_rates_swift = '/home/bazzanini/PYTHON/genetic3/lc_pulse_avalanche/kde_pdf_Swift_peak_count_rates.txt'                   # gravity
#
pdf_peak_count_rates_batse = np.loadtxt(peak_count_rates_batse)
pdf_peak_count_rates_swift = np.loadtxt(peak_count_rates_swift)
#
low_exp_batse  =  2
low_exp_swift  = -3
high_exp_batse =  6
high_exp_swift =  3
x_grid_batse = np.linspace(10**low_exp_batse, 10**high_exp_batse, 2000000)
x_grid_swift = np.linspace(10**low_exp_swift, 10**high_exp_swift, 2000000)
peak_count_rate_batse_sample = generate_rand_from_pdf(pdf_peak_count_rates_batse, x_grid_batse, N=100000) 
peak_count_rate_swift_sample = generate_rand_from_pdf(pdf_peak_count_rates_swift, x_grid_swift, N=100000) 


### Load the (gaussian) errors of the Swift GRBs
path_swift_errs = '../lc_pulse_avalanche/' # LB
#path_swift_errs = '/home/bazzanini/PYTHON/genetic/lc_pulse_avalanche/lc_pulse_avalanche/' # bach
#path_swift_errs = '/home/bazzanini/PYTHON/genetic3/lc_pulse_avalanche/' # gravity
#
# bins_swift_errs = np.array([  0.1, 0.21544347, 0.46415888, 1., 2.15443469, 4.64158883, 10. , 21.5443469 , 46.41588834, 100. ])
bins_swift_errs = np.array([  0.1, 0.21544347, 0.46415888, 1., 2.15443469, 4.64158883, 10.])
dict_errs_swift = {}
for i in range(1, len(bins_swift_errs)+1):
    with open(path_swift_errs+'swift_errs_'+str(i)+'.txt', 'r') as f:
        dict_errs_swift[str(i)] = f.readlines()
for key in dict_errs_swift.keys():
    for i, line in enumerate(dict_errs_swift[key]):
        line       = line.rstrip(' \n')
        errs_split = list(map(float, line.split(' ')))
        dict_errs_swift[key][i] = errs_split

#==============================================================================#
# Define the class LC describing the light curve.                              #
#==============================================================================#

class LC(object):
    """
    A class to generate gamma-ray burst light curves (GRB LCs) using a pulse
    avalanche model ('chain reaction') proposed by Stern & Svensson, ApJ, 
    469: L109 (1996).
    
    :mu: average value of the Poisson distribution that samples the 
         number of child pulses, which is mu_b; average number of baby pulses
    :mu0: average value of the Poisson distribution that samples the 
          number of primary pulses, which is mu_s; average number of spontaneous
          (initial) pulses per GRB
    :alpha: delay parameter
    :delta1: lower boundary of log-normal probability distribution of tau
             (time constant of baby pulse)
    :delta2: upper boundary of log-normal probability distribution of tau
    :tau_min: lower boundary of log-normal probability distribution of tau_0 
              (time constant of spontaneous pulse); should be smaller than res
    :tau_max: upper boundary of log-normal probability distribution of tau_0
    :t_min: GRB LC start time
    :t_max: GRB LC stop time
    :res: GRB LC time resolution (s) (i.e., bin time)
    :eff_area: effective area of instrument (cm2)
    :bg_level: background level rate per unit area of detector (cnt/cm2/s)
    :min_photon_rate: left  boundary of -3/2 log N - log S distribution (ph/cm2/s)
    :max_photon_rate: right boundary of -3/2 log N - log S distribution (ph/cm2/s)
    :sigma: signal above background level
    :n_cut: maximum number of pulses in avalanche (useful to speed up the 
            simulations but in odds with the "classic" approach)
    :with_bg: boolean flag for keeping or removing the background level at the 
              end of the generation
    :use_poisson: boolean flag for using the Poisson or the (rounded) 
                  exponential for sampling the number of initial pulses and child
    """
    
    def __init__(self, mu=1.2, mu0=1, alpha=4, delta1=-0.5, delta2=0, 
                 tau_min=0.2, tau_max=26, t_min=-10, t_max=1000, res=0.256, 
                 eff_area=3600, bg_level=10.67, with_bg=True, use_poisson=True,
                 min_photon_rate=1.3, max_photon_rate=1300, sigma=5, 
                 n_cut=None, instrument='batse', verbose=False):
        
        self._mu      = mu # mu~1 --> critical runaway regime
        self._mu0     = mu0 
        self._alpha   = alpha 
        self._delta1  = delta1
        self._delta2  = delta2
        self._tau_min = tau_min
        self._tau_max = tau_max
        if tau_min > res and not isinstance(self, Restored_LC):
            raise ValueError("tau_min should be smaller than res =", res)
        self._eff_area = eff_area 
        self._bg = bg_level * self._eff_area # cnt/s
        self._min_photon_rate = min_photon_rate  
        self._max_photon_rate = max_photon_rate 
        self._verbose = verbose
        self._res = res # s
        self._n = int(np.ceil((t_max - t_min)/self._res)) + 1 # time steps
        self._t_min = t_min # ms
        self._t_max = (self._n - 1) * self._res + self._t_min # ms
        self._times, self._step = np.linspace(self._t_min, self._t_max, self._n, retstep=True)
        # Arrays of COUNT RATES
        self._rates       = np.zeros(len(self._times))
        self._sp_pulse    = np.zeros(len(self._times))
        self._total_rates = np.zeros(len(self._times))
        # Arrays of COUNTS
        self._child_counts  = np.zeros(len(self._times))
        self._parent_counts = np.zeros(len(self._times))
        # Other parameters
        self._lc_params   = list()
        self._sigma       = sigma
        self._n_cut       = n_cut
        self._n_pulses    = 0
        self._mu_s        = 0 # number of spontaneous (parent) pulses, set in generate_avalanche()
        self._with_bg     = with_bg
        self._use_poisson = use_poisson
        self._instrument  = instrument

        if self._instrument == 'batse':
            self._peak_count_rate_sample = peak_count_rate_batse_sample
        elif self._instrument == 'swift':
            self._peak_count_rate_sample = peak_count_rate_swift_sample
        elif self._instrument == 'sax_lr':
            pass
            #self._peak_count_rate_sample = peak_count_rate_sax_lr_sample
        elif self._instrument == 'sax':
            pass
            #self._peak_count_rate_sample = peak_count_rate_sax_sample
        elif self._instrument == 'fermi':
            pass
            #self._peak_count_rate_sample = peak_count_rate_fermi_sample
        else:
            raise ValueError("Instrument not recognized...")
        
        # if self._verbose:
        #     print("Time resolution: ", self._step)

    #--------------------------------------------------------------------------#
     
    def norris_pulse(self, norm, tp, tau, tau_r):
        """
        Computes a single pulse according to: 
            Norris et al., ApJ, 459, 393 (1996).
        
        :norm: amplitude of the pulse, scalar
        :tp: pulse peak time, scalar
        :tau: pulse width (decay time), scalar
        :tau_r: rise time, scalar

        :returns: an array of COUNT RATES
        """

        self._n_pulses += 1
        
        if self._verbose:
            print("Generating a new pulse with tau = {:0.3f}".format(tau))

        t   = (self._times).astype(np.float64) # times (lc x-axis)
        _tp = np.ones(len(t))*tp
        
        if tau_r == 0 or tau == 0: 
            raise ValueError("`tau_r` or `tau` cannot be zero!")
            #return np.zeros(len(t))
        
        return np.append(norm * np.exp(-(t[t<=tp]-_tp[t<=tp])**2/tau_r**2), \
                         norm * np.exp(-(t[t>tp] -_tp[t>tp])/tau)).astype(np.float64)

    #--------------------------------------------------------------------------#
   
    def _rec_gen_pulse(self, tau1, t_shift, parent_id):
        """
        Recursively generates pulses from Norris function

        :tau1: parent pulse width (decay rime), scalar
        :parent_id: index into self._lc_params of this pulse's immediate
                    progenitor (used only to link children to their parent
                    for plotting; not part of the physical model)
        :t_shift: time delay relative to the parent pulse

        AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        t_shift: time delay relative to the parent pulse
        INITIAL parent pulse, or the IMMEDIATE parent pulse? It should be the INITIAL

        LB: `t_shift` should be the time delay of the immediate parent pulse with 
        respect to the initial invisible trigger event; the time delay of the 
        child pulse w.r.t the parent event instead is computed here below.

        Mi sembra che la definizione di `t_shift` data due righe sopra non sia 
        corretta, cioe' mi pare che ogni volta che chiami la funzione ricorsiva
        devi passare il delay TOTALE, ovvero la somma di tutti i delay dei
        pulses genitori!
        AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA 

        :returns: an array of count rates
        """
            
        # the number of child pulses is given by: 
        #     p2(mu_b) = exp(-mu_b/mu)/mu
        # mu: average, 
        # mu_b: actual number of child pulses.
        if self._use_poisson: # Our code
            mu_b = poisson.rvs(mu=self._mu, 
                               size=1, 
                               random_state=None)
            mu_b = mu_b[0]
        else: # Anastasia
            mu_b = round(exponential(scale=self._mu))
                
        if self._verbose:
            print("Number of child pulses:", mu_b)
            print("--------------------------------------------------------------------------")
        
        # Loop over the child pulses
        for i in range(mu_b):
            
            # The time const of the child pulse (tau) is given by:
            #     p4(log10(tau/tau1)) = 1/(delta2 - delta1)
            # tau1: time const of the parent pulse
            tau = tau1 * 10**(uniform(low=self._delta1, high=self._delta2))
            
            # The avalanche stops when the time constant tau of the pulse goes 
            # below the 1/10 of the time resolution (self._res)
            frac_res = 0.1
            if (tau < (frac_res*self._res)):
                continue
            
            # Rise time
            tau_r = 0.5 * tau
            
            # The time delay (delta_t) of child pulse (with respect to the parent
            # pulse) is given by:
            #     p3(delta_t) = exp(-delta_t/(alpha*tau))/(alpha*tau) 
            delta_t = t_shift + exponential(scale=self._alpha*tau)
            
            # The amplitude (A) of each pulse is given by:
            #     p1(A) = 1, in [0, 1]
            # norm = uniform(low=0.0, high=1.0)
            # Each pulse (count-rate) composing the LC has an amplitude sampled in U[0,A_max]
            norm_A = uniform(low=0.0, high=self._A_max)
            
            # self._rates    += self.norris_pulse(norm, delta_t, tau, tau_r)  # WRONG
            # self._n_pulses -= 1 # since we're calling `norris_pulse` twice the times, we're counting the same pulse twice
            # LB: this is not correct! Indeed, when tau is smaller than the bin_time,
            # then we cannot obtain the counts just by multiplying the count rate
            # times the bin time. In this case, we should integrate the count rate
            # over the length of the pulse (basically, we cannot integrate with
            # rectangles anymore if the integration range is smaller than Delta_t).
            # Instead if the pulse lasts for a time longer than the bin_time, then
            # the total counts can be approximated with count rate times bin_time.
            # Therefore, below we store the arrays of counts (self._parent_counts
            # and self._child_counts), not count rates anymore, and we treat the
            # two cases separately (instead of integrating we multiply times tau).
#
            count_rates_pulse = self.norris_pulse(norm_A, delta_t, tau, tau_r)
            #print('\n\n\n')
            #print('SUM COUNTS RATES:')
            #print(np.sum(count_rates_pulse))
            if tau>self._res:
                counts_pulse        = count_rates_pulse * self._res
                self._child_counts += counts_pulse
            else:
                counts_pulse        = count_rates_pulse * tau
                self._child_counts += counts_pulse
            # export the array `counts_pulse` to a txt file
            #np.savetxt('../simulations/counts_pulse_'+str(norm_A)+'.txt', counts_pulse)
            #print('\n\n\n')
            #print('SUM COUNTS:')
            #print(np.sum(counts_pulse))
            # Save a dictionary with the 4 parameters of the pulse, and the 
            # total counts of the single pulse
            own_id = len(self._lc_params)
            self._lc_params.append(dict(norm=norm_A,
                                        t_delay=delta_t,
                                        tau=tau,
                                        tau_r=tau_r,
                                        counts_pulse=np.sum(counts_pulse),
                                        is_parent=False,
                                        parent_id=parent_id))

            if self._verbose:
                print("Pulse amplitude: {:0.3f}".format(norm_A))
                print("Pulse shift: {:0.3f}".format(delta_t))
                print("Time constant (the decay time): {0:0.3f}".format(tau))
                print("Rise time: {:0.3f}".format(tau_r))
                print("--------------------------------------------------------------------------")

            # The avalanche stops when the time constant tau of the pulse goes 
            # below the 1/10 of the time resolution (self._res), which has
            # been already checked above, OR when the number of total pulses 
            # becomes greater than a given number of our choice (self._n_cut).
            if (tau >= (frac_res*self._res)): # THIS IS ALWAYS TRUE SINCE WE CHECKED IT ABOVE!
                if self._n_cut is None:
                    self._rec_gen_pulse(tau, delta_t, own_id)
                else:
                    if self._n_pulses < self._n_cut:
                        self._rec_gen_pulse(tau, delta_t, own_id)
            # else, stop this brach of the family tree chain
            else:
                continue 

        return self._rates
    
    #--------------------------------------------------------------------------#
    
    def generate_avalanche(self, seed=SEED, return_array=False):
        """
        Generates the pulse avalanche, i.e., the GRB light curve.
        
        :seed: random seed
        :return_array: if True returns arrays of parameters, if False, a dict
                       with parameters for each pulse
        :returns: set of parameters for the generated avalanche
        """
        
        # set seed for random draw (the same as for the avalanche generation)
        np.random.seed(seed)
        
        if self._verbose:
            inspect.getdoc(self.generate_avalanche)
   
        # The number of spontaneous primary pulses (mu_s) is given by: 
        #     p5(mu_s) = exp(-mu_s/mu0)/mu0
        if self._use_poisson: # Our code
            mu_s = 0
            while (mu_s==0):
                mu_s = poisson.rvs(mu=self._mu0, 
                                   size=1, 
                                   random_state=None)
                mu_s = mu_s[0]
        else: # Anastasia code
            mu_s = round(exponential(scale=self._mu0))
            if (mu_s==0):  
                mu_s = 1 
            
        self._mu_s = mu_s

        if self._verbose:
            print("Number of spontaneous (primary) pulses:", mu_s)
            print("--------------------------------------------------------------------------")

        # The amplitude (A) of each pulse is  sampled from a uniform distribution, 
        # from A_min=0 to the value A_max sampled from the pdf of peak count 
        # RATES of each instrument. A_max is the same for a given GRB (and it is
        # drawn here below), and each pulse composing the LC has an amplitude 
        # sampled in U[0,A_max].
        self._A_max = np.random.choice(self._peak_count_rate_sample, size=1)[0]
        
        # For each of the mu_s _parent_ pulse, generate his child pulses
        for i in range(mu_s):
            # The time constant of spontaneous pulses (decay time tau0) is given by: 
            #     p6(log10 tau0) = 1/(log10 tau_max - log10 tau_min)
            tau0 = 10**(uniform(low=np.log10(self._tau_max), high=np.log10(self._tau_min)))
            tau0 = np.float64(tau0)

            # Rise time
            tau_r = 0.5 * tau0
            tau_r = np.float64(tau_r)

            # The time delay (t_delay) of each spontaneous primary pulses with
            # respect to a common invisible trigger event is given by:
            #     p7(t) = exp(-t/(alpha*tau0))/(alpha*tau0)
            t_delay = exponential(scale=self._alpha*tau0)
            t_delay = np.float64(t_delay)

            # The amplitude (A) of each pulse is given by:
            #     p1(A) = 1, in [0, 1]
            # norm = uniform(low=0.0, high=1) 
            # Each pulse composing the LB has an amplitude sampled in U[0,A_max].
            norm_A = uniform(low=0.0, high=self._A_max)
            norm_A = np.float64(norm_A)
            
            if self._verbose:
                print("Spontaneous pulse amplitude: {:0.3f}".format(norm_A))
                print("Spontaneous pulse shift: {:0.3f}".format(t_delay))
                print("Time constant (the decay time) of spontaneous pulse: {0:0.3f}".format(tau0))
                print("Rise time of spontaneous pulse: {:0.3f}".format(tau_r))
                print("--------------------------------------------------------------------------")
            
            # Generate the pulse (count rates), and sum it into the array 'self._sp_pulse'
            # self._sp_pulse += self.norris_pulse(norm, t_delay, tau0, tau_r)  # WRONG
            # self._n_pulses -= 1 # since we're calling `norris_pulse` twice the times, we're counting the same pulse twice
            # LB: this is not correct! Indeed, when tau is smaller than the bin_time,
            # then we cannot obtain the counts just by multiplying the count rate
            # times the bin time. In this case, we should integrate the count rate
            # over the length of the pulse (basically, we cannot integrate with
            # rectangles anymore if the integration range is smaller than Delta_t).
            # Instead if the pulse lasts for a time longer than the bin_time, then
            # the total counts can be approximated with count rate times bin_time.
            # Therefore, below we store the arrays of counts (self._parent_counts
            # and self._child_counts), not count rates anymore, and we treat the
            # two cases separately (instead of integrating we multiply times tau).
            count_rates_pulse = self.norris_pulse(norm_A, t_delay, tau0, tau_r)
            if tau0>self._res:
                counts_pulse         = count_rates_pulse * self._res
                self._parent_counts += counts_pulse
            else:
                counts_pulse         = count_rates_pulse * tau0
                self._parent_counts += counts_pulse
            
            # Save a dictionary with the 4 parameters of the pulse, and the 
            # total counts of the single pulse: 
            #     A (norm_A), t_p (t_delay), tau_0 (tau), tau_r (tau_r), 
            # in Stern & Svensson, ApJ, 469: L109 (1996), pag 2.
            parent_own_id = len(self._lc_params)
            self._lc_params.append(dict(norm=norm_A,
                                        t_delay=t_delay,
                                        tau=tau0,
                                        tau_r=tau_r,
                                        counts_pulse=np.sum(counts_pulse),
                                        is_parent=True,
                                        parent_id=None))

            # generate the avalanche of child pulses.
            # it takes as input the tau of the parent pulse (tau0) and the time
            # delay of the parent pulse (t_delay).
            # N.B. the time constant of the parent pulse il called 'tau1' in
            # the paper by Stern & Svensson, on page 2.
            self._rec_gen_pulse(tau0, t_delay, parent_own_id)

        mini_verbose=0
        if mini_verbose:
            print("--------------------------------------------------------------------------")
            print("Number of spontaneous (primary) pulses:", mu_s)
            print("Total number of child pulses          :", self._n_pulses-mu_s)
            print("---")
            print("Total number of pulses                :", self._n_pulses)

        # To obtain the lc from the avalanche, we sum the lc of the parents 
        # (self._parent_counts) with the lc of the child (self._child_counts). 
        # `self._raw_lc`        has units: cnt/cm2/s
        # `self._raw_lc_counts` has units: cnt/cm2
        # self._raw_lc        = self._sp_pulse      + self._rates        # count RATES (do not use this!)
        self._raw_lc_counts = self._parent_counts + self._child_counts # COUNTS

        # self._max_raw_pcr = self._raw_lc.max()
        # if (self._max_raw_pcr<1.e-12):
        #     # check that we have generated a lc with non-zero values; otherwise,
        #     # exit and set the flag 'self.check=0', which indicates that this
        #     # lc has to be skipped
        #     self.check=0
        #     return 0
        # else:
        #     self.check=1
        self._max_raw_pc = self._raw_lc_counts.max()
        self._peak_value = self._max_raw_pc
        if (self._max_raw_pc<1.e-12):
            # check that we have generated a lc with non-zero values; otherwise,
            # exit and set the flag 'self.check=0', which indicates that this
            # lc has to be skipped
            self.check=0
            return 0
        else:
            self.check=1

        # population = np.geomspace(self._min_photon_rate , self._max_photon_rate, 1000)
        # weights    = list(map(lambda x: x**(-3/2), population))
        # weights    = weights / np.sum(weights)
        # ampl       = np.random.choice(population, p=weights) / self._max_raw_pcr
        # self._ampl = ampl
        # self._peak_value = self._max_raw_pcr * self._ampl

        # lc from avalanche scaled + Poissonian bg added (for BATSE)
        # For BATSE, the variable `_plot_lc` contains the COUNTS (and not the count RATES!)
        if self._instrument == 'batse':
            self._model           = self._raw_lc_counts                                 # model COUNTS 
            self._modelbkg        = self._model + (self._bg * self._res)                # model COUNTS + constant bgk counts
            self._plot_lc         = np.random.poisson(self._modelbkg).astype('float')   # total COUNTS (signal+noise) with Poisson
            self._plot_lc_with_bg = self._plot_lc  
            self._err_lc          = np.sqrt(self._plot_lc)
            if self._with_bg: # lc with background
                pass
            else: # background-subtracted lc
                self._plot_lc = self._plot_lc - (self._bg * self._res)   # total COUNTS (removed the constant bkg level)
        
        # For Swift, the variable `_plot_lc` contains the COUNTS RATES (and not the counts!)
        elif self._instrument == 'swift':
            self._model           = self._raw_lc_counts      # model COUNTS 
            self._model_rate      = self._model / self._res  # model COUNT RATES
            self._modelbkg        = self._model              # bkg 0 in Swift
            self._modelbkg_rate   = self._model_rate         # bkg 0 in Swift
            #
            if np.max(self._model_rate)<bins_swift_errs[1]:
                errs_swift_list = dict_errs_swift['1']
            elif np.max(self._model_rate)<bins_swift_errs[2]: 
                errs_swift_list = dict_errs_swift['2']
            elif np.max(self._model_rate)<bins_swift_errs[3]: 
                errs_swift_list = dict_errs_swift['3']
            elif np.max(self._model_rate)<bins_swift_errs[4]: 
                errs_swift_list = dict_errs_swift['4']
            elif np.max(self._model_rate)<bins_swift_errs[5]: 
                errs_swift_list = dict_errs_swift['5']
            elif np.max(self._model_rate)<bins_swift_errs[6]: 
                errs_swift_list = dict_errs_swift['6']
            else: 
                errs_swift_list = dict_errs_swift['7']
            #
            grb_index             = np.random.randint(len(errs_swift_list))
            errors_to_apply       = np.array(errs_swift_list[grb_index])
            max_err_index         = len(errors_to_apply)
            std_bkg               = np.array([errors_to_apply[np.random.randint(0,max_err_index)] for val in self._model_rate])
            self._plot_lc         = np.random.normal(loc=self._model_rate, scale=std_bkg) # total COUNTS RATE (signal+noise) with Gauss
            self._plot_lc_with_bg = self._plot_lc  
            self._err_lc          = std_bkg 
            # Since we set the noise as gaussian using std from real data, we
            # don't need to add the background level to the model
            # if self._with_bg: # lc with background
            #     pass
            # else: # background-subtracted lc
            #     self._plot_lc = self._plot_lc - (self._bg * self._res)   # total COUNTS (removed the constant bkg level)

        self._get_lc_properties()

        assert self._n_pulses==len(self._lc_params)

        #for p in self._lc_params:
        #    p['norm'] *= 0
        norms         = np.empty((0,))
        t_delays      = np.empty((0,))
        taus          = np.empty((0,))
        tau_rs        = np.empty((0,))
        #counts_pulses = np.empty((0,))

        if return_array:
            for p in self._lc_params:
                norms    = np.append(norms,    p['norm'])
                t_delays = np.append(t_delays, p['t_delay'])
                taus     = np.append(taus,     p['tau'])
                tau_rs   = np.append(tau_rs,   p['tau_r'])
                # counts_pulses = np.append(counts_pulses, p['counts_pulse'])   

            return norms, t_delays, taus, tau_rs, self._peak_value

        else:
            return self._lc_params

    #--------------------------------------------------------------------------#

    def plot_lc(self, rescale=True, save=False, name="./plot_lc.pdf", show_duration=False):
        """
        Plots GRB light curve (COUNTS vs time)
        
        :rescale: to rescale the x-axis plotting only lc around T100
        :save: to save the plot to file
        :name: filename (including path) to save the plot
        """

        plt.figure(figsize=(9,6))
        plt.xlabel(r'$T-T_0$ [s]', size=16)
        if self._instrument == 'batse':
            plt.ylabel('Counts', size=16)
        elif self._instrument == 'swift':
            plt.ylabel('Count Rates', size=16)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)

        #self._restore_lc()
        
        plt.step(self._times, self._plot_lc, where='post', color='k')

        if self._with_bg:
            plt.plot(np.linspace(self._t_min, self._t_max, num=2, endpoint=True), [self._bg, self._bg], 'r--')
        else:
            pass
            #plt.plot(np.linspace(self._t_min, self._t_max, num=2, endpoint=True), [       0,        0], 'r--')
        
        if rescale:
            t_i = max(self._t_start - 0.5*self._t100, self._t_min)
            t_f = self._t_stop + 0.5*self._t100
            plt.xlim([t_i, t_f])
            
        if show_duration:
                plt.axvline(x=self._t_start, alpha=0.7, color='blue', label=r'$T_{100}$')
                plt.axvline(x=self._t_stop,  alpha=0.7, color='blue')
                plt.axvline(x=self._t90_i,   alpha=0.7, color='red',  label=r'$T_{90}$')
                plt.axvline(x=self._t90_f,   alpha=0.7, color='red')
                plt.legend(prop={'size': 16})
        
        if save:
            plt.savefig(name)
        
        plt.show()

    #--------------------------------------------------------------------------#
    
    def _get_lc_properties(self):
        """
        Calculates T90 and T100 durations along with their start and stop times, 
        total number of counts per T100, mean, max, and background count rates.
        """
#   #--------------------------------------------------------------------------#
#   # V1
#   #--------------------------------------------------------------------------#
#        self._aux_index = np.where(self._raw_lc>self._raw_lc.max()*1e-4)
#        #self._aux_index = np.where((self._plot_lc - self._bg) * self._res / (self._bg * self._res)**0.5 >= self._sigma)
#        self._max_snr   = ((self._plot_lc/self._res - self._bg) * self._res / (self._bg * self._res)**0.5).max()
#        self._aux_times = self._times[self._aux_index[0][0]:self._aux_index[0][-1]] # +1 in the index
#        self._aux_lc    = self._plot_lc[self._aux_index[0][0]:self._aux_index[0][-1]] / self._res
#
#        self._t_start = self._times[self._aux_index[0][0]]
#        #self._t_stop = self._times[self._aux_index[0][-1]+1]
#        self._t_stop  = self._times[self._aux_index[0][-1]]
#            
#        self._t100 = self._t_stop - self._t_start
#        
#        self._total_cnts = np.sum(self._aux_lc - self._bg*np.ones(len(self._aux_lc))) * self._res
#                
#        try:
#            # compute T90_i
#            sum_cnt = 0
#            i = 0
#            while sum_cnt < 0.05 * self._total_cnts:
#                sum_cnt += (self._aux_lc[i] - self._bg) * self._res
#                i += 1
#            self._t90_i = self._aux_times[i]
#                                     
#            # compute T90_f
#            sum_cnt = 0
#            j = -1
#            while sum_cnt < 0.05 * self._total_cnts:
#                sum_cnt += (self._aux_lc[j] - self._bg) * self._res
#                j += -1
#            self._t90_f = self._aux_times[j]      
#
#            # Define T90 as the difference between T90_f and T90_i
#            self._t90 = self._t90_f - self._t90_i            
#            self._t90_cnts = np.sum(self._aux_lc[i:j+1] - self._bg) * self._res
#            assert self._t90_i < self._t90_f
#            
#        except:
#            self._t90      = self._t100
#            self._t90_i    = self._t_start
#            self._t90_f    = self._t_stop
#            self._t90_cnts = self._total_cnts
           
#    #--------------------------------------------------------------------------#
#    # V2
#    #--------------------------------------------------------------------------#
#        # LB: I think that `self._plot_lc` should be the background-subtracted
#        self._aux_index = np.where(self._plot_lc>self._plot_lc.max()*1e-4)
#        #self._aux_index = np.where((self._plot_lc - self._bg) * self._res / (self._bg * self._res)**0.5 >= self._sigma)
#        self._max_snr   = ((self._plot_lc_with_bg/self._res - self._bg) * self._res / (self._bg * self._res)**0.5).max()
#        self._aux_times = self._times[self._aux_index[0][0]:self._aux_index[0][-1]] # +1 in the index
#        self._aux_lc    = self._plot_lc_with_bg[self._aux_index[0][0]:self._aux_index[0][-1]] / self._res
#
#        self._t_start = self._times[self._aux_index[0][0]]
#        #self._t_stop = self._times[self._aux_index[0][-1]+1]
#        self._t_stop  = self._times[self._aux_index[0][-1]]
#        self._t100    = self._t_stop - self._t_start
#        
#        self._total_cnts = np.sum(self._aux_lc - self._bg*np.ones(len(self._aux_lc))) * self._res
#        
#        try:
#            # compute T90_i
#            sum_cnt = 0
#            i = 0
#            while sum_cnt < 0.05 * self._total_cnts:
#                sum_cnt += (self._aux_lc[i] - self._bg) * self._res
#                i += 1
#            self._t90_i = self._aux_times[i]
#                                     
#            # compute T90_f
#            sum_cnt = 0
#            j = -1
#            while sum_cnt < 0.05 * self._total_cnts:
#                sum_cnt += (self._aux_lc[j] - self._bg) * self._res
#                j += -1
#            self._t90_f = self._aux_times[j]      
#
#            # Define T90 as the difference between T90_f and T90_i
#            self._t90      = self._t90_f - self._t90_i
#            self._t90_cnts = np.sum(self._aux_lc[i:j+1] - self._bg) * self._res
#            assert self._t90 > 0
#            
#        except:
#            print('Weird stuff happened...')
#            self._t90      = self._t100
#            self._t90_i    = self._t_start
#            self._t90_f    = self._t_stop
#            self._t90_cnts = self._total_cnts
           

    #--------------------------------------------------------------------------#
    # V3
    #--------------------------------------------------------------------------#
       
        self._aux_index = np.where(self._model>self._model.max()*1e-4)
        #self._aux_index = np.where((self._plot_lc - self._bg) * self._res / (self._bg * self._res)**0.5 >= self._sigma)
        self._max_snr   = ((self._plot_lc_with_bg/self._res - self._bg) * self._res / (self._bg * self._res)**0.5).max()
        self._aux_times = self._times[self._aux_index[0][0]:self._aux_index[0][-1]] # +1 in the index
        # as `self._aux_lc` lc, we use the 'model' one, which is just the sum of the single pulses
        self._aux_lc    = self._model[self._aux_index[0][0]:self._aux_index[0][-1]] / self._res  # count RATES

        self._t_start = self._times[self._aux_index[0][0]]
        #self._t_stop = self._times[self._aux_index[0][-1]+1]
        self._t_stop  = self._times[self._aux_index[0][-1]]
        self._t100    = self._t_stop - self._t_start
        
        self._total_cnts = np.sum(self._aux_lc) * self._res
        #self._total_cnts = np.sum(self._aux_lc - self._bg*np.ones(len(self._aux_lc))) * self._res
                
        try:
            # compute T90_i
            sum_cnt = 0
            i = 0
            while sum_cnt < 0.05 * self._total_cnts:
                sum_cnt += (self._aux_lc[i]) * self._res
                #sum_cnt += (self._aux_lc[i] - self._bg) * self._res
                i += 1
            if i!=0:
                i-=1
            self._t90_i = self._aux_times[i]
                                     
            # compute T90_f
            sum_cnt = 0
            j = -1
            while sum_cnt < 0.05 * self._total_cnts:
                sum_cnt += (self._aux_lc[j]) * self._res
                #sum_cnt += (self._aux_lc[j] - self._bg) * self._res
                j += -1
            if j!=-1:
                j+=1
            self._t90_f = self._aux_times[j]      

            # Define T90 as the difference between T90_f and T90_i
            self._t90      = self._t90_f - self._t90_i
            self._t90_cnts = np.sum(self._aux_lc[i:j+1]) * self._res
            #self._t90_cnts = np.sum(self._aux_lc[i:j+1] - self._bg) * self._res
            #if(self._t90<=0):
            #    print(self._t90)
            assert self._t90 > 0
 
        except:
            #print('Weird stuff happened...')
            self._t90      = self._t100
            self._t90_i    = self._t_start
            self._t90_f    = self._t_stop
            self._t90_cnts = self._total_cnts
    #--------------------------------------------------------------------------#

    @property
    def T90(self):
        return "{:0.3f}".format(self._t90), "{:0.3f}".format(self._t90_i), "{:0.3f}".format(self._t90_f)
    
    @property
    def T100(self):
        return "{:0.3f}".format(self._t100), "{:0.3f}".format(self._t_start), "{:0.3f}".format(self._t_stop)
    
    @property
    def total_counts(self):
        return "{:0.2f}".format(self._total_cnts)
    
    @property
    def max_rate(self):
        return "{:0.2f}".format(self._aux_lc.max())
    
    @property
    def mean_rate(self):
        return "{:0.2f}".format(np.mean(self._aux_lc))
    
    @property
    def bg_rate(self):
        return "{:0.2f}".format(self._bg)
    
    @property
    def max_snr(self):
        return "{:0.2f}".format(self._max_snr)
    
    #--------------------------------------------------------------------------#
#    #TODO LB: THIS METHOD HAS TO BE UPDATED TO THE NEW VERSION OF THE CODE!
#    def _restore_lc(self):
#        """Restores GRB LC from avalanche parameters.
#        Here we are plotting the count RATES, not the counts!"""
#        
#        self._raw_lc = np.zeros(len(self._times))
#        
#        for par in self._lc_params:
#            norm          = par['norm']
#            t_delay       = par['t_delay']
#            tau           = par['tau']
#            tau_r         = par['tau_r']
#            self._raw_lc += self.norris_pulse(norm, t_delay, tau, tau_r)
#
#        if self._with_bg:
#            self._plot_lc = (self._raw_lc * self._ampl * self._eff_area) + self._bg # total count rates (signal+bkg)
#            self._plot_lc = np.random.poisson( self._res * self._plot_lc )          # total count (signal+bkg) with Poisson
#            self._plot_lc = self._plot_lc / self._res                               # total count rates (signal+bkg) with Poisson
#        else:
#            self._plot_lc = (self._raw_lc * self._ampl * self._eff_area) + self._bg # total count rates (signal+bkg)
#            self._plot_lc = np.random.poisson( self._res * self._plot_lc )          # total count (signal+bkg) with Poisson
#            self._plot_lc = self._plot_lc / self._res                               # total count rates (signal+bkg) with Poisson
#            self._plot_lc = self._plot_lc - self._bg                                # total count rates (signal) with Poisson
#
#        self._get_lc_properties()
        
    #--------------------------------------------------------------------------#

    def hdf5_lc_generation(self, outfile, overwrite=False, seed=SEED):
        """
        Generates a new avalanche and writes it to an hdf5 file
        
        :n_lcs: number of light curves we want to simulate
        :outfile: file name
        :overwrite: overwrite existing file
        :seed: random seed for the avalanche generation, int or list
        """
        
        if overwrite == False:
            assert os.path.isfile(outfile), 'ERROR: file already exists!'

        self._f = h5py.File(outfile, 'w')

        
        self._f.create_group('GRB_PARAMETERS')
        self._f['GRB_PARAMETERS'].attrs['PARAMETER_ORDER'] = '[K, t_start, t_rise, t_decay]'

        self._grb_counter = 1
            
        if isinstance(seed, list):
            for sd in seed:
                self.aux_hdf5(seed=sd)
                
        else:
            self.aux_hdf5(seed=seed)

        self._f.close()
        
    #--------------------------------------------------------------------------#
        
    def aux_hdf5(self, seed):
        norms, t_delays, taus, tau_rs, peak_value = self.generate_avalanche(seed=seed, return_array=True)
        n_pulses = norms.size

        grb_array = np.concatenate((
                    norms.reshape(n_pulses,1),
                    t_delays.reshape(n_pulses,1),
                    tau_rs.reshape(n_pulses,1),
                    taus.reshape(n_pulses,1)),
                    axis=1
                    )

        self._f.create_dataset(f'GRB_PARAMETERS/GRB_{self._grb_counter}', data=grb_array)
        self._f[f'GRB_PARAMETERS/GRB_{self._grb_counter}'].attrs['PEAK_VALUE'] = peak_value
        self._f[f'GRB_PARAMETERS/GRB_{self._grb_counter}'].attrs['N_PULSES']   = n_pulses
        self._grb_counter += 1


#==============================================================================#
# Define the class Restore_LC
#==============================================================================#
    
class Restored_LC(LC):
    """
    Class to restore an avalanche from yaml file
    
    :res: GRB LC time resolution
    """
    
    def __init__(self, par_list, res=0.256, t_min=-10, t_max=1000, sigma=5):
        
        super(Restored_LC, self).__init__(res=res, t_min=t_min, t_max=t_max, sigma=sigma)

        if not par_list:
            raise TypeError("Avalanche parameters should be given")
        elif not isinstance(par_list, list):
            raise TypeError("The avalanche parameters should be a list of dictionaries")
        else:
            self._lc_params = par_list
 
        self._restore_lc()

#==============================================================================#
#==============================================================================#