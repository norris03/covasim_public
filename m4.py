import covasim as cv
import numpy as np
from scipy import optimize
import sciris as sc
import argparse
import datetime
import os
cv.options.set(show=False)

dir_path = f'{os.path.dirname(os.path.realpath(__file__))}/four'
if not os.path.exists(dir_path):
        os.makedirs(dir_path)

li_pop = {
  '0-9':  4015,
  '10-19': 4017,
  '20-29': 4723,
  '30-39': 4971,
  '40-49': 5250,
  '50-59': 6118,
  '60-69': 4932,
  '70-79': 3486,
  '80+':   1612,
}

data_file_name = 'data.csv'

# Data taken from https://www.indexmundi.com/population-pyramid/liechtenstein
# Total population 39124
cv.data.country_age_data.data['Liechtenstein'] = li_pop

standard_pars = dict(
    pop_size = 39_124, 
    pop_type = 'hybrid',
    location = 'Liechtenstein',
    n_days = 365,
    verbose = 0,
)

def scipy_obj(x, n_runs=10):
    print(f'Running sim for beta={x[0]}, rel_death_prob={x[1]}')
    pars = standard_pars | dict(
        beta           = x[0],
        rel_death_prob = x[1],
    )
    sim = cv.Sim(pars=pars, datafile=data_file_name)
    msim = cv.MultiSim(sim)
    msim.run(n_runs=n_runs)
    mismatches = []
    for sim in msim.sims:
        fit = sim.compute_fit(fitmethod='mse')
        mismatches.append(fit.mismatch)
    mismatch = np.mean(mismatches)
    global best
    if mismatch < best:
        best = mismatch
    global t0
    print(f'Mismatch of {mismatch} (best {best}) at time {datetime.datetime.now()-t0}')
    if x[0] < 0 or x[1] < 0:
        return 10e10
    return mismatch  

def scipy_calibration(maxiter=100, n_runs=10):
    guess = [0.016, 1]
    pars = optimize.minimize(lambda x: scipy_obj(x, n_runs=n_runs), x0=guess, method='nelder-mead',options={'maxiter':maxiter, 'fatol':1e-10, 'xatol':1e-10})
    print(pars)

def optuna_calibration(total_trials=100):        
    pars = sc.objdict(standard_pars | dict(beta = 0.016, rel_death_prob = 1))

    sim = cv.Sim(pars=pars, datafile=data_file_name)

    # Parameters to calibrate -- format is best, low, high
    calib_pars = dict(
        beta           = [pars.beta, 0.016/10, 0.016*10],
        rel_death_prob = [pars.rel_death_prob, 0.1, 10.0],
    )

    calib = sim.calibrate(
        calib_pars=calib_pars, 
        total_trials=total_trials,
        fit_args=dict(
            fitmethod = 'mse'
        )
    )

    return calib

def scipy_method(maxiter=400, n_runs=10):
    print(f'Maxiter: {maxiter}, n_runs: {n_runs}')
    global t0
    global best
    best = 10e10
    t0 = datetime.datetime.now()
    scipy_calibration(maxiter=maxiter,n_runs=n_runs)
    t1 = datetime.datetime.now()
    print(f'Total time for scipy optimization with {n_runs} runs (and maximum of {maxiter} iterations): {t1-t0}')

def optuna_method(total_trials=100):
    print(f'Total trials: {total_trials}')
    t0 = datetime.datetime.now()
    calib = optuna_calibration(total_trials=total_trials)
    t1 = datetime.datetime.now()
    print(f'Total time for optuna optimization with {total_trials} total trials: {t1-t0}')
    fig = calib.plot_sims(to_plot=['cum_infections','cum_deaths']) 
    #fig.savefig(f'Optuna_calibration_trials={args.trials}.png', dpi=1200)
    filename = f'Optuna_calibration_trials={args.trials}.png'
    img_path = f'{dir_path}/{filename}.png'  
    fig.savefig(img_path, dpi=1200)
    print(f'Created figure')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", 
        type=str, 
        required=True,
        help="Optimization method, either 'scipy' or 'optuna'"
    )
    parser.add_argument(
        "--trials", # Optuna parameter 
        type=int, 
        default = 100,
        help="Number total_trials for built-in Optuna calibration (default=100)"
    )
    parser.add_argument(
        "--maxiter", # Scipy parameter
        type = int,
        default = 400,
        help="Maximum number of iterations for scipy calibration (default=400)"
    )
    parser.add_argument(
        "--runs", # Scipy parameter
        type = int,
        default = 10,
        help="Number of n_runs for scipy calibration (per parameter values) (default=10)"
    )
    args = parser.parse_args()

    if args.method == 'scipy':
        scipy_method(maxiter=args.maxiter, n_runs=args.runs)
    elif args.method == 'optuna':
        optuna_method(total_trials=args.trials)
    else:
        print('No method selected')
    
    #calib = optuna_calibration(total_trials=100)
    #fig = calib.plot_sims(to_plot=['cum_infections','cum_deaths'])
    #fig.savefig(f'Optuna_calibration(trials={args.trials}).png', dpi=1200)

    #sim = cv.Sim(standard_pars, datafile='data.csv')
    #sim.initialize()
    #sim.run()
    #sim.plot(to_plot=['cum_infections','cum_deaths'])
    #sim.plot()