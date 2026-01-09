import covasim as cv
import os
import numpy as np

dir_path = f'{os.path.dirname(os.path.realpath(__file__))}/Saved_Sims'

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

cv.data.country_age_data.data['Liechtenstein'] = li_pop

basepars = dict(
    pop_size = 39_124, 
    pop_type = 'hybrid',
    location = 'Liechtenstein',
    beta = 0.00628929212298237,
    rel_death_prob = 0.18680554694728613,
    n_days = 365,
    verbose = 0,
)

if __name__ == "__main__":
    sim = cv.Sim(pars=basepars, datafile=data_file_name)

    msim = cv.MultiSim(sim)
    msim.run(n_runs=10)

    cum_infections_median = np.median([s.results['cum_infections'].values for s in msim.sims], axis=0)
    cum_deaths_median = np.median([s.results['cum_deaths'].values for s in msim.sims], axis=0)

    median_sim = msim.sims[0].copy()
    median_sim.results['cum_infections'].values[:] = cum_infections_median
    median_sim.results['cum_deaths'].values[:] = cum_deaths_median

    fit = median_sim.compute_fit(fitmethod='mse')
    median_sim.plot(to_plot=['cum_infections', 'cum_deaths'])
    fit.plot()
    fit.summarize()
