#!/usr/bin/python3.11

import covasim as cv
import datetime
import os
import argparse

cv.options.set(show=False)

dir_path = f'{os.path.dirname(os.path.realpath(__file__))}/Saved_Sims'

basepars = dict(
    pop_size = 218e3,
    pop_infected = 10e1,
    start_day = '2025-01-01',
    n_days = 365,
    rand_seed = 4
    )

metapars = dict(
    n_runs = 100
)

def vaccine(rollout_day):
    prob = cv.historical_vaccinate_prob.estimate_prob(duration=7,coverage=0.8)
    return cv.vaccinate_prob(vaccine='pfizer',days=rollout_day,prob=prob)

def change_beta(day, factor, layers=None):
    return cv.change_beta(days=day,changes=factor,layers=layers)

mask_non_strict = change_beta(day=21,factor=0.85)
mask_strict = change_beta(day=21,factor=0.50)
early_vaccine = vaccine(rollout_day=21)
late_vaccine = vaccine(rollout_day=42)

scenarios = {
    'Baseline': {
        'interventions': [],
        'label': 'Baseline',
    },
    'Mask(non-strict)': {
        'interventions': [mask_non_strict],
        'label': 'Mask (non-strict)',
    },
    'Mask(strict)': {
        'interventions': [mask_strict],
        'label': 'Mask (strict)',
    },
    'Vaccine(early)': {
        'interventions': [early_vaccine],
        'label': 'Vaccine (early)',
    },
    'Vaccine(late)': {
        'interventions': [late_vaccine],
        'label': 'Vaccine (late)',
    },
    'Mask(non-strict),Vaccine(late)': {
        'interventions': [mask_non_strict, late_vaccine],
        'label': 'Mask (non-strict), Vaccine (late)',
    }
}

def run_multisim_for_scenario(scenario_name):
    if scenario_name not in scenarios:
        raise ValueError(f"Scenario '{scenario_name}' not found. Available: {list(scenarios.keys())}")

    pars = basepars.copy()
    pars['interventions'] = scenarios[scenario_name]['interventions']

    t0 = datetime.datetime.now()
    sim = cv.Sim(**pars)
    sim.label = scenarios[scenario_name]['label']   
    msim = cv.MultiSim(sim, n_runs=metapars['n_runs'])
    msim.run(parallel=False)
    t1 = datetime.datetime.now()

    msim.median()

    time_stamp = t1.strftime("%Y-%m-%d_at_%H-%M-%S")
    filename = f'Simulation_{scenario_name.replace(" ", "_")}_{time_stamp}'

    msim.save(f'{dir_path}/{filename}.sim')
    msim.to_excel(f'{dir_path}/{filename}.xlsx')

    fig = msim.plot(color_by_sim=True)
    img_path = f'{dir_path}/{filename}.png'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    fig.savefig(img_path, dpi=1200)

    print(f"Finished scenario: {scenario_name}")
    print(f"Total simulation time: {t1 - t0}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, required=True, help="Scenario name to run")
    args = parser.parse_args()

    run_multisim_for_scenario(args.scenario)