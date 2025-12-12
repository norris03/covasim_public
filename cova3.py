# Seite 94 SALTELLI_2004
import covasim as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import argparse


class Morris:
    def __init__(self, k, p=4, r=10):
        self.k = k  # number of parameters
        self.p = p  # grid level
        self.r = r  # number of trajectories
        self.step_size = self.p / (2 * (self.p - 1))
        self.grid = np.array([c * 1 / (self.p - 1) for c in range(self.p)])  # grid points
        self.B = np.array(
            [[1 if i > j else 0 for j in range(self.k)] for i in range(self.k + 1)])  # lower (ones) triangular matrix
        self.Jk = np.ones(shape=[self.k + 1, self.k])  # ones matrix
        self.J1 = np.ones(shape=[self.k + 1, 1])  # ones vector
        self.all_trajectories = []
        self.all_uniform_trajectories = []

    def _generate_trajectory_matrices(self):
        P = np.identity(self.k)  # permutation matrix
        np.random.shuffle(P)
        D = np.diag(np.random.choice([1, -1], size=self.k))  # diagonal matrix with random 1 or -1 entries
        return P, D

    def _generate_uniform_trajectory(self):
        x = np.random.choice([x for x in self.grid if x <= 1 - self.step_size], size=self.k).reshape(1, self.k)
        P, D = self._generate_trajectory_matrices()
        trajectory = (self.J1 @ x
                      + self.step_size / 2 * ((2 * self.B - self.Jk) @ D + self.Jk)) @ P

        return trajectory

    @staticmethod
    def upscale(x, interval):
        # Transform x in [0,1] to [a,b]
        if len(interval) != 2:
            raise ValueError(f"Invalid interval: {interval}")
        return x * (interval[1] - interval[0]) + interval[0]

    @staticmethod
    def downscale(x, interval):
        # Transform x in [a,b] to [0,1]
        if len(interval) != 2:
            raise ValueError(f"Invalid interval: {interval}")
        return (x - interval[0]) / (interval[1] - interval[0])

    def _generate_trajectory(self, list_of_intervals=None):
        if list_of_intervals == None: list_of_intervals = [[0, 1] for _ in range(self.k)]
        if len(list_of_intervals) != self.k:
            raise ValueError(f"Number of intervals {len(list_of_intervals)} "
                             f"does not match number of parameters {self.k}")
        trajectory = self._generate_uniform_trajectory()
        uniform_trajectory = trajectory.copy()
        for j in range(self.k):
            interval = list_of_intervals[j]
            for i in range(self.k + 1):
                trajectory[i, j] = Morris.upscale(x=trajectory[i, j], interval=interval)
        return trajectory, uniform_trajectory

    def generate_all_trajectories(self, list_of_intervals=None):
        if list_of_intervals == None: list_of_intervals = [[0, 1] for _ in range(self.k)]
        if len(list_of_intervals) != self.k:
            raise ValueError(f"Number of intervals {len(list_of_intervals)} "
                             f"does not match number of parameters {self.k}")

        self.all_trajectories = []
        self.all_uniform_trajectories = []
        for _ in range(self.r):
            t, tu = self._generate_trajectory(list_of_intervals=list_of_intervals)
            self.all_trajectories.append(t)
            self.all_uniform_trajectories.append(tu)
            # return self.all_trajectories, self.all_uniform_trajectories

    def get_all_trajectories(self):
        if len(self.all_trajectories) == 0:
            raise RuntimeError(
                f"Must first generate trajectories via Morris.generate_all_trajectories(list_of_intervals)")
        return self.all_trajectories

    def compute_elementary_effects(self, sim_results):
        """
        sim_results: Liste von Listen oder 2D-Array. Jede Trajektorie hat k+1 Sim-Ergebnisse.
        """
        if len(sim_results) != len(self.all_trajectories):
            raise ValueError("Anzahl der Simulationsergebnisse stimmt nicht mit Anzahl der Trajektorien überein.")

        EE = {f"x{j}": [] for j in range(self.k)}

        for traj_index, trajectory in enumerate(self.all_uniform_trajectories):
            Y = np.array(sim_results[traj_index])  # Ergebnisse für diese Trajektorie

            for i in range(self.k):
                delta = self.step_size  # tatsächlicher Schritt
                ee = (Y[i+1] - Y[i]) / delta
                index = np.nonzero(trajectory[i+1]-trajectory[i])[0][0]
                EE[f"x{index}"].append(ee)

        # mu_star = Mittelwert der absoluten EE
        mu_star = {var: np.mean(np.abs(vals)) for var, vals in EE.items()}
        sigma = {var: np.std(vals) for var, vals in EE.items()}

        return EE, mu_star, sigma

#def test_sim(parameter_values):
#    return sum(parameter_values)

"""
SIMULATION
"""

cv.options.set(show=False)
dir_path = f'{os.path.dirname(os.path.realpath(__file__))}/Saved_Morris'

# Funktion zum Ausführen einer Covasim-Simulation
def run_covasim(params, n_runs=5, pop_size=10000, n_days=180):
    """
    Run Covasim simulation multiple times for stochastic averaging.
    Returns the average number of deaths across runs.
    """
    #deaths_list = []

    #for _ in range(n_runs):
    sim = cv.Sim(
        pop_size=pop_size,
        n_days=n_days,
        beta=params['beta'],
        rel_death_prob=params['rel_death_prob'],
        asymp_factor=params['asymp_factor'],
        contacts=params['contacts'],
        rand_seed = 4,
        verbose = 0
    )
    msim = cv.MultiSim(sim, n_runs = n_runs)
    msim.run()
    msim.reduce()
    deaths = msim.results['cum_deaths'].values[-1]

    return deaths


# Parameterdefinition für Morris Sensitivity Analysis

problem = {
    'num_vars': 4,
    'names': ['beta', 'rel_death_prob', 'asymp_factor', 'contacts'],
    'bounds': [
        [0.5*0.016, 2*0.016],  # beta: transmission probability 0.005, 0.03
        [0.5, 2.0],  # rel_death_prob: relative mortality 0.2, 1.0
        [0.5, 2.0],  # asymp_factor: scaling asymptomatic infectivity 0.5, 1.5
        [0.5*20, 2.0*20]  # contacts per person 5, 20
    ]
}

if __name__ ==  '__main__': #    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nruns",
        type=int,
        default=5,
        help="Number of runs per simulation (default = 5)"
    )
    parser.add_argument(
        "--p",
        type=int,
        default=4,
        help="Parameter grid level (default = 4)"
    )
    parser.add_argument(
        "--r",
        type=int,
        default=10,
        help="Number of trajectories per simulation (default = 10)"
    )
    args = parser.parse_args()
    n_runs = args.nruns
    number_of_parameters = 4
    grid_level = args.p
    num_trajectories = args.r

    # Morris Designpunkte generieren

    N = 4  # Anzahl der Trajektorien / Designpunkte
    m = Morris(k=number_of_parameters, p=grid_level, r=num_trajectories)
    m.generate_all_trajectories(list_of_intervals=problem['bounds'])
    trajectories = m.get_all_trajectories()

    # Simulationen ausführen

    Y = []
    t0 = datetime.datetime.now()
    #Einzelne Trajektorie besteht aus 4 Parameterkombinationen
    for traj_index, trajectory in enumerate(trajectories):
        sim_results_for_traj = []
        for x in trajectory: # x sind Parameterwerte zu 'einem Zeitpunkt' der Trajektorie
            params_dict = dict(zip(problem['names'], x))
            deaths = run_covasim(params_dict, n_runs=n_runs) # zu jedem der Parameterkombinationen eine Auswertung
            sim_results_for_traj.append(deaths)
        Y.append(sim_results_for_traj)
    # Sensitivitätsanalyse durchführen
    EE, mu_star, sigma = m.compute_elementary_effects(sim_results=Y)
    t1 = datetime.datetime.now()
    time_stamp = t1.strftime("%Y-%m-%d_at_%H-%M-%S")
    filename = f'Morris_{time_stamp}_(n_runs={n_runs},p={grid_level},r={num_trajectories})'


    # Ergebnisse visualisieren

    plt.figure(figsize=(8, 5))
    plt.bar(problem['names'], list(mu_star.values()), color='skyblue')
    plt.ylabel("Morris Sensitivity (mu_star)")
    plt.title("Parameter Sensitivity auf Todesfälle")
    #plt.show()

    fig = plt.gcf()
    img_path = f'{dir_path}/{filename}.png'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    fig.savefig(img_path, dpi=1200)

    print(f"Means: {mu_star}")
    print(f"Stds : {sigma}")

    print(f"Total simulation time: {t1 - t0}")


