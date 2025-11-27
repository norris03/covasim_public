import covasim as cv
import matplotlib.pyplot as plt
import os

cv.options.set(show=False)

dir_path = f'{os.path.dirname(os.path.realpath(__file__))}/Saved_Sims'

msims = []
dir = os.fsencode(dir_path)
for file in os.listdir(dir):
    simname = os.fsdecode(file)
    if not simname.endswith('sim'):
        continue
    print(simname)
    sim = cv.load(f'{dir_path}/{simname}')
    msims.append(sim)


merged = cv.MultiSim.merge(msims,base=True)
fig = merged.plot(color_by_sim=True)
fig.savefig(
    "Merged_image.png",
    dpi = 1200
)