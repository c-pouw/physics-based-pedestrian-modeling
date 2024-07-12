# %%

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use(Path.cwd().parent / "physped/conf/science.mplstyle")

path = Path.cwd().parent / "multirun/single_paths/dxy0.4_r-0-5-10-15-20-30_ntheta4/"
# print(path.is_dir())
list_kl_divergences = []

job_folders = path.glob("job*")
job_folders = list(job_folders)
job_folders.sort()
for job_folder in job_folders:
    try:
        print(job_folder)
        with open(job_folder / "kl_divergence.pkl", "rb") as f:
            kl_divergence = pickle.load(f)
        kl_df = pd.DataFrame.from_dict(kl_divergence, orient="index")
        list_kl_divergences.append(kl_df.T)
    except FileNotFoundError:
        print(f"FileNotFoundError: {job_folder}")

kl_divergences = pd.concat(list_kl_divergences, axis=0)

# %%

sweep_sigma = kl_divergences[kl_divergences.tauu == 0.5].copy()
for tau in kl_divergences.tauu.unique():
    sweep_tauu = kl_divergences[kl_divergences.tauu == tau].copy()
    plt.plot(sweep_tauu.noise, sweep_tauu["joint_kl_divergence"], ".-", label=f"$\\tau_u={tau}$ s")

plt.xlabel("$\\sigma$")
plt.ylabel("$D_{KL}$")
plt.legend(bbox_to_anchor=(-0.1, 1.15), loc="upper left", ncol=3)
plt.ylim(0, 0.3)
plt.xlim(0.2, 1)
# %%

sweep_tauu = kl_divergences[kl_divergences.noise == 0.65].copy()
plt.plot(sweep_tauu.tauu, sweep_tauu["joint_kl_divergence"], ".-", label="Joint")
plt.xlabel("$\\tau_u$")
plt.ylabel("$D_{KL}$")
plt.legend(ncol=3)

# %%
