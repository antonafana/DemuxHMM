import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "sweeps/grid_sweep/"   # directory with params_run_{i}.json and results_run_{i}.csv
PARAM_X = "avg_UMI"           # name of first parameter in JSON
PARAM_Y = "num_generations"           # name of second parameter in JSON
PARAM_X_LABEL = "UMI Per Cell"     # human-readable name for x-axis
PARAM_Y_LABEL = "Number of Generations"     # human-readable name for y-axis
REPEATS = 6 # The number of repeats to the data

# Font settings for readability in LaTeX
plt.rcParams.update({
    "font.size": 25,       # Base font size
    "axes.titlesize": 25,  # Title font size
    "axes.labelsize": 20,  # Axis label font size
    "xtick.labelsize": 14, # X tick font size
    "ytick.labelsize": 14, # Y tick font size
    "figure.titlesize": 20
})
# Collect all runs
records = []

for i in range(REPEATS):
    repeat_dir = os.path.join(DATA_DIR, f"run_{i}/")
    for filename in os.listdir(repeat_dir):
        if filename.startswith("params_run_") and filename.endswith(".json"):
            run_id = filename[len("params_run_"):-len(".json")]
            json_path = os.path.join(repeat_dir, filename)
            csv_path = os.path.join(repeat_dir, f"results_run_{run_id}.csv")

            if not os.path.exists(csv_path):
                continue

            # Read params
            with open(json_path, "r") as f:
                params = json.load(f)

            # Read results
            df_res = pd.read_csv(csv_path)

            hmm_score = df_res.loc[0, "hmm_score"] if "hmm_score" in df_res.columns else None

            records.append({
                PARAM_X: float(params[PARAM_X]),
                PARAM_Y: float(params[PARAM_Y]),
                "hmm_score": hmm_score,
            })

# Convert to DataFrame
df = pd.DataFrame(records)
df = df.groupby([PARAM_Y, PARAM_X], as_index=False).agg(mean_hmm_score=("hmm_score", "mean"))

# Pivot for heatmaps (numeric sort of both axes)
hmm_pivot   = df.pivot(index=PARAM_Y, columns=PARAM_X, values="mean_hmm_score")
hmm_pivot   = hmm_pivot.sort_index(ascending=True).sort_index(axis=1, ascending=True)

# Integer tick labels if possible
x_vals = [int(v) if abs(v - round(v)) < 1e-9 else v for v in hmm_pivot.columns]
y_vals = [int(v) if abs(v - round(v)) < 1e-9 else v for v in hmm_pivot.index]

# Shared color scale
vmin, vmax = 0, 1

# Only HMM available
fig, ax = plt.subplots(1, 1, figsize=(7, 6))
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

sns.heatmap(hmm_pivot, ax=ax, cmap="viridis", annot=True, fmt=".2f",
            vmin=vmin, vmax=vmax, cbar=True, cbar_ax=cbar_ax,
            xticklabels=x_vals, yticklabels=y_vals, annot_kws={"size": 12})

ax.set_xlabel("")
ax.set_ylabel("")
cbar_ax.set_title("ARI", fontsize=10)

# Shared labels
fig.text(0.5, 0.02, PARAM_X_LABEL, ha='center', va='center',
         fontsize=plt.rcParams["axes.labelsize"])
fig.text(0.06, 0.5, PARAM_Y_LABEL, ha='center', va='center', rotation='vertical',
         fontsize=plt.rcParams["axes.labelsize"])

plt.savefig("figures/heatmap.pdf", bbox_inches="tight")
plt.show()

