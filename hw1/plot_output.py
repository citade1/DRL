import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT_DIR = "data"
TAGS = ["Eval_AverageReturn", "Eval_StdReturn", "Eval_AverageEpLen", "Train_AverageReturn", "Train_StdReturn", "Train_AverageEpLen"]
ENV_COLORS = {
    "Ant-v4": "#1f77b4",         # blue
    "HalfCheetah-v4": "#2ca02c", # green
    "Hopper-v4": "#9467bd",      # purple
    "Walker2d-v4": "#ff7f0e"     # orange
}


def load_scalar(logdir, tag):
    ea = EventAccumulator(logdir)
    ea.Reload()
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values

def is_bc_run(dirname):
    return dirname.startswith("q1_") and ("dagger" not in dirname)

def is_dagger_run(dirname):
    return dirname.startswith("q2_")

def extract_env_name(dirname):
    for part in dirname.split("_"):
        if "-v" in part:
            return part
    return None

def extract_steps(dirname):
    # Ex: q1_bc_walker2d_steps5000_...
    for part in dirname.split("_"):
        if part.startswith("steps"):
            return int(part.replace("steps", ""))
    return None

def find_expert_bc_return():
    
    experts = {}
    bcs = {}

    for dirname in os.listdir(ROOT_DIR):
        dirpath = os.path.join(ROOT_DIR, dirname)
        if not os.path.isdir(dirpath):
            continue

        env_name = extract_env_name(dirname)
        if env_name is None:
            continue
        
        if is_bc_run(dirname):
            _, train_vals = load_scalar(dirpath, "Train_AverageReturn")
            _, eval_vals = load_scalar(dirpath, "Eval_AverageReturn")

            experts[env_name] = train_vals[-1]
            bcs[env_name] = eval_vals[-1]
    
    return experts, bcs


def build_bc_table():
    results = {}
    experts, bc_evals = find_expert_bc_return()
    
    for dirname in os.listdir(ROOT_DIR):
        dirpath = os.path.join(ROOT_DIR, dirname)
        if not os.path.isdir(dirpath):
            continue
        if not is_bc_run(dirname):
            continue

        env_name = extract_env_name(dirname)
        results[env_name] = {}

        for tag in TAGS:
            _, values = load_scalar(dirpath, tag)
            results[env_name][tag] = values[-1]
        
        expert = experts[env_name]
        bc_eval = bc_evals[env_name]
        percent = float(100 * bc_eval / expert)
        results[env_name]["BC/Exeprt Percentage"] = percent
    

    df = pd.DataFrame(results)
    return df.T

def plot_dagger():
    fig, axes = plt.subplots(1, 1, figsize=(12,8))
    
    experts, bc_evals = find_expert_bc_return()

    used_bc = set()
    used_expert = set()
    for dirname in os.listdir(ROOT_DIR):
        if not is_dagger_run(dirname):
            continue
        
        dirpath = os.path.join(ROOT_DIR, dirname)

        env = extract_env_name(dirname)
        color = ENV_COLORS.get(env, "black")

        steps, vals = load_scalar(dirpath, "Eval_AverageReturn")
        _, std_vals = load_scalar(dirpath, "Eval_StdReturn")

        axes.plot(steps, vals, label=env, color=color)
        axes.fill_between(steps, vals-std_vals, vals+std_vals, color=color, alpha=0.15)

        if env not in used_bc:
            axes.axhline(bc_evals[env], color=color, linestyle="--", label=f"BC_{env}", alpha=0.6)
            used_bc.add(env)
        if env not in used_expert:
            axes.axhline(experts[env], color=color, linestyle="-.", label=f"Expert_{env}", alpha=0.6)
            used_expert.add(env)

    
    axes.set_title("DAgger learning curves")
    axes.set_xlabel("iteration")
    axes.set_ylabel("return")
    axes.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig("dagger_learning_curves.png")
    plt.show()



def save_table_as_image(df, filename):

    fig, ax = plt.subplots(figsize=(10,4))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        rowLabels=df.index,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(True)
    fig.tight_layout()
    plt.savefig(filename, dpi=300)


def plot_bc_hparam_curve(env_name="Walker2d-v4"):
    """
    Plot BC performance vs hyperparameter value 
    """

    hparams = []
    eval_returns = []
    eval_stds = []

    for dirname in os.listdir(ROOT_DIR):
        dirpath = os.path.join(ROOT_DIR, dirname)
        if not os.path.isdir(dirpath): 
            continue
        if not is_bc_run(dirname):
            continue
        if env_name not in dirname:
            continue
        steps = extract_steps(dirname)
        if steps is None:
            continue

        
        _, mean_vals = load_scalar(dirpath, "Eval_AverageReturn")
        _, std_vals = load_scalar(dirpath, "Eval_StdReturn")

        hparams.append(steps)
        eval_returns.append(mean_vals[-1])
        eval_stds.append(std_vals[-1])

    # Sort by increasing hyperparameter
    order = np.argsort(hparams)
    hparams = np.array(hparams)[order]
    eval_returns = np.array(eval_returns)[order]
    eval_stds = np.array(eval_stds)[order]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        hparams, eval_returns, yerr=eval_stds,
        fmt="-o", capsize=5, linewidth=2
    )
    plt.xlabel("num_agent_train_steps_per_iter", fontsize=12)
    plt.ylabel("Eval_AverageReturn", fontsize=12)
    plt.title(f"BC Performance vs Training Steps ({env_name})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("bc_hparam_curve.png", dpi=150)
    plt.show()



if __name__=="__main__":
    # df = build_bc_table()
    # print(df)
    # save_table_as_image(df, "bc_results.png")
    # plot_dagger()
    plot_bc_hparam_curve()

