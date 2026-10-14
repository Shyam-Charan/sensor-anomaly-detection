import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make(X_test, y_test, preds, path):
    fig, axes = plt.subplots(len(preds), 1, figsize=(11, 2.6 * len(preds)), sharex=True)
    for ax, (name, pred) in zip(axes, preds.items()):
        ax.plot(X_test.index, X_test["value"], lw=0.5, color="tab:blue")
        ax.fill_between(X_test.index, X_test["value"].min(), X_test["value"].max(),
                        where=y_test.to_numpy() == 1, color="tab:green", alpha=0.2, label="labelled anomaly window")
        flagged = pred == 1
        ax.scatter(X_test.index[flagged], X_test["value"][flagged], s=3, color="tab:red", label="alert")
        ax.set_title(name, fontsize=10, loc="left")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
