import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def evaluate(model, X, y):
    start = __import__("time").time()
    preds = model.predict(X)
    return (
        mean_squared_error(y, preds),
        r2_score(y, preds),
        (__import__("time").time() - start) / len(X),
    )


def summarize(results):
    return {
        "mse": results["mse_list"],
        "r2": results["r2_list"],
        "time": results["time_list"],
        "mse_mean": float(np.mean(results["mse_list"])),
        "mse_std": float(np.std(results["mse_list"])),
        "r2_mean": float(np.mean(results["r2_list"])),
        "r2_std": float(np.std(results["r2_list"])),
        "time_mean": float(np.mean(results["time_list"])),
    }
