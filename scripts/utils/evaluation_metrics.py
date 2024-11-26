import numpy as np


def mse(err: np.ndarray) -> float:
    mse_predicted = np.mean(err**2)
    return mse_predicted


def rmse(err: np.ndarray) -> float:
    rmse_predicted = np.sqrt(mse(err))
    return rmse_predicted


def mae(err: np.ndarray) -> float:
    mae_predicted = np.mean(np.abs(err))
    return mae_predicted


def run_metrics(true: np.ndarray, predict: np.ndarray) -> dict:
    ede = euclidean_distance_error(predict, true)
    mse_res = mse(ede)
    rmse_res = rmse(ede)
    mae_res = mae(ede)
    return {"MSE": mse_res, "RMSE": rmse_res, "MAE": mae_res}


def get_metrics(true: np.ndarray, predict: np.ndarray) -> dict:
    if predict.dtype == object:
        total_dict = {}
        for i in range(predict.shape[0]):
            result = run_metrics(true[i], predict[i])
            for key, value in result.items():
                total_dict[key] = total_dict.get(key, 0) + value
        return {key: value / len(predict) for key, value in total_dict.items()}
    else:
        return run_metrics(true, predict)


def get_error_in_percents(true: np.ndarray, error: dict) -> dict:
    percent_dict = {}

    for metric in error.keys():
        if metric == 'MSE':
            avg_gt = np.mean(true**2)
        elif metric == 'RMSE':
            avg_gt = np.sqrt(np.mean(true**2))
        elif metric == 'MAE':
            avg_gt = np.mean(true)
        else:
            continue
        percent_dict[metric] = 100 - (error[metric] / avg_gt) * 100

    return percent_dict


def euclidean_distance_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if len(pred.shape) > 1:
        ed = np.sqrt(np.sum((pred[:, :2] - gt[:, :2]) ** 2, axis=1))
    else:
        ed = pred - gt
    return ed
