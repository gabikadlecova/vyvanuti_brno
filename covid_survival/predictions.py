import os
import pickle

import numpy as np
import torch
import torchtuples as tt
from pycox.models.cox_vacc import CoxVacc

from covid_survival.data import get_starts_vaccmap, get_dataset_part
from covid_survival.model import load_model


def predict_hazards_multiple_times(start_time, end_time, step, model: CoxVacc, x_data, y_data, x_time_var=None,
                                   is_real_time=False, verbose=True, print_freq=1, **kwargs):
    """Predict for multiple timepoints - range(start_time, end_time, step). Also return times of prediction.
    """
    res = {}
    for i, time in enumerate(range(start_time, end_time, step)):
        if verbose and (i % print_freq == 0):
            print(f"Predicting for time {time}")

        pred = _predict_helper(model, time, x_data, y_data, x_time_var=x_time_var, is_real_time=is_real_time, **kwargs)
        res[time] = pred

    return res


def load_or_save_preds(predictions_path, time_from, time_to, step, device, dataset, model_path, is_real_time, labtrans,
                       suffix='', case_count_dict=None):

    if predictions_path is None:
        predictions_path = os.path.splitext(model_path)[0]
        predictions_path = f"{predictions_path}_pred{suffix}{'_real_time' if is_real_time else ''}.pickle"
    else:
        predictions_path = os.path.splitext(predictions_path)
        predictions_path = f"{predictions_path[0]}{suffix}{predictions_path[1]}"

    if not os.path.exists(predictions_path):
        # compute predictions
        print(f"Computing predictions, saving to {predictions_path}.")
        res = compute_and_save_predictions(predictions_path, model_path, dataset, labtrans, device=device,
                                           time_from=time_from, time_to=time_to, step=step,
                                           is_real_time=is_real_time, case_count_dict=case_count_dict)
    else:
        # load predictions
        print(f"Loading predictions from {predictions_path}.")
        res = load_predictions(predictions_path)

    # check if the predictions match the file
    keys = list(res.keys())
    assert keys == list(range(time_from, time_to, step))
    assert len(res[keys[0]]) == len(dataset[0])

    return res


def compute_and_save_predictions(save_path, model, dataset, labtrans, device=None, time_from=0, time_to=300, step=20,
                                 is_real_time=False, case_count_dict=None):
    # compute predictions
    device = torch.device(device) if device is not None else None
    if isinstance(model, str):
        model = load_model(model, labtrans, device=device, case_count_dict=case_count_dict)
    df, x, x_time_var, y = dataset

    res = predict_hazards_multiple_times(time_from, time_to, step, model, x, y,
                                                x_time_var=x_time_var, is_real_time=is_real_time)

    with open(save_path, 'wb') as f:
        pickle.dump(res, f)
    return res


def load_predictions(load_path):
    with open(load_path, 'rb') as f:
        res = pickle.load(f)

    if not isinstance(res, dict):
        res = {k: v for k, v in zip(res[1], res[0])}  # old format to new format
    return res


def predict_hazard_since_vacc_inf(model: CoxVacc, T, x_data, y_data, x_time_var=None, **kwargs):
    """Predicts hazard for a subject in time T since last vaccination/infection.

        Args:
            model: CoxVacc model
            T: time in days since vacc/infections
            x_data: x features
            y_data: either the whole y target used during training or a tuple (starts, vaccmap)
            x_time_var: time-varying x features
            **kwargs: other predict parameters (e.g. batch size)

        Returns:
            Predicted hazard rates in time T since vacc/inf.
    """
    return _predict_helper(model, T, x_data, y_data, x_time_var=x_time_var, is_real_time=False, **kwargs)


def predict_hazard_in_real_time(model: CoxVacc, day_since_day0, x_data, y_data, x_time_var=None, **kwargs):
    """Predicts hazard for subjects at a specific point in time (a day - e.g. 1.11.2021). For every subject,
       the time is converted to its own time since last vaccination/infection.

        Args:
            model: CoxVacc model
            day_since_day0: Day since day 0 (depends on the dataset).
            x_data: x features
            y_data: either the whole y target used during training or a tuple (starts, vaccmap)
            x_time_var: time-varying x features
            **kwargs: other predict parameters (e.g. batch size)

        Returns:
            Predicted hazard rates on day since day 0.
    """
    return _predict_helper(model, day_since_day0, x_data, y_data, x_time_var=x_time_var, is_real_time=True, **kwargs)


def _predict_helper(model: CoxVacc, time, x_data, y_data, x_time_var=None, is_real_time=False, **kwargs):
    time_y = np.zeros((len(x_data), 1), dtype=np.float32) + time

    val_pred = tt.tuplefy(x_data, time_y)  # for prediction, we need to tuplefy it with X

    if len(y_data) != 2:
        starts, vaccmap = get_starts_vaccmap(y_data)
    else:
        starts, vaccmap = y_data

    # predict the hazard
    pred_val = model.predict(val_pred, starts, vaccmap, time_var_input=x_time_var, time_is_real_time=is_real_time,
                             **kwargs)

    return pred_val
