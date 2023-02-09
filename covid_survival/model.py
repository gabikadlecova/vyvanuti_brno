import pickle
import torch
import torchtuples as tt
from torchtuples.callbacks import Callback
from pycox.models.cox_vacc import CoxVacc
from pycox.models.cox_time import MLPVanillaCoxTime
from pycox.models.data import make_at_risk_dict_with_starts


def _load_at_risk_dict(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _sort_durations(target):
    idx_sort = CoxVacc._get_sort_idx(CoxVacc.split_target_starts(target)[0])
    durations, _, starts, _, _ = target
    durations = tt.tuplefy(durations).iloc[idx_sort][0]
    starts = tt.tuplefy(starts).iloc[idx_sort][0]
    return durations, starts


def compute_at_risk_dict(target, save_path=None):
    durs, starts = _sort_durations(target)
    at_risk_dict = make_at_risk_dict_with_starts(durs, starts)

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(at_risk_dict, f)

    return at_risk_dict


def load_model(net_path, labtrans, optimizer=None, lr=0.001, device=None, train_at_risk_path=None,
               val_at_risk_path=None, min_duration=2.0, case_count_dict=None, **kwargs):
    train_dict = None if train_at_risk_path is None else _load_at_risk_dict(train_at_risk_path)
    val_dict = None if val_at_risk_path is None else _load_at_risk_dict(val_at_risk_path)

    optimizer = optimizer if optimizer is not None else tt.optim.Adam
    model = CoxVacc(net_path, optimizer=optimizer, labtrans=labtrans, device=device,
                    train_dict=train_dict, val_dict=val_dict, min_duration=min_duration,
                    case_count_dict=case_count_dict, **kwargs)
    model.optimizer.set_lr(lr)

    return model


def create_model(in_features, labtrans, num_nodes=None, batch_norm=False, dropout=0.1, optimizer=None, lr=0.001,
                 device=None, train_at_risk_path=None, val_at_risk_path=None, case_count_dict=None, **kwargs):

    train_dict = None if train_at_risk_path is None else _load_at_risk_dict(train_at_risk_path)
    val_dict = None if val_at_risk_path is None else _load_at_risk_dict(val_at_risk_path)

    num_nodes = num_nodes if num_nodes is not None else [32, 32, 64]
    optimizer = optimizer if optimizer is not None else tt.optim.AdamWR(decoupled_weight_decay=0.01, cycle_eta_multiplier=0.8,
                                                                        cycle_multiplier=2)

    # just some MLP wrapper
    net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm=batch_norm, dropout=dropout)
    model = CoxVacc(net, optimizer=optimizer, labtrans=labtrans, device=device,
                    train_dict=train_dict, val_dict=val_dict, case_count_dict=case_count_dict, **kwargs)
    model.optimizer.set_lr(lr)

    return model


def train(model: CoxVacc, x_train, y_train, x_time_var_train=None, batch_size=2048, epochs=5, verbose=True,
          x_val=None, y_val=None, x_time_var_val=None, val_batch_size=2048, shuffle=True, n_control=100,
          checkpoint_name=None):

    val = x_val, y_val

    class Flush(Callback):
        def on_epoch_end(self):
            print(flush=True)
            
    if checkpoint_name is not None:
        class Checkpoint(Callback):
            def __init__(self):
                super().__init__()
                self.i = 0
            def  on_epoch_end(self):
                if self.i % 100 == 0:
                    self.model.save_net(f"{checkpoint_name}_{self.i}.pt")
                self.i += 1
        callbacks = [Checkpoint(), Flush()]
    else: 
        callbacks = [Flush()]
        
    log = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose,
                    val_data=val, val_batch_size=val_batch_size, shuffle=shuffle, n_control=n_control,
                    time_var_input=x_time_var_train, val_time_var_input=x_time_var_val,
                    callbacks=callbacks)

    return log
