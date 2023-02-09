import numpy as np
import pandas as pd
from covid_survival.fixed_label_transform import LabTransCoxTime
from covid_survival.train_test_tools import sample_fast, split_data_stratified, split_data_by_date
from helpers import odsmiduj


def get_all_data(path, keep_columns=None, to_onehot=None, drop_columns=None, time_var_columns=None,
                 sample_size=0.05, random_state=42, use_only_flags=False,
                 sample_method='default', validation_date=None, test_date=None, stratify_column=None, version=2,
                 how='raw', const=None, event_column="Infected"):

    data = load_smid_data(path, keep_columns=keep_columns, version=version, event_column=event_column)

    if to_onehot is not None:
        data = onehot_columns(data, to_onehot)

    return get_x_y(data, drop_columns=drop_columns, time_var_columns=time_var_columns,
                   sample_size=sample_size, random_state=random_state,
                   sample_method=sample_method, validation_date=validation_date, test_date=test_date,
                   stratify_column=stratify_column, use_only_flags=use_only_flags, how=how, const=const)


def get_dataset_part(dataset, what='train'):
    return dataset[what]['df'], dataset[what]['x'], dataset[what]['x_time_var'], dataset[what]['y']


def filter_data_by_index(dataset, index):
    df, x, x_time_var, y = dataset
    df = df.iloc[index]
    x = x[index]
    x_time_var = x_time_var[index] if x_time_var is not None else None
    target = [y[i][index] for i in range(4)]
    y = (*target, y[-1]) if len(y) > 4 else target
    return df, x, x_time_var, y


def load_smid_data(path, keep_columns=None, version=2, event_column="Infected"):
    df = pd.read_csv(path)
    df = odsmiduj(df)

    # select only relevant columns
    if keep_columns is None:
        if version == 1:
            keep_columns = ['T1', 'T2', 'Infected', 'InfPrior', 'VaccStatus', 'AgeGr', 'Sex']
        elif version == 2:
            keep_columns = ['T1', 'T2', 'Infected', 'InfPrior', 'InfPriorSeverity', 'VaccStatus', 'Age', 'Sex']
        else:
            raise ValueError(f"Unknown data version: {version}")

    df.drop(columns=[c for c in df.columns if c not in keep_columns], inplace=True)
    df.rename(columns={'T2': 'duration', event_column: 'event'}, inplace=True)

    return df


def get_starts_vaccmap(y_target):
    _, _, starts, vaccmap, _ = y_target
    return starts, vaccmap


def onehot_columns(df, columns):
    return pd.get_dummies(df, columns=columns)


def _sample_by_method(data, sample_size=0.05, random_state=42, sample_method='default', validation_date=None,
                      stratify_column=None):
    if sample_method == 'default':
        return sample_fast(data, sample_size=sample_size, random_state=random_state)
    elif sample_method == 'by_date':
        return split_data_by_date(data, validation_date)
    elif sample_method == 'stratify':
        stratify_column = stratify_column if stratify_column is not None else "VaccStatus"
        return split_data_stratified(data, test_size=sample_size, column=stratify_column, random_state=random_state)
    else:
        raise ValueError("Invalid sample method, allowed: default, by_date, stratify")


def _train_val_test_split(df, validation_date=None, test_date=None, **kwargs):
    # TODO funguje to i pro split by date?
    df_test, df_train = _sample_by_method(df, validation_date=test_date, **kwargs)
    df_val, df_train = _sample_by_method(df_train, validation_date=validation_date, **kwargs)
    return df_train, df_val, df_test


def _get_x(df, to_drop):
    return df.drop(columns=to_drop).astype('float32').to_numpy()


def _get_modified_target(df, y, is_val=False):
    vaccmap = (df['T1'] > 0)

    starts = df['T1'].to_numpy()
    return (*y, starts, vaccmap.reset_index(drop=True), is_val)


def _get_time_vars(df: pd.DataFrame, time_var_columns, max_val, replace_val='_none'):
    x_time_vars = df[time_var_columns]
    assert replace_val in x_time_vars.values, f"The data format probably changed again, {replace_val} is not in timevars."

    x_flags = (x_time_vars != replace_val).astype(int).copy()
    x_flags.columns = [f"{c}_flag" for c in x_flags.columns]
    x_time_vars = x_time_vars.replace(to_replace=replace_val, value=max_val)
    df = pd.concat([df, x_flags], axis=1)
    return df, x_time_vars.astype('float32').to_numpy()


def compute_n_events(df, how='raw', const=None):
    df = df[['duration', 'event']].groupby('duration').count()
    return _return_scaled(df, how, const)


def get_x_y(df, drop_columns=None, time_var_columns=None, max_val_diff=1000, use_only_flags=False,
            sample_method='default', how='raw', const=None, **kwargs):
    y = ['duration', 'event']
    time_var_columns = [] if time_var_columns is None else time_var_columns

    df_train, df_val, df_test = _train_val_test_split(df, sample_method=sample_method, **kwargs)
    x_var_train, x_var_val, x_var_test = None, None, None

    if sample_method == 'by_date':
        n_events = compute_n_events(df, how=how, const=const)
        avg_counts = compute_moving_avg_counts(df, how=how, const=const)
    else:
        n_events = compute_n_events(df_train, how=how, const=const)
        avg_counts = compute_moving_avg_counts(df_train, how=how, const=const)

    # get the time vars
    if len(time_var_columns):
        max_val = df['duration'].max() + max_val_diff
        df_train, x_var_train = _get_time_vars(df_train, time_var_columns, max_val)
        df_val, x_var_val = _get_time_vars(df_val, time_var_columns, max_val)
        df_test, x_var_test = _get_time_vars(df_test, time_var_columns, max_val)
        if use_only_flags:
            x_var_train, x_var_val, x_var_test = None, None, None

    to_drop = y if drop_columns is None else y + drop_columns + time_var_columns
    x_train = _get_x(df_train, to_drop)
    x_val = _get_x(df_val, to_drop)
    x_test = _get_x(df_test, to_drop)

    # transform the labels, add starts and vaccinated/immunized map
    get_target = lambda data: (data['duration'].values, data['event'].values)
    y_train = get_target(df_train)
    y_val = get_target(df_val)
    y_test = get_target(df_test)

    labtrans = LabTransCoxTime()
    labtrans.fit(*y_train)

    y_train = _get_modified_target(df_train, y_train, False)
    y_val = _get_modified_target(df_val, y_val, True)
    y_test = _get_modified_target(df_test, y_test, True)[:-1]

    # return results as a dict
    keys = ['df', 'x', 'x_time_var', 'y']
    train = (df_train, x_train, x_var_train, y_train)
    val = (df_val, x_val, x_var_val, y_val)
    test = (df_test, x_test, x_var_test, y_test)

    res = {}
    for name, data in zip(["train", "val", "test"], [train, val, test]):
        res[name] = {k: v for k, v in zip(keys, data)}

    res['labtrans'] = labtrans
    res['n_events'] = n_events.astype(np.float32)
    res['counts'] = avg_counts.astype(np.float32)
    return res


def compute_moving_avg_counts(df: pd.DataFrame, how='raw', const=None):
    df = df[["duration", "event"]]
    df = df[df["event"] == 1]

    counts = df.groupby("duration").count()

    full_counts = pd.DataFrame(index=range(0, counts.index.max()))
    full_counts = full_counts.join(counts, how="left")
    counts = full_counts.fillna(0)

    counts = counts.rolling(7, center=True).mean()
    counts = counts.fillna(method='bfill').fillna(method='ffill')
    return _return_scaled(counts, how, const)

def _return_scaled(df, how, const, column='event'):
    col = df[column]
    if how == 'raw':
        pass
    elif how == 'log':
        df[column] = np.log(col + 1)
    elif how == 'linear':
        df[column] = col / (col.sum() if const is None else const)
    else:
        raise ValueError(f"{how} is an invalid method for moving avg count scaling.")
    return df
