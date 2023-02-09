import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from helpers import filter_vaxx, filter_nonvaxx, filter_women, filter_men
from helpers import filter_experienced, filter_unexperienced, filter_vacc_by_dose, filter_age_group


def infected_right_now(df, time, step=1):
    """ Get people who got covid at this time + n days, n == step"""
    return df[(df['duration'] >= time) & (df['duration'] < time + step) & (df['event'] == 1)]


def compute_ratio_metric(filtered_data, timepoints, print_freq=10, step=1):
    """Compute metrics for data filtered by `filter_by_groups`

        Args:
            filtered_data: List of dicts - predictions filtered using `filter_by_groups`.
                Should be created using predict(..., is_real_time=True)

            timepoints: List of time values - real time in days since day 0. Should match time that was used to create
                predictions in `filtered_data`

            print_freq: print frequency
            step: count real infected across n days (n == step) - smoothing purposes

    """
    res_pred = []
    res_true = []

    for preds, time in zip(filtered_data, timepoints):
        # get prediction data
        vacc_all = preds['selected_df'].reset_index(drop=True)
        ref_all = preds['reference_df'].reset_index(drop=True)
        vacc_x = preds['selected_preds']
        ref_x = preds['reference_preds']

        # get people at risk in this time interval
        vacc = vacc_all[(vacc_all['T1'] <= time) & (vacc_all['duration'] >= time)]
        ref = ref_all[(ref_all['T1'] <= time) & (ref_all['duration'] >= time)]

        if not len(vacc) or not len(ref):
            print(f"Skipping time {time} because there are no subjects in one of the groups.")
            res_pred.append(None)
            res_true.append(None)
            continue

        vacc_preds = vacc_x[vacc.index]
        ref_preds = ref_x[ref.index]

        # compute predicted vs true ratios
        hazard_ratio = np.exp(np.mean(vacc_preds)) / np.exp(np.mean(ref_preds))
        vacc_inf, ref_inf = infected_right_now(vacc, time, step), infected_right_now(ref, time, step)

        if len(ref_inf) > 0:
            true_infected_ratio = (len(vacc_inf) / len(vacc)) / (len(ref_inf) / len(ref))
        else:
            true_infected_ratio = f"{(len(vacc_inf) / len(vacc))} / {(len(ref_inf) / len(ref))}"

        if time % print_freq == 0:
            print(f"Time {time}, hazard ratio: {hazard_ratio}, true ratio: {true_infected_ratio} | "
                  f"(vacc / antivaxx) {len(vacc_inf)} / {len(ref_inf)}")
            print(f"At time {time}, {len(vacc)} people are vaccinated (total {len(vacc) + len(ref)})")
            print()

        res_pred.append(hazard_ratio)
        res_true.append(true_infected_ratio)

    return res_pred, res_true


def efficiency_curve(vacc, reference, ref_only_0=True):
    eff = lambda x, y: 1 - np.exp(np.mean(x)) / np.exp(np.mean(y))

    if ref_only_0:
        curve = [eff(v_hazard, reference) for v_hazard in vacc]
    else:
        curve = [eff(v_hazard, ref_hazard) for v_hazard, ref_hazard in zip(vacc, reference)]

    return curve


def plot_curve(curve, times, save_path=None, title=None, show=False, ymin=None, ymax=None):
    if title is not None:
        plt.title(title)

    plt.plot(times, curve)
    if ymin is not None:
        plt.ylim(bottom=ymin)
    if ymax is not None:
        plt.ylim(top=ymax)

    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)

def plot_curves(curve, times, save_path=None, title=None, show=False, ymin=None, ymax=None):
    if title is not None:
        plt.title(title)

    df = pd.DataFrame(index=times)
    for i, c in enumerate(curve):
        df[i] =c
    df = df.stack().droplevel(level=1).reset_index()
    df.columns = ["time", "val"]
    sns.lineplot(data=df, x="time", y="val")
    if ymin is not None:
        plt.ylim(bottom=ymin)
    if ymax is not None:
        plt.ylim(top=ymax)

    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    return df

def efficiency_curve_by_groups(df, predictions, *args, **kwargs):
    filtered_list = [filter_by_groups(df, *args, predictions=p, **kwargs) for p in predictions]
    vacc = [f['selected_preds'] for f in filtered_list]
    ref = filtered_list[0]['reference_preds']
    return efficiency_curve(vacc, ref, ref_only_0=True)


def filter_by_groups(df, age_from, age_to, predictions=None, dose_no=None, dose_type=None, preinfections=False,
                     use_vacc=True, gender=None):

    df = df.reset_index(drop=True)
    df = filter_age_group(df, age_from, age_to)
    if gender is not None:
        if gender not in ['Z', 'M']:
            raise ValueError("Other genders than [Z, M] are unfortunately not supported by the Czech state.")
        if gender == 'Z':
            df = filter_women(df)
        else:
            df = filter_men(df)

    reference = filter_nonvaxx(filter_unexperienced(df))
    vacc = filter_vaxx(df) if use_vacc else filter_nonvaxx(df)

    # filter by dose
    if dose_no is not None:
        assert use_vacc, "use_vacc must be True when filtering by doses"
        assert dose_type is not None, "Can filter only by a specific vaccine type now."
        vacc = filter_vacc_by_dose(vacc, dose_no, dose_type)

    # filter by previous infection
    vacc = filter_experienced(vacc) if preinfections else filter_unexperienced(vacc)

    return {
        'selected_df': vacc,
        'reference_df': reference,
        'selected_preds': predictions[vacc.index] if predictions is not None else None,
        'reference_preds': predictions[reference.index] if predictions is not None else None
    }
