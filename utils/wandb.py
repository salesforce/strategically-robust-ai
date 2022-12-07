import numpy as np
from .remote import remote_env_fun
from .profiling import get_speed_stats

def collect_summary(trainer):
    summary = {}

    summary_dict = remote_env_fun(trainer, lambda e: e.summary)

    for env_summary in summary_dict.values():
        for k, v in env_summary.items():
            if k not in summary:
                summary[k] = []
            summary[k].append(v)

    for k, v in summary.items():
        mean = np.nanmean(np.array(v))
        if np.isnan(mean):
            mean = 0.0
        summary[k] = mean

    return summary


RESULT_BLACKLIST = [
    "custom_metrics",
    "date",
    "done",
    "experiment_id",
    "hostname",
    "node_ip",
    "off_policy_estimator",
    "pid",
    "config",
]


def flatten_result(result):
    def flatten_nested_vals(key_root, key, val, flat_dict):
        if key in RESULT_BLACKLIST:
            return
        this_key = key_root + key
        if isinstance(val, (int, float, np.floating)):
            flat_dict[this_key] = val
        elif isinstance(val, list):
            flat_dict[this_key] = np.array(val)
        elif isinstance(val, np.ndarray):
            for i in range(len(val)):
                flat_dict[this_key + "/" + str(i)] = val[i]
        elif not isinstance(val, dict):
            raise TypeError(
                "Value of {} must be an int, float, or dict. Was {}.".format(
                    this_key + "/" + key, type(val)
                )
            )
        else:
            for sub_key, sub_val in val.items():
                flatten_nested_vals(this_key + "/", sub_key, sub_val, flat_dict)

    root = ""
    flat_result = {}
    for k, v in result.items():
        flatten_nested_vals(root, k, v, flat_result)

    for k, v in flat_result.items():
        if "/" not in k:
            del flat_result[k]
            flat_result["trainer/" + k] = v

    return flat_result


def make_wandb_log(
    trainer,
    result,
    whitelist=None,
    models_that_trained_int=None,
    suffix="",
    print_all=False,
    print_common_fields=False,
):

    flat_result = flatten_result(result)

    tag = suffix + "/" if suffix != "" else ""

    if print_all:
        # just show everything
        for skey, sval in collect_summary(trainer).items():
            flat_result["episode/" + skey] = sval
    else:
        # filter keys to report
        # necessary to avoid double reporting when using two trainers
        if suffix == "agent":
            short_key = "/a/"
            long_key = "/agent/"
            short_suffix = "/a"
        elif suffix == "planner":
            short_key = "/p/"
            long_key = "/planner/"
            short_suffix = "/p"
        else:
            raise NotImplementedError

        for skey, sval in collect_summary(trainer).items():

            if whitelist is not None:
                assert isinstance(whitelist, list)
                if skey not in whitelist:
                    continue

            if short_key != "" and long_key != "":
                if long_key in skey or short_key in skey or skey[-2:] == short_suffix:
                    flat_result["episode/" + skey] = sval
                elif print_common_fields:
                    flat_result["episode/" + skey] = sval
                else:
                    print("[W&B] Skipping", skey, sval)
            else:
                flat_result["episode/" + skey] = sval

    # speed stats
    (
        sps_total,
        sps_iter,
        sps_trained,
        sps_sampled,
        tt1m_total,
        tt1m_trained,
    ) = get_speed_stats(result)

    flat_result["perf/{}steps_per_s/total".format(tag)] = sps_total
    flat_result["perf/{}steps_per_s/iter".format(tag)] = sps_iter
    flat_result["perf/{}steps_per_s/trained".format(tag)] = sps_trained
    flat_result["perf/{}steps_per_s/sampled".format(tag)] = sps_sampled
    flat_result["perf/{}hours_to_1m/total".format(tag)] = tt1m_total
    flat_result["perf/{}hours_to_1m/trained".format(tag)] = tt1m_trained

    if models_that_trained_int is not None:
        flat_result["perf/models_that_trained_int"] = models_that_trained_int

    return flat_result
