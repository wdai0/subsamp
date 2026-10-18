"""Permanent ten-signal scenarios; no production fitting code lives here."""

import numpy as np

BETA = np.r_[.1, .5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 5., np.zeros(90)]


def covariance(name):
    sigma = np.eye(100)
    if name == "block11":
        sigma[:11, :11] = .5 ** np.abs(np.arange(11)[:, None] - np.arange(11))
    elif name == "full90":
        sigma = .9 ** np.abs(np.arange(100)[:, None] - np.arange(100))
    elif name != "independent":
        raise ValueError(f"Unknown covariance: {name}")
    return sigma


def scenarios():
    result = [dict(name="paper-independent", covariance="independent", noise_sd=1.),
              dict(name="paper-block11-low", covariance="block11", noise_sd=1.),
              dict(name="paper-block11-high", covariance="block11", noise_sd=3.)]
    for cov in ("independent", "block11", "full90"):
        for snr in (1, 5, 20):
            signal_variance = float(BETA @ covariance(cov) @ BETA)
            result.append(dict(name=f"stress-{cov}-snr{snr}", covariance=cov,
                               noise_sd=float(np.sqrt(signal_variance / snr)), snr=snr))
    return result


def dataset(scenario, seed):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(80, 100)) @ np.linalg.cholesky(covariance(scenario["covariance"])).T
    y = X @ BETA + scenario["noise_sd"] * rng.normal(size=80)
    return X, y


def panel():
    cases = []
    for i, scenario in enumerate(scenarios()):
        for replicate in range(2):
            cases.append(dict(scenario, id=f"{scenario['name']}-{replicate}",
                              data_seed=91000 + 2 * i + replicate,
                              schedule_seed=92000 + 2 * i + replicate, s=10, q=10, m=120))
    cases.append(dict(scenarios()[0], id="paper-independent-m5000", data_seed=91999,
                      schedule_seed=92999, s=10, q=10, m=5000))
    return cases


def selection_metrics(semifinalists, finalists):
    semi, final = set(semifinalists), set(finalists)
    support = set(range(10))
    fp = len(final - support)
    return dict(signal_selected=[int(j in final) for j in range(10)],
                signal_recovery=len(final & support) / 10,
                semifinalist_recall=len(semi & support) / 10,
                false_positives=fp, false_discovery_proportion=fp / max(len(final), 1),
                any_false_positive=int(fp > 0), selected_size=len(final),
                exact_support=int(final == support))


def summarize(records):
    """Failures remain in the denominator report; never pretend they are empty selections."""
    successful = [row for row in records if row.get("error") is None]
    output = dict(datasets=len(records), successes=len(successful), failures=len(records) - len(successful))
    if len(successful) != len(records) or not records:
        output["selection_estimates"] = None
        return output
    output["selection_estimates"] = dict(
        per_signal_power=np.mean([r["signal_selected"] for r in records], axis=0).tolist(),
        average_signal_recovery=float(np.mean([r["signal_recovery"] for r in records])),
        average_signal_count=float(10 * np.mean([r["signal_recovery"] for r in records])),
        semifinalist_recall=float(np.mean([r["semifinalist_recall"] for r in records])),
        mean_false_positives=float(np.mean([r["false_positives"] for r in records])),
        fdr=float(np.mean([r["false_discovery_proportion"] for r in records])),
        fwer=float(np.mean([r["any_false_positive"] for r in records])),
        mean_selected_size=float(np.mean([r["selected_size"] for r in records])),
        exact_support_frequency=float(np.mean([r["exact_support"] for r in records])))
    return output
