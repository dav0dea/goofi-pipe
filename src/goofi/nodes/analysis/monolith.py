import tempfile

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, StringParam


class Monolith(Node):

    def config_params():
        return {
            "monolith": {
                "ignore_features": StringParam("pyspi,toto", doc="Comma-separated list of features to ignore"),
                "clip_value": FloatParam(75, 1, 200, doc="Clip values to this range during preprocessing"),
            }
        }

    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"features": DataType.ARRAY}

    def setup(self):
        pass

    def process(self, data: Data):
        assert "sfreq" in data.meta, "Data must have a 'sfreq' (sampling frequency) in its metadata."
        sfreq = data.meta["sfreq"]
        data_arr = data.data

        ignore_names = set(map(str.strip, self.params.monolith.ignore_features.value.split(",")))

        # Preprocess the data
        data_arr = preprocess(data_arr, sfreq, clip_value=self.params.monolith.clip_value.value)

        features = []
        # compute channel-wise features
        for feat_name, feat_fn in CHANNELWISE_FEAT_FNS.items():
            if feat_name in ignore_names:
                continue

            for channel_data in data_arr:
                ft = feat_fn(channel_data, sfreq)
                if isinstance(ft, np.ndarray) or isinstance(ft, list):
                    features.append(ft)
                else:
                    features.append([ft])

        # compute non channel-wise features
        for feat_name, feat_fn in NON_CHANNEL_WISE_FEAT_FNS.items():
            if feat_name in ignore_names:
                continue

            ft = feat_fn(data_arr, sfreq)
            if isinstance(ft, np.ndarray) or isinstance(ft, list):
                features.append(ft)
            else:
                features.append([ft])

        features = np.concatenate(features, axis=-1)
        meta = data.meta.copy()
        del meta["channels"]

        return {"features": (features, meta)}


def _bandpass_filter(x, sfreq, l_freq, h_freq, order=5):
    from scipy.signal import butter, sosfiltfilt

    nyq = 0.5 * sfreq
    low = l_freq / nyq
    high = h_freq / nyq
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def _notch_filter(x, sfreq, freqs=[50, 60], Q=30):
    from scipy.signal import iirnotch, sosfiltfilt

    nyq = sfreq / 2
    sos_list = []
    for base_freq in freqs:
        n = 1
        while True:
            f = base_freq * n
            if f >= nyq:
                break
            # iirnotch returns (b, a); convert to SOS
            b, a = iirnotch(f, Q, sfreq)
            from scipy.signal import tf2sos

            sos = tf2sos(b, a)
            sos_list.append(sos)
            n += 1
    # Concatenate all SOS
    sos_all = np.concatenate(sos_list, axis=0)
    return sosfiltfilt(sos_all, x, axis=-1)


def preprocess(data: np.ndarray, sfreq: float, clip_value: float = 75):
    # bandpass filter the data
    data = _bandpass_filter(data, sfreq, l_freq=3, h_freq=30)
    # apply notch filters at 50Hz and 60Hz
    data = _notch_filter(data, sfreq, freqs=[50, 60])
    # remove DC offset
    data = data - np.mean(data, axis=-1, keepdims=True)
    # clip values
    data = np.clip(data, -clip_value, clip_value)
    # standardize
    data = (data - np.mean(data)) / np.std(data)
    return data


def skewness(x, sfreq):
    from scipy.stats import skew

    return skew(x)


def kurt(x, sfreq):
    from scipy.stats import kurtosis

    return kurtosis(x)


def hjorth_activity(x, sfreq):
    return np.var(x)


def hjorth_mobility(x, sfreq):
    return np.sqrt(np.var(np.diff(x)) / np.var(x))


def hjorth_complexity(x, sfreq):
    diff1 = np.diff(x)
    diff2 = np.diff(diff1)
    return (np.sqrt(np.var(diff2) / np.var(diff1))) / (np.sqrt(np.var(diff1) / np.var(x)))


def spectral(x, sfreq):
    from scipy.signal import welch

    f, Pxx = welch(x, sfreq)
    mean_frequency = np.sum(f * Pxx) / np.sum(Pxx)
    delta = np.trapz(Pxx[(f >= 0) & (f <= 4)], f[(f >= 0) & (f <= 4)])
    theta = np.trapz(Pxx[(f > 4) & (f <= 8)], f[(f > 4) & (f <= 8)])
    alpha = np.trapz(Pxx[(f > 8) & (f <= 12)], f[(f > 8) & (f <= 12)])
    beta = np.trapz(Pxx[(f > 12) & (f <= 30)], f[(f > 12) & (f <= 30)])
    gamma = np.trapz(Pxx[(f > 30) & (f <= 45)], f[(f > 30) & (f <= 45)])

    from fooof import FOOOF

    model = FOOOF(peak_width_limits=(2, 12))
    model.fit(f, Pxx, freq_range=(3, 30))

    return np.array([delta, theta, alpha, beta, gamma, mean_frequency] + list(model.aperiodic_params_))


def compute_detrended_fluctuation(x, sfreq):
    from antropy import detrended_fluctuation

    return detrended_fluctuation(x)


def compute_higuchi_fd(x, sfreq):
    from antropy import higuchi_fd

    return higuchi_fd(x)


def compute_lziv_complexity(x, sfreq):
    from antropy import lziv_complexity

    return lziv_complexity(x > x.mean(), normalize=True)


def compute_petrosian_fd(x, sfreq):
    from antropy import petrosian_fd

    return petrosian_fd(x)


def binarize_by_mean(x):
    return (x > np.mean(x)).astype(int)


def entropy_shannon(x, sfreq):
    import neurokit2 as nk2

    bin_ts = binarize_by_mean(x)
    return nk2.entropy_shannon(bin_ts, base=2)[0]


def entropy_renyi(x, sfreq):
    import neurokit2 as nk2

    bin_ts = binarize_by_mean(x)
    return nk2.entropy_renyi(bin_ts, alpha=2)[0]


def entropy_approximate(x, sfreq):
    import neurokit2 as nk2

    return nk2.entropy_approximate(x, delay=1, dimension=2, tolerance="sd", Corrected=True)[0]


def entropy_sample(x, sfreq):
    import neurokit2 as nk2

    return nk2.entropy_sample(x, delay=1, dimension=2, tolerance="sd")[0]


def entropy_rate(x, sfreq):
    import neurokit2 as nk2

    return nk2.entropy_rate(x, kmax=10, symbolize="mean")[0]


def entropy_permutation(x, sfreq):
    import neurokit2 as nk2

    return nk2.entropy_permutation(x, delay=1, dimension=2, corrected=True, weighted=False, conditional=False)[0]


def entropy_permutation_weighted(x, sfreq):
    import neurokit2 as nk2

    return nk2.entropy_permutation(x, delay=1, dimension=2, corrected=True, weighted=True, conditional=False)[0]


def entropy_permutation_conditional(x, sfreq):
    import neurokit2 as nk2

    return nk2.entropy_permutation(x, delay=1, dimension=2, corrected=True, weighted=False, conditional=True)[0]


def entropy_permutation_weighted_conditional(x, sfreq):
    import neurokit2 as nk2

    return nk2.entropy_permutation(x, delay=1, dimension=2, corrected=True, weighted=True, conditional=True)[0]


def entropy_multiscale(x, sfreq):
    import neurokit2 as nk2

    return nk2.entropy_multiscale(x, dimension=2, tolerance="sd", method="MSPEn")[0]


def entropy_bubble(x, sfreq):
    import neurokit2 as nk2

    return nk2.entropy_bubble(x, delay=1, dimension=2, alpha=2, tolerance="sd")[0]


def entropy_svd(x, sfreq):
    import neurokit2 as nk2

    return nk2.entropy_svd(x, delay=1, dimension=2)[0]


def entropy_attention(x, sfreq):
    import neurokit2 as nk2

    return nk2.entropy_attention(x)[0]


def entropy_dispersion(x, sfreq):
    import neurokit2 as nk2

    return nk2.entropy_dispersion(x, delay=1, dimension=2, c=6, symbolize="NCDF")[0]


def compute_spi_features(data, subset="fast"):
    from pyspi.calculator import Calculator

    # Create a temporary file with the config
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(PYSPI_FAST_CONFIG)
        pyspi_fast_config = f.name

    calc = Calculator(data, subset=subset, configfile=pyspi_fast_config)
    calc.compute()
    s = calc.table.stack(list(range(calc.table.columns.nlevels))).dropna()
    print(s, s.values)
    return s.values


def phiid_pairwise_metrics(x, sfreq, tau=5, kind="gaussian", redundancy="MMI"):
    """
    Compute all PhiID atom metrics for all pairs of channels (asymmetric, src->tgt and tgt->src).
    Returns a flattened array of all atom means for each channel pair and direction.
    """

    try:
        from phyid.calculate import calc_PhiID
    except ImportError:
        raise ImportError(
            "The phyid package is not installed. Please install it with the following command:\n"
            "pip install git+https://github.com/Imperial-MIND-lab/integrated-info-decomp.git"
        )

    data = np.atleast_2d(x)
    n_channels, n_time = data.shape

    atom_names = [
        "rtr",
        "rtx",
        "rty",
        "rts",
        "xtr",
        "xtx",
        "xty",
        "xts",
        "ytr",
        "ytx",
        "yty",
        "yts",
        "str",
        "stx",
        "sty",
        "sts",
    ]
    # For each ordered pair (i, j), i != j
    results = []
    for i in range(n_channels):
        for j in range(n_channels):
            if i == j:
                continue
            src = data[i]
            trg = data[j]
            atoms_res, _ = calc_PhiID(src, trg, tau, kind=kind, redundancy=redundancy)
            atom_means = [float(np.mean(atoms_res[name])) for name in atom_names]
            results.append(atom_means)
    return np.array(results, dtype=np.float32).flatten()


def compute_toto_embedding(x, sfreq):
    from toto.inference.embedding import embed as embed_toto

    return embed_toto(x, global_average=True)


CHANNELWISE_FEAT_FNS = {
    "detrended_fluctuation": compute_detrended_fluctuation,
    "higuchi_fd": compute_higuchi_fd,
    "lziv_complexity": compute_lziv_complexity,
    "petrosian_fd": compute_petrosian_fd,
    "skewness": skewness,
    "kurtosis": kurt,
    "hjorth_activity": hjorth_activity,
    "hjorth_mobility": hjorth_mobility,
    "hjorth_complexity": hjorth_complexity,
    "spectral": spectral,
    "entropy_shannon": entropy_shannon,
    "entropy_renyi": entropy_renyi,
    "entropy_rate": entropy_rate,
    "entropy_bubble": entropy_bubble,
    "entropy_svd": entropy_svd,
    "entropy_attention": entropy_attention,
    "entropy_dispersion": entropy_dispersion,
}

NON_CHANNEL_WISE_FEAT_FNS = {
    "pyspi": compute_spi_features,
    "phiid": phiid_pairwise_metrics,
    "toto": compute_toto_embedding,
}

BATCHABLE_FEATS = ["toto"]

PYSPI_FAST_CONFIG = """
.statistics.basic:
  Covariance:
    labels:
    - undirected
    - linear
    - signed
    - multivariate
    - contemporaneous
    dependencies: null
    configs:
    - estimator: ShrunkCovariance
      squared: true
  Precision:
    labels:
    - undirected
    - linear
    - signed
    - multivariate
    - contemporaneous
    dependencies: null
    configs:
    - estimator: ShrunkCovariance
      squared: true
  SpearmanR:
    labels:
    - undirected
    - nonlinear
    - signed
    - bivariate
    - contemporaneous
    dependencies: null
    configs:
    - squared: false
  KendallTau:
    labels:
    - undirected
    - nonlinear
    - signed
    - bivariate
    - contemporaneous
    dependencies: null
    configs:
    - squared: false
  CrossCorrelation:
    labels:
    - undirected
    - linear
    - signed/unsigned
    - bivariate
    - time-dependent
    dependencies: null
    configs:
    - statistic: mean
      squared: true
      sigonly: false
.statistics.distance:
  PairwiseDistance:
    labels:
    - unsigned
    - unordered
    - nonlinear
    - undirected
    dependencies: null
    configs:
    - metric: braycurtis
  DynamicTimeWarping:
    labels:
    - undirected
    - nonlinear
    - unsigned
    - bivariate
    - time-dependent
    dependencies: null
    configs:
    - global_constraint: sakoe_chiba
  LongestCommonSubsequence:
    labels:
    - undirected
    - nonlinear
    - unsigned
    - bivariate
    - time-dependent
    dependencies: null
    configs:
    - global_constraint: sakoe_chiba
.statistics.infotheory:
  JointEntropy:
    labels:
    - undirected
    - nonlinear
    - unsigned
    - bivariate
    - contemporaneous
    dependencies:
    - java
    configs:
    - estimator: kernel
  ConditionalEntropy:
    labels:
    - undirected
    - nonlinear
    - unsigned
    - bivariate
    - contemporaneous
    dependencies:
    - java
    configs:
    - estimator: kernel
  CrossmapEntropy:
    labels:
    - unsigned
    - directed
    - time-dependent
    - bivariate
    dependencies:
    - java
    configs:
    - estimator: gaussian
      history_length: 10
  StochasticInteraction:
    labels:
    - undirected
    - nonlinear
    - unsigned
    - bivariate
    - time-dependent
    dependencies:
    - java
    configs:
    - estimator: gaussian
  MutualInfo:
    labels:
    - undirected
    - nonlinear
    - unsigned
    - bivariate
    - contemporaneous
    dependencies:
    - java
    configs:
    - estimator: gaussian
  TimeLaggedMutualInfo:
    labels:
    - directed
    - nonlinear
    - unsigned
    - bivariate
    - time-dependent
    dependencies:
    - java
    configs:
    - estimator: gaussian
.statistics.spectral:
  CoherencePhase:
    labels:
    - undirected
    - linear
    - unsigned
    - bivariate
    - frequency-dependent
    dependencies: null
    configs:
    - fmin: 0.25
      fmax: 0.5
      statistic: max
  CoherenceMagnitude:
    labels:
    - undirected
    - linear
    - unsigned
    - bivariate
    - frequency-dependent
    dependencies: null
    configs:
    - fmin: 0.25
      fmax: 0.5
      statistic: max
  ImaginaryCoherence:
    labels:
    - undirected
    - linear
    - unsigned
    - bivariate
    - frequency-dependent
    dependencies: null
    configs:
    - fmin: 0.25
      fmax: 0.5
      statistic: max
  PhaseSlopeIndex:
    labels:
    - directed
    - linear/nonlinear
    - unsigned
    - bivariate
    - frequency-dependent
    dependencies: null
    configs:
    - fmin: 0.25
      fmax: 0.5
  PhaseLockingValue:
    labels:
    - undirected
    - linear
    - unsigned
    - bivariate
    - frequency-dependent
    dependencies: null
    configs:
    - fmin: 0.25
      fmax: 0.5
      statistic: max
  PhaseLagIndex:
    labels:
    - undirected
    - linear
    - unsigned
    - bivariate
    - frequency-dependent
    dependencies: null
    configs:
    - fmin: 0.25
      fmax: 0.5
      statistic: max
  WeightedPhaseLagIndex:
    labels:
    - undirected
    - linear
    - unsigned
    - bivariate
    - frequency-dependent
    dependencies: null
    configs:
    - fmin: 0.25
      fmax: 0.5
      statistic: max
  DebiasedSquaredPhaseLagIndex:
    labels:
    - undirected
    - linear
    - unsigned
    - bivariate
    - frequency-dependent
    dependencies: null
    configs:
    - fmin: 0.25
      fmax: 0.5
      statistic: max
  DebiasedSquaredWeightedPhaseLagIndex:
    labels:
    - undirected
    - linear
    - unsigned
    - bivariate
    - frequency-dependent
    dependencies: null
    configs:
    - fmin: 0.25
      fmax: 0.5
      statistic: max
  PairwisePhaseConsistency:
    labels:
    - undirected
    - linear
    - unsigned
    - bivariate
    - frequency-dependent
    dependencies: null
    configs:
    - fmin: 0.25
      fmax: 0.5
      statistic: max
  GroupDelay:
    labels:
    - directed
    - linear
    - unsigned
    - bivariate
    - frequency-dependent
    dependencies: null
    configs:
    - fmin: 0.25
      fmax: 0.5
      statistic: delay
  SpectralGrangerCausality:
    labels:
    - directed
    - linear
    - unsigned
    - bivariate
    - frequency-dependent
    dependencies: null
    configs:
    - fmin: 0.25
      fmax: 0.5
      order: 20
      method: parametric
      statistic: max
"""
