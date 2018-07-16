import numpy as np

def to_points(datafile):
    shots = datafile.shots
    points = datafile.data.shape[0] // (4 * shots)
    return np.transpose(np.reshape(datafile.data, (points, shots, 4)), [0, 2, 1])

def single_spectrum_probabilities(datafile, cool_threshold, count_threshold):
    points = to_points(datafile)
    out = []
    for cool, cool_err, counts, counts_err in points:
        mask = ((cool < cool_threshold) + cool_err + counts_err) < 0.5
        allowed_counts = counts[mask]
        dark_counts = sum(map(lambda x: x <= count_threshold, allowed_counts))
        out.append(dark_counts / allowed_counts.shape[0])
    return np.array(out)[1:]

def continuous_single_pass(datafile, cool_threshold, count_threshold):
    probs = single_spectrum_probabilities(datafile, cool_threshold,
                                          count_threshold)
    start_frequency = datafile.aom_start - datafile.carrier
    frequencies = np.array([start_frequency + (i + 1) * datafile.step_size\
                            for i in range(probs.shape[0])])
    errors = np.max([[0.01] * probs.shape[0], np.sqrt((probs * (1 - probs)) /
        datafile.shots)], axis=0)
    return np.transpose([frequencies, probs, errors])
