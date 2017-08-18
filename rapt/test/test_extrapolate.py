import numpy
from rapt import raptparams, nccfparams, Rapt


def _extrapolate_lag_val(lag_results, min_valid_correlation,
                         max_allowed_candidates, params):
    extrapolated_cands = []

    if len(lag_results[0]) == 0:
        return extrapolated_cands
    elif len(lag_results[0]) == 1:
        current_lag = 0 + params[1].shortest_lag_per_frame
        new_lag = int(round(current_lag * params[0].sample_rate_ratio))
        extrapolated_cands.append((new_lag, lag_results[0][0]))
        return extrapolated_cands

    least_lag = params[0].sample_rate_ratio * params[1].shortest_lag_per_frame
    most_lag = params[0].sample_rate_ratio * params[1].longest_lag_per_frame
    for k, k_val in enumerate(lag_results[0]):
        print('function extrapolated_cands = ',extrapolated_cands)
        if k_val > min_valid_correlation:
            current_lag = k + params[1].shortest_lag_per_frame
            new_lag = int(round(current_lag * params[0].sample_rate_ratio))

            if k == 0:
                # if at 1st lag value, interpolate using 0,0 input on left
                prev_lag = k - 1 + params[1].shortest_lag_per_frame
                new_prev = int(round(prev_lag * params[0].sample_rate_ratio))
                next_lag = (k + 1 + params[1].shortest_lag_per_frame)
                new_next = int(round(next_lag * params[0].sample_rate_ratio))
                lags = numpy.array([new_prev, new_lag, new_next])
                vals = numpy.array([0.0, k_val, lag_results[0][k + 1]])
                para = numpy.polyfit(lags, vals, 2)
                final_lag = int(round(-para[1] / (2 * para[0])))
                final_corr = float(para[0] * final_lag**2 + para[1] *
                                   final_lag + para[2])
                if (final_lag < least_lag or final_lag > most_lag or
                        final_corr < -1.0 or final_corr > 1.0):
                    current_lag = k + params[1].shortest_lag_per_frame
                    new_lag = int(round(current_lag *
                                        params[0].sample_rate_ratio))
                    extrapolated_cands.append((new_lag, k_val))
                else:
                    extrapolated_cands.append((final_lag, final_corr))
            elif k == len(lag_results[0]) - 1:
                # if at last lag value, interpolate using 0,0 input on right
                next_lag = k + 1 + params[1].shortest_lag_per_frame
                new_next = int(round(next_lag * params[0].sample_rate_ratio))
                prev_lag = (k - 1 + params[1].shortest_lag_per_frame)
                new_prev = int(round(prev_lag * params[0].sample_rate_ratio))
                lags = numpy.array([new_prev, new_lag, new_next])
                vals = numpy.array([lag_results[0][k - 1], k_val, 0.0])
                para = numpy.polyfit(lags, vals, 2)
                final_lag = int(round(-para[1] / (2 * para[0])))
                final_corr = float(para[0] * final_lag**2 + para[1] *
                                   final_lag + para[2])
                if (final_lag < least_lag or final_lag > most_lag or
                        final_corr < -1.0 or final_corr > 1.0):
                    current_lag = k + params[1].shortest_lag_per_frame
                    new_lag = int(round(current_lag *
                                        params[0].sample_rate_ratio))
                    extrapolated_cands.append((new_lag, k_val))
                else:
                    extrapolated_cands.append((final_lag, final_corr))
            else:
                # we are in middle of lag results - use left and right
                next_lag = (k + 1 + params[1].shortest_lag_per_frame)
                new_next = int(round(next_lag * params[0].sample_rate_ratio))
                prev_lag = (k - 1 + params[1].shortest_lag_per_frame)
                new_prev = int(round(prev_lag * params[0].sample_rate_ratio))
                lags = numpy.array([new_prev, new_lag, new_next])
                vals = numpy.array([lag_results[0][k - 1], k_val,
                                    lag_results[0][k + 1]])
                para = numpy.polyfit(lags, vals, 2)
                final_lag = int(round(-para[1] / (2 * para[0])))
                final_corr = float(para[0] * final_lag**2 + para[1] *
                                   final_lag + para[2])
                if (final_lag < least_lag or final_lag > most_lag or
                        final_corr < -1.0 or final_corr > 1.0):
                    current_lag = k + params[1].shortest_lag_per_frame
                    new_lag = int(round(current_lag *
                                        params[0].sample_rate_ratio))
                    extrapolated_cands.append((new_lag, k_val))
                else:
                    extrapolated_cands.append((final_lag, final_corr))

    return extrapolated_cands


params = raptparams.Raptparams()
nccf = nccfparams.Nccfparams()
is_first_pass = False
fs = 2000

# value 'n' in NCCF equation
nccf.samples_correlated_per_lag = int(round(params.correlation_window_size * fs))

# start value of k in NCCF equation
if is_first_pass:
    nccf.shortest_lag_per_frame = int(round(fs / params.maximum_allowed_freq))
else:
    nccf.shortest_lag_per_frame = 0

# value 'K' in NCCF equation
nccf.longest_lag_per_frame = int(round(fs / params.minimum_allowed_freq))

# value z in NCCF equation
nccf.samples_per_frame = int(round(params.frame_step_size * fs))

# value of M-1 in NCCF equation
nccf.max_frame_count = int(round(float(5875) / float(nccf.samples_per_frame)) - 1)


print('shortest_lag_per_frame = ', nccf.shortest_lag_per_frame)
print('longest_lag_per_frame = ', nccf.longest_lag_per_frame)
print('samples_per_frame = ', nccf.samples_per_frame)
print('max_frame_count = ', nccf.max_frame_count)


lag_results = [[0.660751900728446,	0.535225567807628,	0.410126524131404,	0.306364531335129,	0.211637006748392,	0.137397952251725,	0.0807624320788661,	0.0460625778729787,	0.00336881809650662,	-0.0398273383215233,	-0.103491155412674,	-0.162383869543343,	-0.227073394474005,	-0.265610673016430,	-0.301774666744059,	-0.309584995154267,	-0.355499348177031,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0], 0.660751900728446]
print(lag_results)
min_valid_correlation = (lag_results[1] * params.min_acceptable_peak_val)
print(min_valid_correlation)

params.sample_rate_ratio = 8.0
param= (params, nccf)

print (_extrapolate_lag_val(lag_results, min_valid_correlation, 0, param))



rap = rapt.Rapt()
rap.nccfparams = nccf
rap.params = params
print(rap._extrapolate_lag_val(lag_results, min_valid_correlation))
