import numpy as np
from my_util import *
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats
import time
import pandas as pd
import math


def parse_tle_decimal(s):
    """Parse a floating point with implicit leading dot.
    >>> parse_tle_decimal('378')
    0.378
    """
    return float('.' + s)


def parse_tle_float(s):
    """Parse a floating point with implicit dot and exponential notation.
    >>> parse_tle_float(' 12345-3')
    0.00012345
    >>> parse_tle_float('+12345-3')
    0.00012345
    >>> parse_tle_float('-12345-3')
    -0.00012345
    """
    return float(s[0] + '.' + s[1:6] + 'e' + s[6:8])


def write_tle_float(number):
    if number == 0.0:
        return ' 00000-0'
    a, b = '{:.4E}'.format(number).split('E')
    ret = f"{float(a)/10:.5f}"
    exp_str = f"{int(b)+1:+02d}"
    if ret.startswith("0."):
        return " " + ret[2:] + exp_str
    if ret.startswith("-0."):
        return "-" + ret[3:] + exp_str
    print('oh no')


# https://github.com/FedericoStra/tletools/blob/master/tletools/tle.py
def from_lines(line1, line2):
    norad = line1[2:7]
    # classification = line1[7]
    int_desig = line1[9:17]
    epoch_year = line1[18:20]
    epoch_day = float(line1[20:32])
    dn_o2 = float(line1[33:43])
    ddn_o6 = parse_tle_float(line1[44:52])
    bstar = parse_tle_float(line1[53:61])
    # set_num = line1[64:68]
    inc = float(line2[8:16])
    raan = float(line2[17:25])
    ecc = parse_tle_decimal(line2[26:33])
    argp = float(line2[34:42])
    M = float(line2[43:51])  # mean anomaly
    n = float(line2[52:63])  # mean motion
    rev_num = int(line2[63:68])
    return (norad, int_desig, epoch_year, epoch_day, dn_o2, bstar, inc, raan, ecc, argp, M, n, rev_num)


def _conv_year(s):
    """Interpret a two-digit year string."""
    if isinstance(s, int):
        return s
    y = int(s)
    return y + (1900 if y >= 57 else 2000)


def make_batch_tle_arrays(entire_file, return_pandas=False):
    """
    entire_file: a list of tuples; each tuple is the two lines of a TLE: (line1, line2)
    """
    n_tles = len(entire_file)
    epoch_str = np.empty((n_tles,), dtype='U14')
    epoch_ds50 = np.zeros(n_tles)
    dn_o2 = np.zeros(n_tles)
    ddn_o6 = np.zeros((n_tles,), dtype=np.float32)
    bstar = np.zeros((n_tles,), dtype=np.float32)
    inc = np.zeros((n_tles,), dtype=np.float32)
    raan = np.zeros((n_tles,), dtype=np.float32)
    ecc = np.zeros((n_tles,), dtype=np.float32)
    argp = np.zeros((n_tles,), dtype=np.float32)
    M = np.zeros((n_tles,), dtype=np.float32)
    n = np.zeros(n_tles)
    rev_num = np.zeros((n_tles,), dtype=np.uintc)
    for idx, (l1, l2) in enumerate(entire_file):
        epoch_str[idx] = l1[18:32]
        year = _conv_year(l1[18:20])
        dt = datetime(year, 1, 1) + timedelta(float(l1[20:32]) - 1)
        epoch_ds50[idx] = days_since_yr(dt, 1950)
        dn_o2[idx] = float(l1[33:43])
        ddn_o6[idx] = parse_tle_float(l1[44:52])
        bstar[idx] = parse_tle_float(l1[53:61])
        inc[idx] = float(l2[8:16])
        raan[idx] = float(l2[17:25])
        ecc[idx] = parse_tle_decimal(l2[26:33])
        argp[idx] = float(l2[34:42])
        M[idx] = float(l2[43:51])  # mean anomaly
        n[idx] = float(l2[52:63])  # mean motion
        rev_num[idx] = int(l2[63:68])
    if return_pandas:
        return pd.DataFrame(data={
            'epoch_str': epoch_str, 'epoch_ds50': epoch_ds50,
            'dn_o2': dn_o2, 'ddn_o6': ddn_o6, 'bstar': bstar,
            'inc': inc, 'raan': raan, 'ecc': ecc, 'argp': argp, 'M': M, 'n': n,
            'rev_num': rev_num
        })
    else:
        return epoch_str, epoch_ds50, dn_o2, ddn_o6, bstar, inc, raan, ecc, argp, M, n, rev_num


def save_tle_npz(fname, df2, section_pos, norads, int_desigs):
    np.savez_compressed(fname, section_pos=section_pos, norad=norads,
                        int_desig=int_desigs, epoch_str=df2['epoch_str'].to_numpy().astype('U'),
                        epoch_ds50=df2['epoch_ds50'],
                        dn_o2=df2['dn_o2'], ddn_o6=df2['ddn_o6'],
                        bstar=df2['bstar'], inc=df2['inc'], raan=df2['raan'], ecc=df2['ecc'], argp=df2['argp'],
                        mean_anom=df2['M'], mean_motion=df2['n'], rev_num=df2['rev_num'])


def tle_npz_to_df(data):
    return pd.DataFrame(data={
            'epoch_str': data['epoch_str'], 'epoch_ds50': data['epoch_ds50'],
            'dn_o2': data['dn_o2'], 'ddn_o6': data['ddn_o6'], 'bstar': data['bstar'],
            'inc': data['inc'], 'raan': data['raan'], 'ecc': data['ecc'], 'argp': data['argp'],
            'M': data['mean_anom'], 'n': data['mean_motion'],
            'rev_num': data['rev_num']
        })


def import_batch_tle_v2(fname):
    with open(fname, 'r') as fp:
        line = fp.readline()
        tle_no = 0
        prev_id = ''
        section_pos = []
        norad_ids = []
        int_desigs = []
        entire_file = []
        while line:
            assert len(line) > 10
            line_type = line[0]
            if line_type == '1':
                this_id = line[2:7]
                if this_id != prev_id:
                    # print(this_id, line_no)
                    norad_ids.append(this_id)
                    int_desigs.append(line[9:17])
                    section_pos.append(tle_no)
                    prev_id = this_id

            prev_line = line
            line = fp.readline()
            if line:
                if line[0] == '1':
                    tle_no += 1
                elif line[0] == '2':
                    entire_file.append((prev_line.rstrip('\n'), line.rstrip('\n')))
        section_pos.append(len(entire_file))
        section_pos = np.array(section_pos, dtype=np.int)
        norads = np.array(norad_ids, dtype='U5')
        int_desigs = np.array(int_desigs, dtype='U8')
        return entire_file, section_pos, norads, int_desigs


def unshuffle_launch_4(entire_file, section_pos, norads):
    switch_epoch = 20070.1  # 70.1 days after start of 2020
    before_to_after = {45187:45188, 45224:45232, 45196:45190, 45231:45223, 45219:45226, 45235:45228, 45202:45231, 45223:45227, 45232:45207, 45234:45224, 45207:45182, 45182:45203, 45197:45184, 45203:45202, 45193:45195, 45191:45193, 45192:45194, 45184:45185, 45190:45192, 45227:45235, 45185:45186, 45186:45187, 45188:45189, 45226:45234, 45189:45191, 45195:45197, 45194:45196, 45228:45219, 45212:45206, 45206:45212}
    after_to_before = {}
    for key, value in before_to_after.items():
        after_to_before[value] = key

    norad_int = [int(n) for n in norads]
    norad_idx = {}
    for idx, norad in enumerate(norad_int):
        norad_idx[norad] = idx
    new_file = []
    new_section_pos = []
    tle_no = 0
    for k, norad in enumerate(norad_int):
        new_section_pos.append(tle_no)
        if norad in before_to_after:
            pre_norad = after_to_before[norad]
            i1 = section_pos[norad_idx[pre_norad]]
            i2 = section_pos[norad_idx[pre_norad]+1]
            for lidx in range(i1, i2):
                if float(entire_file[lidx][0][18:32]) < switch_epoch:
                    new_file.append(entire_file[lidx])
                    tle_no += 1
            i1 = section_pos[k]
            i2 = section_pos[k+1]
            for lidx in range(i1, i2):
                if float(entire_file[lidx][0][18:32]) > switch_epoch:
                    new_file.append(entire_file[lidx])
                    tle_no += 1
        else:
            i1 = section_pos[k]
            i2 = section_pos[k+1]
            for lidx in range(i1, i2):
                new_file.append(entire_file[lidx])
                tle_no += 1
    new_section_pos.append(len(new_file))
    new_section_pos = np.array(new_section_pos, dtype=np.int)
    return new_file, new_section_pos


def hampel_filter_timeseries(time, series, window_size, n_sigmas=3, min_dev=0.01):
    n = len(series)
    k = 1.4826  # scale factor for Gaussian distribution
    indices = []

    for i in range(n):
        if i < window_size:
            i1 = 0
        else:
            i1 = i - window_size
        if i > n - window_size - 1:
            i2 = n
        else:
            i2 = i + window_size + 1

        if i == 0:
            t_tot = np.sum(time[i1:i2])
            y_tot = np.sum(series[i1:i2])
        else:
            if i1 > i1_old:
                t_tot -= time[i1_old]
                y_tot -= series[i1_old]
            if i2 > i2_old:
                t_tot += time[i2-1]
                y_tot += series[i2-1]
        t_dev = time[i1:i2] - t_tot / (i2 - i1)
        y_dev = series[i1:i2] - y_tot / (i2 - i1)
        slope = np.dot(t_dev, y_dev)/np.dot(t_dev, t_dev)
        deviations = np.abs(y_dev - slope * t_dev)

        # reg = stats.linregress(time[i1:i2], series[i1:i2])
        # series_fit = reg.intercept + reg.slope * time[i1:i2]
        # deviations = np.abs(series[i1:i2] - series_fit)
        S0 = k * np.median(deviations)
        if deviations[i - i1] > max(min_dev, n_sigmas * S0):
            indices.append(i)

        i1_old = i1
        i2_old = i2

    return indices


def bad_indices_from_lines(indices, diffs, use_diff_sign=True, remove_limit=4):
    '''
    If indices i1 and i2 are paired, then i1+1, i1+2, ..., i2 are identified to be filtered out
    '''
    if indices.size == 0:
        return []
    remove_indices = set()
    for k, idx in enumerate(indices):
        pair_idx = -2
        for k2, idx2 in enumerate(indices):
            if k != k2 and abs(idx2 - idx) <= remove_limit and (not use_diff_sign or diffs[k]*diffs[k2] < 0.0):
                pair_idx = idx2
        if pair_idx == -2:
            if idx + 1 <= remove_limit:
                pair_idx = -1
        if pair_idx > -2:
            i1, i2 = min(idx, pair_idx), max(idx, pair_idx)
            for bad_idx in range(i1+1, i2+1):
                remove_indices.add(bad_idx)
    return sorted(remove_indices)


def deg_to_complex(arr):
    return np.exp(1j * arr * math.pi/180.)


def identify_bad_tles(section_pos, epoch_str, epoch_ds50, norads, inc, n, raan, dn_o2,
                inc_std, inc_margin, min_dt, max_dndt, max_dn, reentry_n, max_dradt,
                high_n, filter_dn_o2=True, do_plot=False):
    # bad inclination values
    exist_bad_inc = False
    norad_bad_inc_count = 0
    tle_bad_inc_count = 0
    mask = np.full((len(epoch_str),), True)
    if filter_dn_o2:
        for k in range(len(norads)):
            i1 = section_pos[k]
            i2 = section_pos[k + 1]
            bad_idx = np.nonzero(np.logical_and(n[i1:i2] < high_n, dn_o2[i1:i2] > 0.03))[0]
            mask[bad_idx + i1] = False

    for k in range(len(norads)):
        i1 = section_pos[k]
        i2 = section_pos[k + 1]
        bad_inc_idx = np.nonzero(np.abs(inc[i1:i2] - inc_std) > inc_margin)[0]
        if bad_inc_idx.size > 0:
            mask[bad_inc_idx + i1] = False
        if do_plot and bad_inc_idx.size > 0 or norads[k] == '48601':
            norad_bad_inc_count += 1
            tle_bad_inc_count += bad_inc_idx.size
            if not exist_bad_inc  or norads[k] == '48601':
                exist_bad_inc = True
                plt.figure(0)
            plt.plot(epoch_ds50[i1:i2], inc[i1:i2], '.-', label=norads[k])
            for bad in bad_inc_idx:
                plt.scatter(epoch_ds50[i1+bad], inc[i1+bad], s=80, facecolors='none', edgecolors='r')
                plt.text(epoch_ds50[i1+bad], inc[i1+bad], epoch_str[i1+bad])

    if do_plot and exist_bad_inc:
        plt.title('Inclination, (' + str(norad_bad_inc_count) + '/' + str(len(norads)) + ', '
                  + str(tle_bad_inc_count) + ')')
        plt.legend()

    # bad mean motion (n) values
    exist_bad_n = False
    norad_bad_n_count = 0
    tle_bad_n_count = 0
    tle_bad_n_line_count = 0
    for k in range(len(norads)):
        # print(k)
        i1 = section_pos[k]
        i2 = section_pos[k + 1]

        dn = n[(i1+1):i2] - n[i1:(i2-1)]
        dt = np.maximum(epoch_ds50[(i1+1):i2] - epoch_ds50[i1:(i2-1)], min_dt)
        bad_n_line_idx = np.nonzero(np.logical_and(n[i1:(i2-1)] < reentry_n, np.logical_or(np.abs(dn / dt) > max_dndt,
                                                                                           np.abs(dn) > max_dn)))[0]
        bad_n = dn[bad_n_line_idx]
        if do_plot and bad_n_line_idx.size > 0:
            tle_bad_n_line_count += bad_n_line_idx.size
            if not exist_bad_n:
                exist_bad_n = True
                plt.figure(1)

        bad_n_idx = hampel_filter_timeseries(epoch_ds50[i1:i2], n[i1:i2], 5, n_sigmas=5)
        bad_n_idx_2 = bad_indices_from_lines(bad_n_line_idx, bad_n)
        if len(bad_n_idx) > 0:
            mask[np.array(bad_n_idx) + i1] = False
        if len(bad_n_idx_2) > 0:
            mask[np.array(bad_n_idx_2) + i1] = False
        if do_plot:
            if len(bad_n_idx) > 0 or norads[k] == '48601':
                tle_bad_n_count += len(bad_n_idx)
                if not exist_bad_n:
                    exist_bad_n = True
                    plt.figure(1)

            # if len(bad_n_idx_2) > 0:
            #     print(bad_n_idx_2)
            # if True: #int(norads[k]) in [45205]: # [45219, 45212, 45206, 45235, 45205, 45228, 45227]:
            if bad_n_line_idx.size > 0 or len(bad_n_idx) > 0 or norads[k] == '48601':
                norad_bad_n_count += 1
                plt.plot(epoch_ds50[i1:i2], n[i1:i2], '.-', label=norads[k])
                # for bidx in range(i1, i2):
                #     if 25636 < epoch_ds50[bidx] < 25641:
                #         plt.text(epoch_ds50[bidx], n[bidx], epoch_str[bidx] + '-' + norads[k])
            if bad_n_line_idx.size > 0:
                # print(norads[k], bad_n_line_idx, bad_n)
                for bad in bad_n_line_idx:
                    plt.scatter(epoch_ds50[(i1+bad):(i1+bad+2)], n[(i1+bad):(i1+bad+2)], s=110, facecolors='none', edgecolors='g')
                    plt.plot(epoch_ds50[(i1 + bad):(i1 + bad + 2)], n[(i1 + bad):(i1 + bad + 2)], 'r')
                    plt.text(epoch_ds50[i1 + bad], n[i1 + bad], epoch_str[i1 + bad])
                    plt.text(epoch_ds50[i1 + bad + 1], n[i1 + bad + 1], epoch_str[i1 + bad + 1])

                for bad in bad_n_idx_2:
                    plt.scatter(epoch_ds50[i1+bad], n[i1+bad], s=140, facecolors='none', edgecolors='b')
                    plt.text(epoch_ds50[i1+bad], n[i1+bad], epoch_str[i1+bad])
            if len(bad_n_idx) > 0:
                for bad in bad_n_idx:
                    plt.scatter(epoch_ds50[i1+bad], n[i1+bad], s=60, facecolors='none', edgecolors='r')
                    plt.text(epoch_ds50[i1+bad], n[i1+bad], epoch_str[i1+bad])

    if do_plot:
        if exist_bad_n:
            plt.title('Mean motion, (' + str(norad_bad_n_count) + '/' + str(len(norads)) + ', '
                      + str(tle_bad_n_count) + ', ' + str(tle_bad_n_line_count) + ')')
            plt.legend()

    # bad raan values
    # print('filtering raan values...')
    exist_bad_raan = False
    tle_bad_raan_line_count = 0
    norad_bad_raan_count = 0
    max_ra_rate = max_dradt * math.pi / 180.
    for k in range(len(norads)):
        # print(k)
        i1 = section_pos[k]
        i2 = section_pos[k + 1]

        dra = np.abs(deg_to_complex(raan[(i1 + 1):i2]) - deg_to_complex(raan[i1:(i2 - 1)]))
        dt = np.maximum(epoch_ds50[(i1 + 1):i2] - epoch_ds50[i1:(i2 - 1)], min_dt)
        max_dra = dt * max_ra_rate
        bad_raan_line_idx = np.nonzero(np.logical_and(n[i1:(i2 - 1)] < reentry_n, np.abs(dra) > max_dra))[0]
        bad_raan = dra[bad_raan_line_idx]
        if do_plot and bad_raan_line_idx.size > 0:
            tle_bad_raan_line_count += bad_raan_line_idx.size
            if not exist_bad_raan:
                exist_bad_raan = True
                plt.figure(3)

        bad_raan_idx = bad_indices_from_lines(bad_raan_line_idx, bad_raan, use_diff_sign=False)
        if len(bad_raan_idx) > 0:
            mask[np.array(bad_raan_idx) + i1] = False
        if bad_raan_line_idx.size > 0 or len(bad_raan_idx) > 0:
            norad_bad_raan_count += 1
            plt.plot(epoch_ds50[i1:i2], raan[i1:i2], '.-', label=norads[k])
            # for bidx in range(i1, i2):
            #     if 25636 < epoch_ds50[bidx] < 25641:
            #         plt.text(epoch_ds50[bidx], n[bidx], epoch_str[bidx] + '-' + norads[k])
        if len(bad_raan_idx) > 0:
            # print(norads[k], bad_n_line_idx, bad_n)
            for bad in bad_raan_line_idx:
                plt.scatter(epoch_ds50[(i1 + bad):(i1 + bad + 2)], raan[(i1 + bad):(i1 + bad + 2)], s=110,
                            facecolors='none', edgecolors='g')
                plt.plot(epoch_ds50[(i1 + bad):(i1 + bad + 2)], raan[(i1 + bad):(i1 + bad + 2)], 'r')
                plt.text(epoch_ds50[i1 + bad], raan[i1 + bad], epoch_str[i1 + bad])
                plt.text(epoch_ds50[i1 + bad + 1], raan[i1 + bad + 1], epoch_str[i1 + bad + 1])

            for bad in bad_raan_idx:
                plt.scatter(epoch_ds50[i1 + bad], raan[i1 + bad], s=140, facecolors='none', edgecolors='b')
                plt.text(epoch_ds50[i1 + bad], raan[i1 + bad], epoch_str[i1 + bad])

    if do_plot:
        if exist_bad_inc or exist_bad_n or exist_bad_raan:
            plt.show()
    return mask


def process_batch_tles(entire_file, section_pos, norads, norad_id_range):
    if norad_id_range == '45178--45237':
        entire_file, section_pos = unshuffle_launch_4(entire_file, section_pos, norads)

    # sort each section by epoch
    for k in range(len(norads)):
        i1 = section_pos[k]
        i2 = section_pos[k + 1]
        # below sort works as long as the satellites are 2000 or later
        entire_file[i1:i2] = sorted(entire_file[i1:i2], key=lambda x: float(x[0][18:32]))

    epoch_str, epoch_ds50, dn_o2, ddn_o6, _, inc, _, _, _, _, n, _ = make_batch_tle_arrays(entire_file)
    df = make_batch_tle_arrays(entire_file, return_pandas=True)
    section_idx = np.zeros(df.shape[0], dtype=np.int32)
    for k in range(len(norads)):
        section_idx[section_pos[k]:section_pos[k + 1]] = k
    df = df.assign(section_idx=section_idx)

    mask = identify_bad_tles(section_pos, epoch_str, epoch_ds50, norads, inc, n,
                             inc_std=53.03, inc_margin=0.5,
                             min_dt=0.2, max_dndt=0.11, max_dn=0.1, reentry_n=16.1,
                             do_plot=False)

    df2 = df[mask]
    df2.reset_index(drop=True)
    new_section_pos = np.zeros(section_pos.shape, dtype=np.int32)
    prev_idx = -1
    for k, s_idx in enumerate(df2['section_idx'].values):
        if s_idx != prev_idx:
            prev_idx = s_idx
            new_section_pos[s_idx] = k
    new_section_pos[-1] = df2.shape[0]
    return df2, new_section_pos


def format_float_after_decimal(fmt, val, keep_point):
    ret = fmt % val
    if keep_point:
        if ret.startswith("0."):
            return " " + ret[1:]
        if ret.startswith("-0."):
            return "-" + ret[2:]
    else:
        if ret.startswith("0."):
            return " " + ret[2:]
        if ret.startswith("-0."):
            return "-" + ret[3:]
    return ret


def recreate_single_tle(row, norad, int_desig):
    dn_o2_str = format_float_after_decimal('%.8f', row['dn_o2'], keep_point=True)
    ddn_o6_str = write_tle_float(row['ddn_o6'])
    bstar_str = write_tle_float(row['bstar'])
    ecc_str = format_float_after_decimal('%.7f', row['ecc'], keep_point=False)
    out  = f"1 {norad}U {int_desig} {row['epoch_str']} {dn_o2_str} {ddn_o6_str} {bstar_str} 0  9999"
    out2 = f"2 {norad} {row['inc']:8.4f} {row['raan']:8.4f}{ecc_str} {row['argp']:8.4f} {row['M']:8.4f} {row['n']:11.8f}{row['rev_num']:5d}9"
    return out + '\n' + out2


def recreate_tle_range(df, norad, int_desig):
    strings = df.apply(lambda r: recreate_single_tle(r, norad, int_desig), axis=1, result_type='reduce')
    return strings.str.cat(sep='\n')


if __name__ == "__main__":
    fname = '../spacetrack_tle/starlink_tle_44713--44772.txt'

    entire_file, section_pos, norad_ids = import_batch_tle(fname)
    print(entire_file[2:4])
    print(from_lines(entire_file[2], entire_file[3]))
