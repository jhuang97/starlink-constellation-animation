from tle_util import *
from SatProp import Sgp4Prop
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import date, datetime, timedelta
from my_util import *
from scipy.stats import linregress
from scipy.signal import savgol_filter
import math
from collections import defaultdict


GM = 3.9860e14  # m^3 s^-2
const = GM / 4 / math.pi / math.pi
# r_E = 6371  # km, mean radius
r_E = 6378.1  # km, equatorial radius

some_6P_card = '0.00000030.            480.                                          6P'


def perdelta(start, end, delta):
    curr = start
    while curr < end:
        yield curr
        curr += delta


def align_times(times_dt, t_start, n_day_divs, n_ticks):
    start_dest_idx = round((times_dt[0] - t_start)/timedelta(days=1) * n_day_divs)
    end_dest_idx = round((times_dt[-1] - t_start) / timedelta(days=1) * n_day_divs) + 1
    time_within_range = True
    if start_dest_idx < 0:
        start_src_idx = -start_dest_idx
        start_dest_idx = 0
    elif start_dest_idx >= n_ticks:
        print('TLE starts after end of time window: ', my_path)
        start_src_idx = None
        time_within_range = False
    else:
        start_src_idx = 0
    if end_dest_idx > n_ticks:
        end_src_idx = len(times_dt) - (end_dest_idx - n_ticks)
        end_dest_idx = n_ticks
    elif end_dest_idx < 0:
        print('TLE ends before start of time window: ', my_path)
        end_src_idx = None
        time_within_range = False
    else:
        end_src_idx = len(times_dt)
    return start_dest_idx, end_dest_idx, start_src_idx, end_src_idx, time_within_range


if __name__ == '__main__':
    launch_dates = []
    launch_names_all = []
    norad_ranges = []
    with open('../sat_info/shell_1_ids_by_launch.txt', 'r') as f:
        lines = f.read().splitlines()
        for l in lines:
            tokens = l.split('\t')
            launch_dates.append(tokens[0])
            launch_names_all.append(tokens[1])
            norad_ranges.append(tokens[2])

    mode = 1
    if mode == 1:
        n_day_divs = 4
    elif mode == 2:  # modes 2 and 3 used to adjust parameters of reference satellite
        n_day_divs = 60
    elif mode == 3:
        n_day_divs = 120
    elif mode == 4:  # smoothing
        n_day_divs = 24

    # build arrays to hold on all satellite info
    t_start = datetime(2019, 11, 14, 18, 0, 0)
    # t_start = datetime(2020, 5, 8)
    t_end = datetime(2021, 8, 27)
    t_all = list(perdelta(t_start, t_end, timedelta(days=1) / n_day_divs))
    n_ticks = len(t_all)
    # print(t_all)
    start_indices = {}
    end_indices = {}  # not inclusive

    app = Sgp4Prop('reference_sat', '../input/reference_satellite_1.inp', n_day_divs=n_day_divs, end_dt=t_end)
    times_dt_ref, mean_elems_ref, osc_elems_ref = app.run()
    start_dest_idx, end_dest_idx, start_src_idx, end_src_idx, time_within_range = align_times(times_dt_ref, t_start,
                                                                                              n_day_divs, n_ticks)
    assert (start_dest_idx == 0 and end_dest_idx == n_ticks)
    mean_elems_ref = mean_elems_ref[start_src_idx:end_src_idx, :]
    osc_elems_ref = osc_elems_ref[start_src_idx:end_src_idx, :]

    for launch_idx, norad_id_range in enumerate(norad_ranges[26:28]):
        tle_data = np.load('../filtered_tle_data/f_' + norad_id_range + '.npz')
        print(list(tle_data.keys()))
        tle_df = tle_npz_to_df(tle_data)
        print(tle_df)
        section_pos = tle_data['section_pos']
        norad_ids = tle_data['norad']
        print(section_pos)
        if mode == 1:
            calc_range = range(len(section_pos)-1)
        elif mode == 2 or mode == 3:
            calc_range = [1, 4, 5, 6, 8]
        # calc_range = [1, 4]
        elif mode == 4:
            # calc_range = range(11, 21)
            calc_range = range(len(section_pos)-1)
            # calc_range = [20, 26, 28, 29, 30, 38, 41, 44]

        mean_elems_all = np.zeros((n_ticks, 6, len(section_pos)))
        osc_elems_all = np.zeros((n_ticks, 6, len(section_pos)))
        tle_tidx_by_sat = []
        tle_epochs_by_sat = []
        for calc_idx, k in enumerate(calc_range):
            i1 = section_pos[k]
            i2 = section_pos[k + 1]
            lines = recreate_tle_range(tle_df[i1:i2], norad_ids[k], tle_data['int_desig'][k])
            my_path = '../input/test_' + tle_data['norad'][k] + '.inp'
            with open(my_path, 'w') as fw:
                fw.write(some_6P_card + '\n')
                fw.write(lines)

            app = Sgp4Prop('', my_path, n_day_divs=n_day_divs, backtrack=True, save_tle_epochs=True)
            times_dt, mean_elems, osc_elems, tle_tidxs, tle_epochs = app.run()
            tle_tidx_by_sat.append(tle_tidxs)
            tle_epochs_by_sat.append(tle_epochs)
            # if times_dt[0] < datetime(2020, 2, 17):
            #     print(norad_ids[k])
            #     print(tle_epochs_by_sat[k][:20])

            # figure where in matrix to insert orbit history
            start_dest_idx, end_dest_idx, start_src_idx, end_src_idx, time_within_range = align_times(times_dt, t_start,
                                                                                                      n_day_divs,
                                                                                                      n_ticks)
            assert (end_dest_idx - start_dest_idx == end_src_idx - start_src_idx)
            start_indices[k] = start_dest_idx
            end_indices[k] = end_dest_idx
            mean_elems_all[start_dest_idx:end_dest_idx, :, k] = mean_elems[start_src_idx:end_src_idx, :]
            osc_elems_all[start_dest_idx:end_dest_idx, :, k] = osc_elems[start_src_idx:end_src_idx, :]
            mean_elems_all[:start_dest_idx, :, k] = np.nan
            osc_elems_all[:start_dest_idx, :, k] = np.nan
            mean_elems_all[end_dest_idx:, :, k] = np.nan
            osc_elems_all[end_dest_idx:, :, k] = np.nan

            rel_node = np.remainder(mean_elems_all[start_dest_idx:end_dest_idx, 3, k] -
                                    mean_elems_ref[start_dest_idx:end_dest_idx, 3], 360.)
            alt = (const * (86400 / mean_elems_all[start_dest_idx:end_dest_idx, 0, k]) ** 2) ** (1 / 3) / 1000 - r_E
            os.remove(my_path)

        long_past_asc_node = np.remainder(osc_elems_all[:, 4, :] + osc_elems_all[:, 5, :], 360.)

        start_new_orbit = datetime(2020, 11, 24, 19)
        print('new ref orbit, time epoch 2020 ', days_since_yr(start_new_orbit, 2020))

        if mode == 1 or mode == 4:
            for k in calc_range:
                t1 = start_indices[k]
                t2 = end_indices[k]
                # plt.plot(t_all[t1:t2], np.remainder(mean_elems_all[t1:t2, 3, k] - mean_elems_ref[t1:t2, 3], 360.))

                # longitude past ascending node and smoothing
                plt.figure(1)
                rel_long = np.remainder(long_past_asc_node[t1:t2, k] - mean_elems_ref[t1:t2, 5], 360.)
                plt.plot(t_all[t1:t2], rel_long)
                # if mode == 4:
                #     rel_rad = rel_long / 180 * np.pi
                #     rel_rad_filtered = savgol_filter(np.vstack((np.cos(rel_rad), np.sin(rel_rad))), 11, 3)
                #     smooth_rel_long = np.remainder(np.arctan2(rel_rad_filtered[1], rel_rad_filtered[0]) / np.pi * 180, 360.)
                #     plt.plot(t_all[t1:t2], smooth_rel_long)

                plt.figure(2)
                rel_node = np.remainder(mean_elems_all[t1:t2, 3, k] - mean_elems_ref[t1:t2, 3], 360.)
                plt.plot(t_all[t1:t2], rel_node, label=k)

                # altitude and smoothing
                plt.figure(3)
                alt = (const * (86400 / mean_elems_all[t1:t2, 0, k]) ** 2) ** (1 / 3) / 1000 - r_E
                plt.plot(t_all[t1:t2], alt, label=norad_ids[k])
                if np.any(alt > 600):
                    print('bad', norad_ids[k])
                # if mode == 4:
                #     alt_filtered = savgol_filter(alt, 31, 3)
                #     plt.plot(t_all[t1:t2], alt_filtered)

                # plt.plot(t_all[t1:t2], np.remainder(long_past_asc_node[t1:t2, k] - mean_elems_ref[t1:t2, 5], 360.), label=k)
            plt.legend()

        ref_time_1 = (datetime(2020, 5, 1), datetime(2020, 12, 1))
        ref_time_2 = (datetime(2021, 1, 9), datetime(2021, 5, 1))
        ref_time_3 = (start_new_orbit - timedelta(hours=3), start_new_orbit + timedelta(hours=3))

        if mode == 2 or mode == 3:
            # count laps
            angle_total = 0
            interval_start = False
            for tidx in range(start_indices[calc_range[0]] + 1, end_indices[calc_range[0]]):
                t = t_all[tidx]
                if ref_time_3[0] < t < ref_time_3[1]:
                    if not interval_start:
                        interval_start = True
                        start_time = t
                        start_idx = tidx
                    diff = long_past_asc_node[tidx, 1] - long_past_asc_node[tidx - 1, 1]
                    if long_past_asc_node[tidx, 1] < 180 < long_past_asc_node[tidx - 1, 1]:
                        diff -= 360
                    angle_total += diff
                    end_time = t
                    end_idx = tidx
            n_revs = angle_total / 360
            days_elapsed = (end_time - start_time) / timedelta(days=1)
            print(n_revs, days_elapsed, n_revs / days_elapsed)

            fig, ax = plt.subplots(1, 2)
            rlong_slopes = []
            rnode_slopes = []
            t1 = start_idx
            t2 = end_idx + 1
            if mode == 3:
                ax[0].plot(t_all[t1:t2], mean_elems_ref[t1:t2, 3], '.-', label='ref')
                ax[1].plot(t_all[t1:t2], mean_elems_ref[t1:t2, 5], '.-', label='ref')
                ax[0].grid()
                ax[1].grid()
            for k in calc_range:
                rel_node = np.remainder(mean_elems_all[t1:t2, 3, k] - mean_elems_ref[t1:t2, 3], 360.)
                rel_longitude = np.remainder(long_past_asc_node[t1:t2, k] - mean_elems_ref[t1:t2, 5], 360.)
                time_days = [(t - t_all[t1]) / timedelta(days=1) for t in t_all[t1:t2]]
                ax[0].plot(t_all[t1:t2], rel_node, label=k)
                ax[1].plot(t_all[t1:t2], rel_longitude, label=k)
                result = linregress(time_days, rel_longitude)
                rlong_slopes.append(result.slope)
                result = linregress(time_days, rel_node)
                rnode_slopes.append(result.slope)

                # plt.plot(t_all[t1:t2], mean_elems_all[t1:t2, 0, k], label=k)
                # plt.plot(t_all[t1:t2], np.remainder(long_past_asc_node[t1:t2, k] - mean_elems_ref[t1:t2, 5], 360.), label=k)

            ax[0].legend()
            ax[1].legend()
            rlong_slopes.sort()
            rnode_slopes.sort()
            print('relative longitude: ', ', '.join(['%.2E'] * len(rlong_slopes)) % tuple(rlong_slopes))
            print('relative node: ', ', '.join(['%.2E'] * len(rnode_slopes)) % tuple(rnode_slopes))

        # plt.plot(t_all, np.nanmedian(mean_elems_all[:, 0, :], axis=-1), '.-')
        # print(earliest_time, days_since_yr(earliest_time, 2019))
        # print(latest_time, days_since_yr(latest_time, 2019))
        plt.show()
