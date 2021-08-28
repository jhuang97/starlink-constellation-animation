from SatProp import Sgp4Prop
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from datetime import date, datetime, timedelta
from my_util import *
from scipy.stats import linregress
from scipy.signal import savgol_filter
from collections import defaultdict
from tle_util import *


GM = 3.9860e14  # m^3 s^-2
const = GM / 4 / math.pi / math.pi
# r_E = 6371  # km, mean radius
r_E = 6378.1  # km, equatorial radius


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
        print('TLE starts after end of time window')
        start_src_idx = None
        time_within_range = False
    else:
        start_src_idx = 0
    if end_dest_idx > n_ticks:
        end_src_idx = len(times_dt) - (end_dest_idx - n_ticks)
        end_dest_idx = n_ticks
    elif end_dest_idx < 0:
        print('TLE ends before start of time window')
        end_src_idx = None
        time_within_range = False
    else:
        end_src_idx = len(times_dt)
    return start_dest_idx, end_dest_idx, start_src_idx, end_src_idx, time_within_range


# app = Sgp4Prop('', '../input/44238_tle.inp', n_day_divs=32)
# times_dt, mean_elems, osc_elems = app.run()

some_6P_card = '0.00000030.            480.                                          6P'

# earliest_time = datetime(2100, 1, 1)
# latest_time = datetime(1900, 1, 1)

start_new_orbit = datetime(2020, 11, 24, 19)
# print('new ref orbit, time epoch 2020 ', days_since_yr(start_new_orbit, 2020))


def calculate_data(norad_id_range, n_day_divs, t_start, t_end):
    tle_data = np.load('../filtered_tle_data/f_' + norad_id_range + '.npz')
    print(list(tle_data.keys()))
    tle_df = tle_npz_to_df(tle_data)
    print(tle_df)
    section_pos = tle_data['section_pos']
    num_sats = len(section_pos) - 1
    norad_ids = tle_data['norad']
    t_all = list(perdelta(t_start, t_end, timedelta(days=1)/n_day_divs))
    n_ticks = len(t_all)
    print(n_ticks)
    start_indices = {}
    end_indices = {}  # not inclusive
    start_indices_list = []
    end_indices_list = []
    norad_ids_list = []

    # build arrays to hold on all satellite info
    mean_elems_all = np.zeros((n_ticks, 6, num_sats))
    osc_elems_all = np.zeros((n_ticks, 6, num_sats))

    # reference satellite
    app = Sgp4Prop('reference_sat', '../input/reference_satellite_1.inp', n_day_divs=n_day_divs, end_dt=t_end)
    times_dt_ref, mean_elems_ref, osc_elems_ref = app.run()
    start_dest_idx, end_dest_idx, start_src_idx, end_src_idx, time_within_range = align_times(times_dt_ref, t_start, n_day_divs, n_ticks)
    assert(start_dest_idx == 0 and end_dest_idx == n_ticks)
    mean_elems_ref = mean_elems_ref[start_src_idx:end_src_idx, :]
    osc_elems_ref = osc_elems_ref[start_src_idx:end_src_idx, :]

    calc_range = range(num_sats)
    # calc_range = [1, 4, 5, 6, 8]

    for k in calc_range:
        i1 = section_pos[k]
        i2 = section_pos[k + 1]
        lines = recreate_tle_range(tle_df[i1:i2], norad_ids[k], tle_data['int_desig'][k])
        my_path = '../input/test_' + norad_ids[k] + '.inp'
        with open(my_path, 'w') as fw:
            fw.write(some_6P_card + '\n')
            fw.write(lines)

        app = Sgp4Prop('', my_path, n_day_divs=n_day_divs, backtrack=True)
        times_dt, mean_elems, osc_elems = app.run()

        # figure where in matrix to insert orbit history
        start_dest_idx, end_dest_idx, start_src_idx, end_src_idx, time_within_range = align_times(times_dt, t_start, n_day_divs, n_ticks)
        assert(end_dest_idx - start_dest_idx == end_src_idx - start_src_idx)
        start_indices[k] = start_dest_idx
        end_indices[k] = end_dest_idx
        start_indices_list.append(start_dest_idx)
        end_indices_list.append(end_dest_idx)
        norad_ids_list.append(norad_ids[k])
        mean_elems_all[start_dest_idx:end_dest_idx, :, k] = mean_elems[start_src_idx:end_src_idx, :]
        osc_elems_all[start_dest_idx:end_dest_idx, :, k] = osc_elems[start_src_idx:end_src_idx, :]
        mean_elems_all[:start_dest_idx, :, k] = np.nan
        osc_elems_all[:start_dest_idx, :, k] = np.nan
        mean_elems_all[end_dest_idx:, :, k] = np.nan
        osc_elems_all[end_dest_idx:, :, k] = np.nan

        os.remove(my_path)

    long_past_asc_node = np.remainder(osc_elems_all[:, 4, :] + osc_elems_all[:, 5, :], 360.)
    altitudes = (const * (86400 / mean_elems_all[:, 0, :]) ** 2) ** (1/3) / 1000 - r_E

    rel_node = np.remainder(mean_elems_all[:, 3, :] - np.transpose(np.reshape(np.tile(mean_elems_ref[:, 3], num_sats), (-1, n_ticks))), 360.)
    rel_longitude = np.remainder(long_past_asc_node[:, :] - np.transpose(np.reshape(np.tile(mean_elems_ref[:, 5], num_sats), (-1, n_ticks))), 360.)

    for k in calc_range:
        altitudes[start_indices[k]:end_indices[k], k] = savgol_filter(altitudes[start_indices[k]:end_indices[k], k], 31, 3)

        rel_rad = rel_longitude[start_indices[k]:end_indices[k], k] / 180 * np.pi
        rel_rad_filtered = savgol_filter(np.vstack((np.cos(rel_rad), np.sin(rel_rad))), 11, 3)
        rel_longitude[start_indices[k]:end_indices[k], k] = np.remainder(np.arctan2(rel_rad_filtered[1], rel_rad_filtered[0]) / np.pi * 180, 360.)

    plot_data = np.dstack((rel_node, rel_longitude, altitudes))
    t_lims = np.array([t_start, t_end])

    fname = 'anim_data_' + norad_id_range + '_day' + str(n_day_divs)
    np.savez_compressed('../animation_data/' + fname, plot_data=plot_data,
        idx1=start_indices_list, idx2=end_indices_list, norad_ids=norad_ids_list)


# norad_id_range = '47413--47422'  # v1.0 Tr-1
# norad_id_range = '44713--44772'  # v1.0 L1
# norad_id_range = '44914--44973'  # v1.0 L2
# norad_id_range = '45044--45103'
# norad_id_range = '45178--45237'  # v1.0 L4
# norad_id_range = '47787--47846'

launch_dates = []
launch_names = []
norad_ranges = []
with open('../sat_info/shell_1_ids_by_launch.txt', 'r') as f:
    lines = f.read().splitlines()
    for l in lines:
        tokens = l.split('\t')
        launch_dates.append(tokens[0])
        launch_names.append(tokens[1])
        norad_ranges.append(tokens[2])
print(norad_ranges)

n_day_divs = 48
t_start = datetime(2019, 11, 14, 18, 0, 0)
t_end = datetime(2021, 8, 27)

for nr in norad_ranges[15:]:
    calculate_data(nr, n_day_divs, t_start, t_end)

# calculate_data(norad_id_range, n_day_divs, t_start, t_end)
# print('anim_data_' + norad_id_range + '_day' + str(n_day_divs))

# t_lims = np.array([t_start, t_end])
# np.savez_compressed('../animation_data_old/test', a=t_lims)
# loaded = np.load('../animation_data_old/test.npz', allow_pickle=True)
# print(loaded['a'])