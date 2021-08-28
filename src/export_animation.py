import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import time
from datetime import date, datetime, timedelta
from my_util import *
from colormath.color_objects import LCHuvColor, sRGBColor
from colormath.color_conversions import convert_color
from matplotlib.colors import ListedColormap
from scipy.ndimage import convolve1d

mode = 1

GM = 3.9860e14  # m^3 s^-2
const = GM / 4 / math.pi / math.pi
# r_E = 6371  # km, mean radius
r_E = 6378.1  # km, equatorial radius


def perdelta(start, end, delta):
    curr = start
    while curr < end:
        yield curr
        curr += delta

n_day_divs = 48
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
id_ranges = []
launch_names = []
for k, nr in enumerate(norad_ranges):
    fname = '../animation_data/anim_data_' + nr + '_day' + str(n_day_divs) + '.npz'
    if os.path.isfile(fname):
        id_ranges.append(nr)
        launch_names.append(launch_names_all[k])
# norad_id_range = '47413--47422'  # v1.0 Tr-1
# id_ranges = ['44713--44772', '44914--44973', '45044--45103', '45178--45237']
# norad_id_range = '45044--45103'
# norad_id_range = '47787--47846'

# made colormaps
h0 = 220
dh = 37
n_pts = 30
n_colors = len(id_ranges)
n_steps = 30
my_cmaps = []
for k in range(n_colors):
    cmap_mat = np.zeros((n_steps, 4))
    ch_vals = np.linspace(85., 30., n_steps)
    h = h0 + k * dh
    for k2, ch in enumerate(ch_vals):
        c_tup = convert_color(LCHuvColor(62, ch, h), sRGBColor).get_value_tuple()
        cmap_mat[k2, :3] = np.array(c_tup)
    cmap_mat[:, 3] = 0.4
    my_cmaps.append(ListedColormap(cmap_mat))

data = []
for id_range in id_ranges:
    fname = 'anim_data_' + id_range + '_day' + str(n_day_divs)
    data.append(np.load('../animation_data/' + fname + '.npz', allow_pickle=True))

for k in data[0].keys():
    print(k)

t_start = datetime(2019, 11, 14, 18, 0, 0)
t_end = datetime(2021, 8, 27)
t_orbit2 = datetime(2020, 11, 24, 19)
tidx_o2 = int((t_orbit2 - t_start) / timedelta(days=1) * n_day_divs)
t_all = list(perdelta(t_start, t_end, timedelta(days=1)/n_day_divs))
n_ticks = len(t_all)
ref_alt = 550. * np.ones(n_ticks)
ref_alt[tidx_o2:] = 547.5
# print(n_ticks)

rel_node = []
rel_long = []
alts = []
cvals = []
flen = 12
in_orbit_by_sat = []
num_orbit_by_launch = []
launch_tidx = []
total_in_orbit = np.zeros(n_ticks)
num_at_target_alt = np.zeros(n_ticks)
print('calculating... ')
for k in range(len(id_ranges)):
    print(k, len(id_ranges))
    rel_node.append(data[k]['plot_data'][:, :, 0])
    rel_long.append(data[k]['plot_data'][:, :, 1])
    alts.append(data[k]['plot_data'][:, :, 2])
    # print(data[k]['plot_data'].shape)
    alt_correct = (np.abs(ref_alt[:, np.newaxis] - alts[k]) < 0.3).astype(int).astype(float)
    new_cval = convolve1d(alt_correct, np.ones(flen)/flen, axis=0, mode='constant', origin=-flen//2)
    num_at_target_alt += np.sum((new_cval > 0.95).astype(int), axis=-1)
    cvals.append(new_cval)
    in_orbit_by_sat.append(np.logical_not(np.isnan(rel_node[k])))
    num_orbit_by_launch.append(np.sum(in_orbit_by_sat[k].astype(int), axis=1))
    total_in_orbit += num_orbit_by_launch[k]
    launch_tidx.append(np.argmax(num_orbit_by_launch[k] > 0))

if mode == 1:

    fig = plt.figure(tight_layout=True, figsize=(12, 8))
    gs = fig.add_gridspec(4, 2, width_ratios=[6, 1])
    ax1 = fig.add_subplot(gs[:2, 0])
    ax2 = fig.add_subplot(gs[2:, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1:, 1])
    ax3.axis('off')

    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_ylabel('Co-Rotating Anomaly past the Ascending Node (°)')
    ax2.set_xlabel('Co-Precessing Longitude of the Ascending Node (°)')
    ax2.set_ylabel('Altitude (km)')
    ax3.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    tx_dt1 = ax3.text(0.9, 0.74, '', fontsize='x-large', family='monospace', ha='right')
    tx_dt2 = ax3.text(0.9, 0.6, '', fontsize='x-large', family='monospace', ha='right')
    tx_at_alt = ax3.text(0.05, 0.35 + 0.12, 'At target altitude:')
    tx_at_alt_count = ax3.text(0.9, 0.35, '', fontsize='x-large', family='monospace', ha='right')
    tx_total = ax3.text(0.05, 0.05 + 0.12, 'Total in orbit:')
    tx_total_count = ax3.text(0.9, 0.05, '', fontsize='x-large', family='monospace', ha='right')

    ax4.tick_params(left=False, labelleft=False,
                    labelbottom=False, bottom=False)

    num_per_shell = 22
    x_legend = np.array([0, 2] + [1]*num_per_shell + [3]*num_per_shell)
    c_idx_legend = np.array([0, 1] + [0]*num_per_shell + [1]*num_per_shell)
    ax4.set_xlim(-3.5, 3.5)
    ax4.set_ylim(-1, 30)

    x, y1, y2, sc1, sc2 = [], [], [], [], []
    for k in range(len(id_ranges)):
        x.append([])
        y1.append([])
        y2.append([])
        sc1.append(ax1.scatter(x[k], y1[k], c=[], cmap=my_cmaps[k], vmin=0, vmax=1))
        sc2.append(ax2.scatter(x[k], y2[k], c=[], cmap=my_cmaps[k], vmin=0, vmax=1))


    def a_fwd(arr, ref_alt): # ref_alt 550
        return 1/(1 + np.exp(-(arr - ref_alt)*0.009))
    def a_inv(arr, ref_alt): # ref_alt 550
        return np.log(1/arr - 1)/(-0.009) + ref_alt
    def a_fwd(arr, alt_hi, alt_lo):
        return np.piecewise(arr, [arr < alt_lo, (alt_lo <= arr) & (arr < alt_hi), arr >= alt_hi],
            [lambda x: (x - alt_lo) / 3 + alt_lo, lambda x: x,
             lambda x: (x - alt_hi) * 5 + alt_hi])


    def a_inv(arr, alt_hi, alt_lo):
        return np.piecewise(arr, [arr < alt_lo, (alt_lo <= arr) & (arr < alt_hi), arr >= alt_hi],
            [lambda x: (x - alt_lo) * 3 + alt_lo, lambda x: x,
             lambda x: (x - alt_hi) / 5 + alt_hi])


    def init():
        ax1.set_xlim(0, 360)
        ax1.set_ylim(0, 360)
        # ax1.set_ylim(165, 175)
        ax1.xaxis.set_major_locator(MultipleLocator(20))
        ax1.xaxis.set_minor_locator(MultipleLocator(5))
        ax1.yaxis.set_major_locator(MultipleLocator(45))

        ax1.invert_yaxis()
        ax2.set_yscale('function', functions=(lambda a: a_fwd(a, 520, 300), lambda a: a_inv(a, 520, 300)))

        ax2.set_ylim(80, 580)
        ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%g'))
        # ax2.set_ylim(546, 551)
        # ax2.set_yticks([547.5, 550-0.2, 550, 550+0.2])

        ax1.set_axisbelow(True)
        ax1.grid()
        ax1.grid(b=True, which='minor', color='lightgray')
        ax2.set_axisbelow(True)
        ax2.grid()
        ax2.grid(b=True, which='minor', color='lightgray')
        # return line,

        for spine in ax4.spines.values():
            spine.set_edgecolor('0.8')


    anim_skip = 1


    def animate(i):
        # ax1.set_title(str(i) + " - " + str(t_all[i]))
        tx_dt1.set_text(t_all[i].strftime("%Y-%m-%d"))
        tx_dt2.set_text(t_all[i].strftime("%H:%M:%S"))

        if tidx_o2 <= i < tidx_o2 + anim_skip + 1:
            ax2.set_yticks([100, 200, 300, 400, 520, 547.5, 580])
            ax2.set_yticks(list(range(320, 400, 20)) + list(range(420, 520, 20)) + [530, 540, 547.5, 550, 560, 570],
                           minor=True)
        elif 0 <= i < anim_skip + 1:
            ax2.set_yticks([100, 200, 300, 400, 520, 550, 580])
            ax2.set_yticks(list(range(320, 400, 20)) + list(range(420, 520, 20)) + [530, 540, 547.5, 550, 560, 570],
                           minor=True)

        # mask = np.isnan(rel_node[i, :])
        # line.set_data(rel_node[i, ~mask], rel_longitude[i, ~mask])
        # x = rel_node[i, :]
        # y = rel_longitude[i, :]

        # line.set_data(rel_node[i, :], rel_longitude[i, :])
        # line.set_data(rel_node[i, :], altitudes[i, :])
        # x = rel_node[i, :]
        # y = altitudes[i, :]

        # x = data[0]['plot_data'][i, :, 0]
        # y = data[0]['plot_data'][i, :, 2]
        for k in range(len(id_ranges)):
            x[k] = rel_node[k][i, :]
            y1[k] = rel_long[k][i, :]
            y2[k] = alts[k][i, :]
            sc1[k].set_offsets(np.c_[x[k], y1[k]])
            sc2[k].set_offsets(np.c_[x[k], y2[k]])
            sc1[k].set_array(cvals[k][i, :])
            sc2[k].set_array(cvals[k][i, :])

        tx_total_count.set_text(str(round(total_in_orbit[i])))
        tx_at_alt_count.set_text(str(round(num_at_target_alt[i])))

        for k, tidx in enumerate(launch_tidx):
            if tidx <= i < tidx + anim_skip + 1:
                ax4.scatter(x_legend, k * np.ones(len(x_legend)), c=c_idx_legend, cmap=my_cmaps[k], vmin=0, vmax=1)
                ax4.text(-0.5, k, launch_names[k], ha='right', va='center')

        # return line,


    ani = animation.FuncAnimation(
        fig, animate, frames=range(0, n_ticks, anim_skip), interval=1000/60, init_func=init)
    # plt.show()
    FFwriter = animation.FFMpegWriter(fps=60)
    ani.save('Starlink_animation_2021-08-27.mp4', writer=FFwriter, dpi=160)

elif mode == 2:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for k in range(len(id_ranges)):
        ax1.plot(t_all, num_orbit_by_launch[k], label=launch_names[k])
    ax1.legend()
    ax2.plot(t_all, total_in_orbit, label='total in orbit')
    ax2.plot(t_all, num_at_target_alt, label='# at target altitude')
    ax2.legend()
    plt.show()

# print(np.shape(data[0]['plot_data']))
print(str(t_all[0]))
# print(np.c_[data[0]['plot_data'][2, :, 0], data[0]['plot_data'][2, :, 2]])
# print(type(data[0]['plot_data'][30, :, 0]))
