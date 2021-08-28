import math
from datetime import datetime, timedelta
import numpy as np


def ds50UTC_to_datetime(ds):
    return datetime(1950, 1, 1) + timedelta(ds - 1)


def days_since_yr(dt, yr):
    return (dt - datetime(yr, 1, 1) + timedelta(1)).total_seconds()/86400.


def shift_time(ds, n_day_divs):
    frac, whole = math.modf(ds)
    return whole + math.ceil(n_day_divs * frac) / n_day_divs


def import_batch_tle(fname):
    with open(fname, 'r') as fp:
        line = fp.readline()
        line_no = 0
        prev_id = ''
        section_pos = []
        norad_ids = []
        while line:
            assert len(line) > 10
            line_type = line[0]
            if line_type == '1':
                this_id = line[2:7]
                if this_id != prev_id:
                    # print(this_id, line_no)
                    norad_ids.append(this_id)
                    section_pos.append(line_no)
                    prev_id = this_id

            line = fp.readline()
            line_no += 1
        fp.seek(0)
        entire_file = fp.read().splitlines()
        return entire_file, section_pos, norad_ids


# Input arrays in degrees
def circular_interp(arr1, arr2, weights):
    arr1 = np.exp(1j * arr1 * np.pi / 180)
    arr2 = np.exp(1j * arr2 * np.pi / 180)
    return np.remainder(np.angle(arr1 * (1-weights) + arr2 * weights, deg=True), 360.)


def epoch_match(tle, epoch_list):
    for e in epoch_list:
        if e == tle[18:32]:
            return True
    return False
