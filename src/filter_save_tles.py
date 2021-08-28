import numpy as np
from tle_util import *

if __name__ == "__main__":
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
    launch_idx = 11
    norad_id_range = norad_ranges[launch_idx]  # check: 3, 11, 12
    # print(norad_id_range)

    for launch_idx, norad_id_range in enumerate(norad_ranges[27:28]):
        print(norad_id_range)
    # for _ in [0]:
        entire_file, section_pos, norads, int_desigs = import_batch_tle_v2('../spacetrack_tle/starlink_tle_' + norad_id_range + '.txt')

        if norad_id_range == '45178--45237':
            entire_file, section_pos = unshuffle_launch_4(entire_file, section_pos, norads)

        # entire_file, section_pos, norads, int_desigs = import_batch_tle_v2(fname)

        # sort each section by epoch
        for k in range(len(norads)):
            i1 = section_pos[k]
            i2 = section_pos[k + 1]
            # below sort works as long as the satellites are 2000 or later
            entire_file[i1:i2] = sorted(entire_file[i1:i2], key=lambda x: float(x[0][18:32]))

        # print(norads)
        # print(norads.dtype)
        # print(int_desigs)
        # print(int_desigs.dtype)
        # print(len(section_pos), len(norads), len(int_desigs))
        # print(entire_file[0])
        # print(entire_file[1])
        # print('...')
        # print(entire_file[section_pos[1] - 1])
        # print(entire_file[section_pos[1]])
        # print(entire_file[section_pos[1] + 1])
        # print('...')
        # print(entire_file[-2])
        # print(entire_file[-1])

        # dt = datetime(2019, 1, 1) + timedelta(318.91668981 - 1)
        # print(days_since_yr(dt, 1950))

        epoch_str, epoch_ds50, dn_o2, ddn_o6, bstar, inc, raan, _, _, _, n, _ = make_batch_tle_arrays(entire_file)
        df = make_batch_tle_arrays(entire_file, return_pandas=True)
        section_idx = np.zeros(df.shape[0], dtype=np.int32)
        for k in range(len(norads)):
            section_idx[section_pos[k]:section_pos[k+1]] = k
        df = df.assign(section_idx=section_idx)

        # for k in range(len(norads)):
        #     i1 = section_pos[k]
        #     i2 = section_pos[k + 1]
        #     # plt.plot(ddn_o6[i1:i2], n[i1:i2], '.-')
        #     plt.plot(epoch_ds50[i1:i2], ddn_o6[i1:i2], '.-')
        # plt.show()
        filter_dn_o2 = True  # I'm not sure if filter_dn_o2 is a good idea...
        if norad_id_range in ['44914--44973', '45044--45103', '45178--45237', '45360--45419']:
            filter_dn_o2 = False
        mask = identify_bad_tles(section_pos, epoch_str, epoch_ds50, norads, inc, n, raan, dn_o2,
                    inc_std=53.03, inc_margin=0.5,
                    min_dt=0.2, max_dndt=0.11, max_dn=0.1, reentry_n=16.1, max_dradt=10,
                    high_n=15.9, filter_dn_o2=filter_dn_o2,
                    do_plot=False)

        # print(np.where(~mask))
        df2 = df[mask]
        df2.reset_index(drop=True)
        new_section_pos = np.zeros(section_pos.shape, dtype=np.int32)
        prev_idx = -1
        for k, s_idx in enumerate(df2['section_idx'].values):
            if s_idx != prev_idx:
                prev_idx = s_idx
                new_section_pos[s_idx] = k
        new_section_pos[-1] = df2.shape[0]

        # print(df.shape, df2.shape)
        # print(df2.shape)
        # print(section_pos)
        # print(new_section_pos)
        if True:
            fig = plt.figure(figsize=(16, 9))
            for k in range(len(norads)):
                if norads[k] in ['48601', '44957', '44958', '44959', '44973', '44965', '44970', '45055', '45097']:
                    i1, i2 = new_section_pos[k], new_section_pos[k+1]
                    # plt.plot(df2['epoch_ds50'][i1:i2], df2['n'][i1:i2], '.-', label=norads[k])
                    plt.plot(df2['epoch_ds50'][i1:i2], df2['dn_o2'][i1:i2], '.-', label=norads[k])
                    # plt.plot(df2['epoch_ds50'][i1:i2], df2['inc'][i1:i2], '.-')
            plt.title(str(launch_idx) + ' ' + norad_id_range)
            plt.legend()
            plt.show()

        # print(df2.dtypes)
        # print(df2['epoch_str'].to_numpy().astype('U').dtype)

        # save filtered TLEs
        save_tle_npz('../filtered_tle_data/f_' + norad_id_range, df2, new_section_pos, norads,
            int_desigs)
