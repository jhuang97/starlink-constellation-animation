import numpy as np
import pandas as pd

df = pd.read_json('../sat_info/starlink_satcat_2021-08-27.json', orient='records')
launches_df = pd.read_pickle('../sat_info/starlink_launches.pkl')
mission_name_dict = dict(zip(launches_df.launch_date, launches_df.Mission))
print(df.columns)

df1 = df[['NORAD_CAT_ID', 'SATNAME', 'LAUNCH', 'DECAY']]
print(df1)
cat_ids = df['NORAD_CAT_ID'].to_numpy()
cat_ids.sort()
print(cat_ids)
cat_ids = np.append(cat_ids, -1)

id_str_list = []
is_streak = False
streak_start = -1
prev = -1
launch_dates = []
for k, nid in enumerate(cat_ids):
    if is_streak:
        if nid != prev+1:  # finish streak
            if streak_start != prev:
                id_str_list.append(f'{streak_start}--{prev}')
                launch_dates.append(df.LAUNCH[k-1])
            else:
                id_str_list.append(f'{prev}')
                launch_dates.append(df.LAUNCH[k-1])
            is_streak = False
            streak_start = nid
    else:
        if nid == prev+1:
            is_streak = True
        else:  # finish streak
            if k > 0:
                id_str_list.append(f'{prev}')
                launch_dates.append(df.LAUNCH[k-1])
            streak_start = nid
    prev = nid

out_str = ','.join(id_str_list)
print(out_str)
print(len(id_str_list))

# launch_dates = df1.LAUNCH.unique()
print(launch_dates)
print(len(launch_dates))

with open('../sat_info/shell_1_ids_by_launch.txt', 'w') as mfile:
    for idx, l in enumerate(launch_dates):
        m_name = mission_name_dict[l]
        if 'v1.0 L' in m_name:
            line = '\t'.join([l, m_name, id_str_list[idx]])
            print(line)
            mfile.write(line + '\n')
