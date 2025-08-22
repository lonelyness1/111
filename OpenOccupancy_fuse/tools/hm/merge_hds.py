import numpy as np


hds_paths = [
    '/root/tools/h5tool_new/nuscenens_occupancy.hds',
    '/root/tools/h5tool_new/hds_label_icu30_nuscenes.hds',
]

save_path = '/root/tools/h5tool_new/nuscenes_all.hds'

data_list = []
for hds in hds_paths:
    lst = np.loadtxt(hds, np.str).tolist()
    data_list += lst

hds_all = np.array(data_list)

np.savetxt(save_path, hds_all, fmt='%s')
