import os
import numpy as np
import Tool_Functions.Functions as Functions
import post_processing.parenchyma_enhancement as further_rescale


fn_list = os.listdir('/home/zhoul0a/Desktop/prognosis_project/rescaled_ct_array/')

for fn in fn_list[::4]:
    print("processing:", fn)
    time = fn.split('_')[1]
    mouth = time.split('-')[1]
    mouth = int(mouth)
    print(mouth)
    if mouth == 7:
        print("not hospitalize")
        continue
    if os.path.exists('/home/zhoul0a/Desktop/prognosis_project/rescaled_ct_enhanced/hospitalization/' + fn[:-4] + '.npz'):
        print(fn, "processed")
        continue
    path = '/home/zhoul0a/Desktop/prognosis_project/rescaled_ct_array/' + fn
    rescaled_array = np.load(path)
    Functions.array_stat(rescaled_array)

    new_array = further_rescale.prepare_arrays_raw_for_normal_and_hospitalize(rescaled_array, normal=False, mask_name=fn[:-4],
                                                                              save_dict='/home/zhoul0a/Desktop/prognosis_project/masks/hospitalization/')
    new_array = new_array[:, :, :, 0]

    Functions.save_np_array('/home/zhoul0a/Desktop/prognosis_project/rescaled_ct_enhanced/hospitalization/', fn[:-4], new_array, True)
