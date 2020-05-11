import numpy as np
import Functions
import visualize_mask_and_raw

prediction = np.load('/home/zhoul0a/Desktop/COVID-19/visualization/6_unet_npz/xgfy-A000042_2020-03-02_prediction.npz')['prediction']

print(np.min(prediction), np.max(prediction))

array = np.load('/home/zhoul0a/Desktop/COVID-19/arrays_raw/xgfy-A000042_2020-03-02.npy')

data = array[:,:,:,0]
gt = array[:,:,:,1]

for i in range(0, 100, 10):
    test = np.array(prediction > i/100, 'float32')
    print(Functions.f1_sore_for_binary_mask(test, gt))
exit()

visualize_mask_and_raw.visualize_mask_and_raw_array('/home/zhoul0a/Desktop/COVID-19/visualization/plotting/A042/', data, prediction)


exit()

array = np.load('/home/zhoul0a/Desktop/COVID-19/visualization/aug_100_npz/xgfy-A000015_2020-03-03_prediction.npz')['prediction']


print(np.shape(array))

Functions.image_show(array[:,:,300])