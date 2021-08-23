import numpy as np
import Tool_Functions.Functions as Functions
import cv2


def show_2d_function(func, x_range=(0, 1), y_range=(0, 1), resolution=(1000, 1000), leave_cpu_num=1, show=True):
    resolution_x = resolution[1]
    resolution_y = resolution[0]
    step_x = (x_range[1] - x_range[0])/resolution_x
    step_y = (y_range[1] - y_range[0])/resolution_y
    import multiprocessing as mp
    cpu_cores = mp.cpu_count() - leave_cpu_num
    pool = mp.Pool(processes=cpu_cores)
    locations_x = np.ones([resolution_y, resolution_x], 'float32') * np.arange(x_range[0], x_range[1], step_x)
    locations_y = np.ones([resolution_y, resolution_x], 'float32') * np.arange(y_range[0], y_range[1], step_y)
    locations_y = cv2.flip(np.transpose(locations_y), 0)
    locations = np.stack([locations_x, locations_y], axis=2)
    locations = np.reshape(locations, [resolution_y * resolution_x, 2])
    picture = np.array(pool.map(func, locations), 'float32')
    picture = np.reshape(picture, [resolution_y, resolution_x])
    if show:
        Functions.image_show(picture)
    return picture

