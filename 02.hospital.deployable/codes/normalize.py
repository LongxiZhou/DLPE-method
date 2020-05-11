import numpy as np
import warnings
import cv2


def rescale_to_standard(array, resolution, target_resolution=(334/512, 334/512, 1), target_shape=(512, 512, 512)):
    # pad and rescale the array to the same resolution and shape for further processing.
    # input: array must has shape (x, y, z) and resolution is a list or tuple with three elements

    original_shape = np.shape(array)
    target_volume = (target_resolution[0]*target_shape[0], target_resolution[1]*target_shape[1], target_resolution[2]*target_shape[2])
    shape_of_target_volume = (int(target_volume[0]/resolution[0]), int(target_volume[1]/resolution[1]), int(target_volume[2]/resolution[2]))

    if original_shape[2] * resolution[2] > target_volume[2]:
        warnings.warn('z-axis is longer than expectation. Make sure lung is near the center of z-axis.', SyntaxWarning)
        array = array[:, :, 100::]
        original_shape = np.shape(array)

    x = max(shape_of_target_volume[0], original_shape[0]) + 2
    y = max(shape_of_target_volume[1], original_shape[1]) + 2
    z = max(shape_of_target_volume[2], original_shape[2]) + 2

    x_start = int(x/2)-int(original_shape[0]/2)
    x_end = x_start + original_shape[0]
    y_start = int(y/2)-int(original_shape[1]/2)
    y_end = y_start + original_shape[1]
    z_start = int(z / 2) - int(original_shape[2] / 2)
    z_end = z_start + original_shape[2]

    array_intermediate = np.zeros((x, y, z), 'float32')
    array_intermediate[x_start:x_end, y_start:y_end, z_start:z_end] = array

    x_start = int(x / 2) - int(shape_of_target_volume[0] / 2)
    x_end = x_start + shape_of_target_volume[0]
    y_start = int(y / 2) - int(shape_of_target_volume[1] / 2)
    y_end = y_start + shape_of_target_volume[1]
    z_start = int(z / 2) - int(shape_of_target_volume[2] / 2)
    z_end = z_start + shape_of_target_volume[2]

    array_intermediate = array_intermediate[x_start:x_end, y_start:y_end, z_start:z_end]  # Now the array is padded

    # rescaling:
    array_standard_xy = np.zeros((target_shape[0], target_shape[1], shape_of_target_volume[2]), 'float32')
    for s in range(shape_of_target_volume[2]):
        array_standard_xy[:, :, s] = cv2.resize(array_intermediate[:, :, s], (target_shape[0], target_shape[1]), cv2.INTER_LANCZOS4)

    array_standard = np.zeros(target_shape, 'float32')
    for s in range(target_shape[0]):
        array_standard[s, :, :] = cv2.resize(array_standard_xy[s, :, :], (target_shape[1], target_shape[2]), cv2.INTER_LINEAR)

    return array_standard


def rescale_to_original(array, resolution, target_resolution=(334/512, 334/512, 1), target_shape=(512, 512, 512)):
    # pad and rescale the array to the same resolution and shape for further processing.
    # input: array must has shape (x, y, z) and resolution is a list or tuple with three elements

    print('rescaling prediction to previous shape:', target_resolution, target_shape)

    original_shape = np.shape(array)
    target_volume = (target_resolution[0]*target_shape[0], target_resolution[1]*target_shape[1], target_resolution[2]*target_shape[2])
    shape_of_target_volume = (int(target_volume[0]/resolution[0]), int(target_volume[1]/resolution[1]), int(target_volume[2]/resolution[2]))

    x = max(shape_of_target_volume[0], original_shape[0]) + 2
    y = max(shape_of_target_volume[1], original_shape[1]) + 2
    z = max(shape_of_target_volume[2], original_shape[2]) + 2

    x_start = int(x/2)-int(original_shape[0]/2)
    x_end = x_start + original_shape[0]
    y_start = int(y/2)-int(original_shape[1]/2)
    y_end = y_start + original_shape[1]
    z_start = int(z / 2) - int(original_shape[2] / 2)
    z_end = z_start + original_shape[2]

    array_intermediate = np.zeros((x, y, z), 'float32')
    array_intermediate[x_start:x_end, y_start:y_end, z_start:z_end] = array

    x_start = int(x / 2) - int(shape_of_target_volume[0] / 2)
    x_end = x_start + shape_of_target_volume[0]
    y_start = int(y / 2) - int(shape_of_target_volume[1] / 2)
    y_end = y_start + shape_of_target_volume[1]
    z_start = int(z / 2) - int(shape_of_target_volume[2] / 2)
    z_end = z_start + shape_of_target_volume[2]

    array_intermediate = array_intermediate[x_start:x_end, y_start:y_end, z_start:z_end]  # Now the array is padded

    # rescaling:
    array_standard_xy = np.zeros((target_shape[0], target_shape[1], shape_of_target_volume[2]), 'float32')
    for s in range(shape_of_target_volume[2]):
        array_standard_xy[:, :, s] = cv2.resize(array_intermediate[:, :, s], (target_shape[0], target_shape[1]), cv2.INTER_LANCZOS4)

    array_standard = np.zeros(target_shape, 'float32')
    for s in range(target_shape[0]):
        array_standard[s, :, :] = cv2.resize(array_standard_xy[s, :, :], (target_shape[2], target_shape[1]), cv2.INTER_LINEAR)

    return array_standard
