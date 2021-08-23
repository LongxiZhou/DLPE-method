import csv
import numpy as np
import h5py
import pickle
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import io
from scipy.stats import norm
from medpy import io
import imageio
import os
import random
import nibabel as nib
import cv2
import pydicom
import SimpleITK as sitk

np.set_printoptions(suppress=True)


def get_father_dict():
    return os.path.abspath(os.path.join(os.getcwd(), '..'))


def load_dicom(path, show=False):
    # return a numpy array of the dicom file, and the slice number
    if show:
        content = pydicom.read_file(path)
        print(content)
        # print(content['ContentDate'])

    ds = sitk.ReadImage(path)

    img_array = sitk.GetArrayFromImage(ds)

    #  frame_num, width, height = img_array.shape

    return img_array[0, :, :], pydicom.read_file(path)['InstanceNumber'].value


def get_dicom_resolution(path):
    first_content = pydicom.read_file(path)
    resolutions = first_content.PixelSpacing
    resolutions.append(first_content.SliceThickness)
    return resolutions


def dicom_and_prediction(prediction, dicom_image, ww_wc=(1600, -600)):
    dicom_image = dicom_image - ww_wc[1]
    dicom_image = dicom_image / ww_wc[0]
    dicom_image = dicom_image + 0.5

    shape = np.shape(dicom_image)
    merge = np.zeros((shape[0], shape[1] * 2, 3), 'float32')
    merge[:, 0: shape[1], 0] = dicom_image
    merge[:, 0: shape[1], 1] = dicom_image
    merge[:, 0: shape[1], 2] = dicom_image
    merge[:, shape[1]::, 0] = dicom_image
    merge[:, shape[1]::, 1] = dicom_image - prediction
    merge[:, shape[1]::, 2] = dicom_image - prediction
    return np.clip(merge, 0, 1)


def convert_png_to_np_array(file_path):
    return imageio.imread(file_path)


def extract_wc_ww(value):
    try:
        return int(value)
    except:
        try:
            return int(value[0])
        except:
            print('wc_ww strange')
            exit(1)


def wc_ww(path):
    info = pydicom.read_file(path)
    wc = info['WindowCenter'].value
    ww = info['WindowWidth'].value
    return extract_wc_ww(wc), extract_wc_ww(ww)


def array_stat(array):
    print('array has shape:', np.shape(array))
    print('min-average-max:', np.min(array), np.average(array), np.max(array))
    print('std:', np.std(array))


def load_nii(path):
    # return a numpy array of this .nii or .nii.gz file
    return nib.load(path).get_data()


def pickle_save_object(save_path, object_being_save):
    """
    :param save_path: like /home/zhoul0a/Desktop/hospitalize_data_dict.pickle
    :param object_being_save: like a dictionary
    :return: None
    """
    with open(save_path, 'wb') as handle:
        pickle.dump(object_being_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load_object(save_path):
    with open(save_path, 'rb') as handle:
        return pickle.load(handle)


def save_np_as_nii_gz(array, diction, file_name):
    # save a numpy array as .nii.gz or .nii file
    # e.g. diction = '/home/Desktop/test/' file_name = 'hello.nii.gz', then we have a '/home/Desktop/test/hello.nii.gz'
    # note: np.array_equal(load_nii(diction+file_name), array) is True
    if not os.path.exists(diction):
        os.makedirs(diction)
    nii_file = nib.Nifti1Image(array, np.eye(4))
    nii_file.to_filename(os.path.join(diction, '%s' % file_name))


def save_np_as_mha(np_array, save_dict, file_name):
    if not os.path.exists(save_dict):
        os.makedirs(save_dict)

    if file_name[-4::] == '.mha':
        file_name = file_name[:-4]

    np_array = np.transpose(np_array, (1, 0, 2))
    np_array[np_array < 0.5] = 0
    np_array[np_array >= 0.5] = 1
    np_array = np_array.astype("uint8")
    header = io.Header(spacing=(1, 1, 1))
    print("mha file path:", os.path.join(save_dict, file_name) + '.mha')
    io.save(np_array, os.path.join(save_dict, file_name) + '.mha', hdr=header, use_compression=True)


def rescale_2d_array(input_array, new_shape_tuple):
    """
    :param input_array: a 2d image array with float32
    :param new_shape_tuple: the output shape, i.e., np.shape(return array)
    :return: the shape normalized array
    """
    assert len(np.shape(input_array)) == 2 and len(new_shape_tuple) == 2
    shape_normalize = cv2.resize(input_array, (new_shape_tuple[1], new_shape_tuple[0]), cv2.INTER_AREA)
    return shape_normalize


def save_np_array(save_dict, file_name, np_array, compress=False):
    # if the save_dict not exist, we make the dict
    if not save_dict[-1] == '/':
        save_dict = save_dict + '/'
    if not os.path.exists(save_dict):
        os.makedirs(save_dict)
    if not compress:
        np.save(save_dict + file_name, np_array)
    else:
        np.savez_compressed(save_dict + file_name, array=np_array)


def f1_sore_for_binary_mask(prediction, ground_truth, threshold=0.5):
    prediction = np.array(prediction > threshold, 'float32')
    ground_truth = np.array(ground_truth > threshold, 'float32')
    over_lap = np.sum(prediction * ground_truth)
    return 2 * over_lap / (np.sum(prediction) + np.sum(ground_truth))


def get_rim(mask):
    # mask is the mask file which is a [a, b] np array
    # return the rim of the input np array

    a = np.shape(mask)[0]
    b = np.shape(mask)[1]

    return_array = np.zeros([a, b], dtype='int32')

    for i in range(1, a - 1):
        for j in range(1, b - 1):
            if mask[i, j] != 0:
                if mask[i - 1, j] == 0:
                    return_array[i, j] = 1
                    return_array[i - 1, j] = 1
                if mask[i + 1, j] == 0:
                    return_array[i, j] = 1
                    return_array[i + 1, j] = 1
                if mask[i, j - 1] == 0:
                    return_array[i, j] = 1
                    return_array[i, j - 1] = 1
                if mask[i, j + 1] == 0:
                    return_array[i, j] = 1
                    return_array[i, j + 1] = 1

    return return_array


def image_show(picture_in, gray=False):
    if not gray:
        plt.imshow(picture_in)
        plt.show()
        return picture_in
    picture = np.array(picture_in, 'float32')
    picture = picture - np.min(picture)
    picture = picture / (np.max(picture) + 0.00000001)
    s = np.shape(picture)
    image = np.zeros([s[0], s[1], 3], 'float32')
    image[:, :, 0] = picture
    image[:, :, 1] = picture
    image[:, :, 2] = picture
    image_show(image, False)
    return image


def merge_two_picture(picture, mask):
    # picture is a 2-d array, mask is also a 2-d array
    picture = cast_to_0_1(picture)
    mask = cast_to_0_1(mask)

    a = np.shape(picture)[0]
    b = np.shape(picture)[1]
    assert np.shape(picture) == np.shape(mask)
    output = np.zeros([a, b * 2, 3], 'float32')
    output[:, 0:b, 0] = picture
    output[:, 0:b, 1] = picture
    output[:, 0:b, 2] = picture
    output[:, b::, 0] = picture + mask
    output[:, b::, 1] = picture - mask
    output[:, b::, 2] = picture - mask
    output = np.clip(output, 0, 1)
    return output


def merge_image_with_mask(image, mask_image, convert_to_rim=False, save_path=None, show=True):
    temp = np.array(image)

    if convert_to_rim:
        rim_array = get_rim(mask_image)
        temp = merge_two_picture(temp, rim_array)
    else:
        temp = merge_two_picture(temp, mask_image)

    try:
        image_save(temp, save_path, high_resolution=True)
    except:
        if show:
            image_show(temp)

    return temp


def image_save(picture, path, gray=False, high_resolution=False, dpi=None):
    save_dict = path[:-len(path.split('/')[-1])]
    if not os.path.exists(save_dict):
        os.makedirs(save_dict)
    picture = linear_value_change(picture, 0, 1)
    if not gray:
        plt.cla()
        plt.axis('off')
        plt.imshow(picture)
        if dpi is not None:
            plt.savefig(path, pad_inches=0.0, bbox_inches='tight', dpi=dpi)
            return None
        if high_resolution:
            plt.savefig(path, pad_inches=0.0, bbox_inches='tight', dpi=600)
        else:
            plt.savefig(path, pad_inches=0.0, bbox_inches='tight')
    else:
        gray_img = np.zeros([np.shape(picture)[0], np.shape(picture)[1], 3], 'float32')
        gray_img[:, :, 0] = picture
        gray_img[:, :, 1] = picture
        gray_img[:, :, 2] = picture
        if dpi is not None:
            plt.savefig(path, pad_inches=0.0, bbox_inches='tight', dpi=dpi)
            return None
        if high_resolution:
            plt.cla()
            plt.axis('off')
            plt.imshow(gray_img)
            plt.savefig(path, pad_inches=0.0, bbox_inches='tight', dpi=600)
        else:
            plt.cla()
            plt.imshow(gray_img)
            plt.savefig(path)
    return None


def cast_to_0_1(input_array):
    # rescale the input array into range (0, 1)
    max_value = np.max(input_array)
    min_value = np.min(input_array)
    out_array = np.array((input_array - min_value) * 1.0, 'float32')
    out_array = out_array / (max_value - min_value + 0.00001)
    return out_array


def signal_distribution(input_array, max_value, min_value):
    distribution = np.zeros([max_value - min_value + 1], 'int32')

    for x in range(min_value, max_value + 1):
        distribution[x - min_value] = np.sum((input_array >= x) & (input_array < (x + 1)))

    return distribution


def show_data_points(x, y, save_path=None, data_label='data points', x_name='x_axis', y_name='y_axis', title='scatter'):
    plt.close()

    if data_label is not None:
        # plot1 = plt.plot(x, y, 'r', label='data points')
        plot1 = plt.plot(x, y, '*', label='data points')
    else:
        # plot1 = plt.plot(x, y, 'r')
        plot1 = plt.plot(x, y, '*')

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend(loc=4)  # 指定legend的位置,读者可以自己help它的用法
    plt.title(title)
    if save_path is None:
        plt.show()
        plt.close()
    else:
        plt.rcParams['savefig.dpi'] = 600
        plt.rcParams['figure.dpi'] = 600
        plt.savefig(save_path)
        plt.close()


def derivative(func, args, precision=0.000000000001):
    # func = func(args)
    # returns a array of d(func)/d(args) at given args, with error = O(precision)
    # if the third order derivative = 0, then error = 0
    h = math.sqrt(abs(precision))
    num_args = len(args)
    return_list = []
    for i in range(num_args):
        if i > 4:
            return_list.append(0)
            continue
        args[i] += h
        ahead = func(args)
        args[i] -= 2 * h
        behind = func(args)
        args[i] += h
        return_list.append((ahead - behind) / 2 / h)

    return np.array(return_list, 'float32')


def shuffle_array(input_array):
    slices = np.shape(input_array)[0]
    all_indices = np.arange(0, slices, 1)
    random.shuffle(all_indices)
    return_array = input_array[all_indices, :, :, :]
    return return_array


def linear_fit(x, y, show=True):
    N = float(len(x))
    sx, sy, sxx, syy, sxy = 0, 0, 0, 0, 0
    for i in range(0, int(N)):
        sx += x[i]
        sy += y[i]
        sxx += x[i] * x[i]
        syy += y[i] * y[i]
        sxy += x[i] * y[i]
    a = (sy * sx / N - sxy) / (sx * sx / N - sxx)
    b = (sy - a * sx) / N
    r = (sy * sx / N - sxy) / math.sqrt((sxx - sx * sx / N) * (syy - sy * sy / N))
    if show:
        print("the fitting result is: y = %10.5f x + %10.5f , r = %10.5f" % (a, b, r))
    return a, b, r


def scale_free_check(scale_list, frequency, cache=10, show=True):
    # scale_list is a ordered list recording the measurements, like area, degree, etc
    # frequency is a list recording the frequency or probability of each scale
    scale_list = np.array(scale_list)
    frequency = np.array(frequency)
    length = len(scale_list)
    assert len(scale_list) == len(frequency)
    if show:
        print("the length of the list is", length)
    step = round(length / cache)

    def get_center(sub_list_scale, sub_list_frequency):
        return sum(sub_list_scale * sub_list_frequency) / sum(sub_list_frequency)

    center_list = []
    total_frequency_list = []
    for loc in range(0, length, step):
        if loc + step >= length:
            end = length
        else:
            end = loc + step
        list_cache_scale = scale_list[loc: end]
        list_cache_frequency = frequency[loc: end]
        center_list.append(get_center(list_cache_scale, list_cache_frequency))
        total_frequency = np.sum(list_cache_frequency)
        if total_frequency == 0:
            print("detect 0 frequency, replace with 1")
            total_frequency = 1
        total_frequency_list.append(total_frequency)
    if show:
        show_data_points(np.log(center_list), np.log(total_frequency_list))
    return linear_fit(np.log(center_list), np.log(total_frequency_list))


def linear_value_change(array, min_value, max_value, data_type='float32'):
    # linearly cast to [min_value, max_value]
    max_original = np.max(array) + 0.000001
    min_original = np.min(array)
    assert max_value > min_value
    assert max_original > min_original
    return_array = np.array(array, data_type)
    return_array -= min_original
    return_array = return_array / ((max_original - min_original) * (max_value - min_value)) + min_value
    return return_array


def sigmoid(array, a, b):
    # linearly cast to [0, 1], sigmoid, then linearly cast to [min(array), max(array)]
    min_original = np.min(array)
    max_original = np.max(array)
    return_array = linear_value_change(array, 0, 1)  # cast to [0, 1]
    assert a > 0 and b > 0
    return_array = 1 / (1 + a * np.exp(-b * return_array))  # sigmoid
    return_array = linear_value_change(return_array, min_original, max_original)  # cast to [min(array), max(array)]
    return return_array


def func_parallel(func, list_inputs, leave_cpu_num=1):
    """
    :param func: func(list_inputs[i])
    :param list_inputs: each element is the input of func
    :param leave_cpu_num: num of cpu that not use
    :return: [return_of_func(list_inputs[0]), return_of_func(list_inputs[1]), ...]
    """
    import multiprocessing as mp
    cpu_cores = mp.cpu_count() - leave_cpu_num
    pool = mp.Pool(processes=cpu_cores)
    list_outputs = pool.map(func, list_inputs)
    pool.close()
    return list_outputs


def visualize_stl(stl_path):
    import vtk
    filename = stl_path

    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)

    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(reader.GetOutput())
    else:
        mapper.SetInputConnection(reader.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Create a rendering window and renderer
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    # Create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Assign actor to the renderer
    ren.AddActor(actor)

    # Enable user interface interactor
    iren.Initialize()
    renWin.Render()
    iren.Start()


def read_in_mha(path):
    ar = sitk.ReadImage(path)
    mask = sitk.GetArrayFromImage(ar)
    mask = np.swapaxes(mask, 0, 2)
    mask = np.swapaxes(mask, 0, 1)
    mask = np.array(mask > 0, 'int32')
    return mask  # (x, y, z)


def get_bounding_box(mask):
    # mask is a binary array
    # return a list [(x_min, x_max), (y_min, y_max), ...] which is the bounding box of each dimension
    bounding_box = []
    positive_locs = np.where(mask > 0.5)
    for loc_array in positive_locs:
        min_loc = np.min(loc_array)
        max_loc = np.max(loc_array)
        bounding_box.append((min_loc, max_loc))
    return bounding_box


def chi2_contigency_test(list_a, list_b, a_level=3, b_level=3):
    """
    we have two different variables: list_a, list_b. test whether they are independent.
    :param list_a: variable a
    :param list_b: variable b
    :param a_level: level of variable a
    :param b_level: level of variable b
    :return: p value
    """
    from scipy.stats import chi2_contingency

    length_a = len(list_a)
    length_b = len(list_b)
    assert length_a == length_b
    interval_a = round(length_a / a_level)
    interval_b = round(length_b / b_level)

    p_value_log = 0

    tested_num = 0
    potential_p = []

    for j in range(1000):
        list_a_new = list(list_a)
        list_b_new = list(list_b)
        for i in range(length_a):  # add a small noise to make every observation distinguishable
            list_a_new[i] = list_a[i] + random.random() / 10000000
            list_b_new[i] = list_b[i] + random.random() / 10000000
        sorted_a = list(list_a_new)
        sorted_a.sort()
        sorted_b = list(list_b_new)
        sorted_b.sort()

        contigency_array = np.zeros([a_level, b_level], 'int32')
        for i in range(length_a):  # patient i
            value_a = list_a_new[i]
            value_b = list_b_new[i]
            loc_a = sorted_a.index(value_a)
            loc_b = sorted_b.index(value_b)
            contigency_array[min(int(loc_a / interval_a), a_level - 1), min(int(loc_b / interval_b), b_level - 1)] += 1
        current_p_log = math.log(chi2_contingency(contigency_array)[1])
        p_value_log = p_value_log + current_p_log
        tested_num += 1
        if current_p_log not in potential_p:
            potential_p.append(p_value_log)
        if tested_num % 100 == 9:
            if np.std(potential_p) / tested_num < 0.01:
                # print("converged at", tested_num - 8)
                break

    p_value_log = p_value_log / tested_num

    return math.exp(p_value_log)


def geometric_mean(inputs_array):  # all inputs should greater than 0
    log_out = np.sum(np.log(inputs_array))
    shape = np.shape(inputs_array)
    total_count = 1
    for i in shape:
        total_count *= i
    return math.exp(log_out / total_count)


def dependency_test(list_a, list_b, a_level_trial=(2, 4), b_level_trial=(2, 4), single_value=False):
    a_max = min(a_level_trial[1], len(set(list_a)))
    a_min = min(a_level_trial[0], len(set(list_a)))
    b_max = min(b_level_trial[1], len(set(list_b)))
    b_min = min(b_level_trial[0], len(set(list_b)))
    p_array = np.zeros([a_max - a_min + 1, b_max - b_min + 1], 'float32')
    for a in range(a_min, a_max + 1):
        for b in range(b_min, b_max + 1):
            p_array[a - a_min, b - b_min] = chi2_contigency_test(list_a, list_b, a, b)
    if single_value:
        return geometric_mean(p_array)
    return p_array


def probability_binomial(n, m):
    if n < 100:
        return math.factorial(n) / math.factorial(m) / math.factorial(n - m) * math.pow(0.5, n)
    log_n_factorial = log_factorial(n)
    if m > 100:
        log_m_factorial = log_factorial(m)
    else:
        log_m_factorial = math.log(math.factorial(m))
    if n - m > 100:
        log_n_m_factorial = log_factorial(n - m)
    else:
        log_n_m_factorial = math.log(math.factorial(n - m))
    return math.exp(log_n_factorial + n * math.log(0.5) - log_m_factorial - log_n_m_factorial)


def log_factorial(n):
    return n * math.log(n) - n + 0.5 * math.log(
        2 * 3.1415926535897932384626433 * n) + 1 / 12 / n - 1 / 360 / n / n / n + 1 / 1260 / n / n / n / n / n - 1 / 1680 / n / n / n / n / n / n / n


def customized_sort(list_like, compare_func, reverse=False):
    """

    :param reverse:
    :param list_like: iterative object
    :param compare_func: takes two element, a, b as input, return -1 or 1.
    if a > b return 1 and reverse is False, the sort is Increasing.
    :return:
    """

    from functools import cmp_to_key
    list_like.sort(key=cmp_to_key(compare_func), reverse=reverse)
    return list_like


def stat_on_mask(reference_array, mask, remove_outliers=0.2):
    """
    stat on the given mask
    :param remove_outliers: e.g. removes largest 20% and smallest 20%
    :param reference_array: like a 3D CT data
    :param mask: like airway mask, binary value
    :return: value mean, std on of the reference_array value on mask
    """
    mask = np.array(mask > 0, 'float32')
    locations = np.where(mask > 0)
    num_voxels = len(locations[0])
    value_list = []
    for i in range(num_voxels):
        value_list.append(reference_array[locations[0][i], locations[1][i], locations[2][i]])
    value_list.sort()
    assert remove_outliers < 0.5
    value_list = value_list[int(num_voxels * remove_outliers): num_voxels - 2 - int(num_voxels * remove_outliers)]
    value_list = np.array(value_list)
    return np.median(value_list), np.std(value_list)


def rename_path(old_path, new_path):
    os.rename(old_path, new_path)


def split_dict_and_name(path):
    name = path.split('/')[-1]
    return path[0: len(path) - len(name)], name


def save_np_to_path(save_path, np_array):
    save_dict, file_name = split_dict_and_name(save_path)
    assert file_name[-4::] == '.npy' or file_name[-4::] == '.npz'
    if file_name[-1] == 'z':
        save_np_array(save_dict, file_name, np_array, compress=True)
    else:
        save_np_array(save_dict, file_name, np_array, compress=False)


def get_heat_map(cam_map, target_shape=None):
    # input a numpy array with shape (a, b)
    min_value, max_value = np.min(cam_map), np.max(cam_map)
    cam_map = (cam_map - min_value) / (max_value + 0.00001) * 255
    cam_map = np.array(cam_map, 'int32')
    if target_shape is not None:
        assert len(target_shape) == 2

        cam_map = cv2.resize(np.array(cam_map, 'float32'), target_shape)  # must in float to resize
    colored_cam = cv2.normalize(cam_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    colored_cam = cv2.applyColorMap(colored_cam, cv2.COLORMAP_JET)

    return_image = np.zeros(np.shape(colored_cam), 'int32')
    return_image[:, :, 0] = colored_cam[:, :, 2]
    return_image[:, :, 1] = colored_cam[:, :, 1]
    return_image[:, :, 2] = colored_cam[:, :, 0]

    return return_image / 255


def merge_with_heat_map(data_image, cam_map, signal_rescale=False):
    """

    :param signal_rescale: 0-1 rescale of data_image
    :param data_image: a numpy array with shape (a, b) or (a, b, 3)
    :param cam_map: a numpy array with shape (c, d)
    :return: merged image with shape (a, b, 3), in float32, min 0 max 1.0
    """
    shape_image = np.shape(data_image)
    if not shape_image == np.shape(cam_map):
        heat_map = get_heat_map(cam_map, target_shape=(shape_image[0], shape_image[1]))
    else:
        heat_map = get_heat_map(cam_map, target_shape=None)
    if signal_rescale:
        min_value, max_value = np.min(data_image), np.max(data_image)
        data_image = (data_image - min_value) / (max_value + 0.00001)
    cam_map = cv2.resize(np.array(cam_map, 'float32'), (shape_image[0], shape_image[1]))  # must in float to resize
    weight_map = cam_map / (np.max(cam_map) + 0.00001)
    weight_map_image = 1 - weight_map
    return_image = np.zeros((shape_image[0], shape_image[1] * 2, 3), 'float32')
    if len(shape_image) == 2:
        return_image[:, 0: shape_image[1], 0] = data_image
        return_image[:, 0: shape_image[1], 1] = data_image
        return_image[:, 0: shape_image[1], 2] = data_image
    else:
        return_image[:, 0: shape_image[1], :] = data_image

    return_image[:, shape_image[1]::, 0] = \
        weight_map_image * return_image[:, 0: shape_image[1], 0] + weight_map * heat_map[:, :, 0]
    return_image[:, shape_image[1]::, 1] = \
        weight_map_image * return_image[:, 0: shape_image[1], 1] + weight_map * heat_map[:, :, 1]
    return_image[:, shape_image[1]::, 2] = \
        weight_map_image * return_image[:, 0: shape_image[1], 2] + weight_map * heat_map[:, :, 2]
    return return_image


if __name__ == '__main__':
    fn_list = os.listdir('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/enhanced_arrays/')
    dict_name = 'stage_two_last_cnn_version4'
    for fn in fn_list:
        print(fn)
        data = np.load('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/visualization/cam_maps/' + dict_name + '/' + fn[
                                                                                                             :-4] + '_sample_Z.npy')[:, :, 1]
        data = np.swapaxes(data, 0, 1)
        heat_map_cam = np.load('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/visualization/cam_maps/' + dict_name + '/' + fn[
                                                                                                             :-4] + '_heatmap_Z.npy')
        heat_map_cam = np.swapaxes(heat_map_cam, 0, 1)
        final_image = merge_with_heat_map(data, heat_map_cam)

        image_save(final_image, '/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/visualization/picture_cam/' + dict_name + '/' + fn[:-4] + '.png', high_resolution=True)
