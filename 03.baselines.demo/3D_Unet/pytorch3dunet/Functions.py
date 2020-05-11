import csv
import numpy as np
import h5py
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import io
import os
import random
import nibabel as nib
import cv2
import pydicom
import SimpleITK as sitk


def get_father_dict():
    return os.path.abspath(os.path.join(os.getcwd(), '..'))


def get_current_dict():
    return os.getcwd()


def get_patient_id(show=False):
    top_dic = os.path.abspath(os.path.join(os.getcwd(), '..'))
    already_checked = open(top_dic + '/check_format/id_already_checked', 'r')
    lines = already_checked.readlines()
    id_list = []
    for line in lines:
        id_list.append(line[0:12])
    if show:
        print('we have', len(id_list), 'patients, they are:')
        print(id_list)
    return id_list


def patient_id_have_dynamic_parameters():
    top_dic = os.path.abspath(os.path.join(os.getcwd(), '..'))
    return os.listdir(top_dic + '/dynamic/')


def load_dicom(path, show=False):
    # return a numpy array of the dicom file, and the slice number
    if show:
        content = pydicom.read_file(path)
        print(content)

    ds = sitk.ReadImage(path)

    img_array = sitk.GetArrayFromImage(ds)

    #  frame_num, width, height = img_array.shape

    return img_array[0, :, :], pydicom.read_file(path)['InstanceNumber'].value


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


def id_and_time(array_name):
    patient_id = array_name[0:12]
    collected_time = array_name[13:-4]
    return patient_id, collected_time


def load_nii(path):
    # return a numpy array of this .nii or .nii.gz file
    return nib.load(path).get_data()


def save_np_as_nii_gz(array, diction, file_name):
    # save a numpy array as .nii.gz or .nii file
    # e.g. diction = '/home/Desktop/test/' file_name = 'hello.nii.gz', then we have a '/home/Desktop/test/hello.nii.gz'
    # note: np.array_equal(load_nii(diction+file_name), array) is True
    if not os.path.exists(diction):
        os.makedirs(diction)
    nii_file = nib.Nifti1Image(array, np.eye(4))
    nii_file.to_filename(os.path.join(diction, '%s' % file_name))


def rescale_image(input_image, new_shape_tuple):
    # the input image will be rescale into [-1, 1]: negative value divide min, positive value divide max, 0 remains untouched
    # the shape will be rescaled according to the rescale factors
    # note, new_shape = np.shape(rescaled_image).reverse for two dim picture
    if np.max(input_image) > 1: # this means the input is data
        value_normalize = np.array(input_image, 'float32')
    else: # this means the input is mask
        value_normalize = np.array(input_image, 'float32')
    shape_normalize = cv2.resize(value_normalize, new_shape_tuple, cv2.INTER_AREA)
    return shape_normalize


def save_np_array(dict, file_name, np_array, compress=False):
    # if the dict not exist, we make the dict
    if not os.path.exists(dict):
        os.makedirs(dict)
    if not compress:
        np.save(dict + file_name, np_array)
    else:
        np.savez_compressed(dict + file_name, array=np_array)


def f1_sore_for_binary_mask(prediction, ground_truth):
    prediction = np.array(prediction > 0.5, 'float32')
    ground_truth = np.array(ground_truth > 0.5, 'float32')
    over_lap = np.sum(prediction * ground_truth)
    return 2 * over_lap / (np.sum(prediction) + np.sum(ground_truth))


def convert_to_final_pic(original, mask_rim):
    #original is a [448, 448] array that is the original MRI image
    #mask_rim is the rim of the mask that could circle the tumor
    #returns a [448, 448, 3] array that combines the mask_rim and the original picture

    return_array = np.zeros([448, 448, 3], dtype='int32')
    return_array[:,:,0] = original
    return_array[:,:,1] = original
    return_array[:,:,2] = original

    max = np.max(original)

    for i in range(448):
        for j in range(448):
            if mask_rim[i, j] != 0:
                return_array[i, j, 0] = max
                return_array[i, j, 1] = 0
                return_array[i, j, 2] = 0

    return return_array


def get_rim(mask):
    #mask is the mask file wich is a [a, b] np array
    #return the rim of the input np array

    a = np.shape(mask)[0]
    b = np.shape(mask)[1]

    return_array = np.zeros([a, b], dtype='int32')

    for i in range(1, a-1):
        for j in range(1, b-1):
            if mask[i, j] != 0:
                if mask[i-1, j] == 0:
                    return_array[i,j] = 1
                    return_array[i-1,j] = 1
                if mask[i+1, j] == 0:
                    return_array[i,j] = 1
                    return_array[i+1,j] = 1
                if mask[i, j-1] == 0:
                    return_array[i,j] = 1
                    return_array[i,j-1] = 1
                if mask[i, j+1] == 0:
                    return_array[i,j] = 1
                    return_array[i,j+1] = 1

    return return_array


def image_show(picture, gray=False):
    if not gray:
        plt.imshow(picture)
        plt.show()
        return 0
    picture = picture - np.min(picture)
    picture = picture/(np.max(picture)+0.00000001)
    s = np.shape(picture)
    image = np.zeros([s[0], s[1], 3], 'float32')
    image[:, :, 0] = picture
    image[:, :, 1] = picture
    image[:, :, 2] = picture
    image_show(image, False)


def merge_two_picture(picture, mask, normalize=True):

    # picture is a 2-d array, mask is also a 2-d array
    if normalize:
        picture -= np.min(picture)
        picture_normalized = cast_to_0_1(picture)
        mask_normalized = cast_to_0_1(mask)
    else:
        picture_normalized = picture
        mask_normalized = mask
    a = np.shape(picture)[0]
    b = np.shape(picture)[1]
    output = np.zeros([a, b, 3], 'float32')
    output[:,:,0] = picture_normalized
    output[:,:,1] = picture_normalized
    output[:,:,2] = picture_normalized
    #image_show(mask_normalized)
    for x in range(a):
        for y in range(b):
            if mask_normalized[x,y] > 0:
                output[x,y,1] = 0
                output[x,y,2] = 0
                output[x,y,0] = 1
    return output


def merge_image_with_mask(image, mask_image, save_path=None, normalize=True):

    temp = np.array(image)

    rim = get_rim(mask_image)

    temp = merge_two_picture(temp, rim, normalize)

    try:
        image_save(temp, save_path)
    except:
        image_show(temp)

    return temp


def image_save(picture, path, gray=False):
    if not gray:
        plt.cla()
        plt.imshow(picture)
        plt.savefig(path)
    else:
        gray_img = np.zeros([np.shape(picture)[0], np.shape(picture)[1], 3], 'float32')
        gray_img[:, :, 0] = picture
        gray_img[:, :, 1] = picture
        gray_img[:, :, 2] = picture
        plt.cla()
        plt.imshow(gray_img)
        plt.savefig(path)


def cast_to_0_1(input_array):
    # rescale the input array into range (0, 1)
    max_value = np.max(input_array)
    min_value = np.min(input_array)
    out_array = np.array((input_array + min_value) * 1.0, 'float32')
    out_array = out_array/(max_value - min_value)
    return out_array


def cast_to_upper_and_lower(input_array, upper_bound=100, lower_bound=-100, data_type='float32'):
    # divide the data into two parts: positive and negative
    # linearly rescale the positive value to upper_bound, and negative value to lower_bound
    # Because 0 is our universal bench mark, so upper_bound must no less than 0, and lower_bound must no greater than 0.
    # max for the input array should no less than 0 and min should no greater than 0.
    if upper_bound < 0 or lower_bound > 0:
        print('bound value error, upper_bound must no less than 0, and lower_bound must no greater than 0.')
        exit()
    max_value = np.max(input_array)
    min_value = np.min(input_array)
    positive_rescale_factor = 0
    negative_rescale_factor = 0
    print("min = ", min_value, "max = ", max_value)
    if max_value > 0:
        positive_rescale_factor = upper_bound / max_value
    if min_value < 0:
        negative_rescale_factor = lower_bound / min_value

    output_array = np.zeros(np.shape(input_array), data_type)
    positive_array_of_input = np.array((input_array > 0) * input_array, data_type)
    negative_array_of_input = np.array((input_array < 0) * input_array, data_type)

    output_array += np.array(positive_array_of_input * positive_rescale_factor, data_type)
    output_array += np.array(negative_array_of_input * negative_rescale_factor, data_type)

    return output_array


def normalization(Four_D_array):
    non_zeros = Four_D_array > 10
    Four_D_array = Four_D_array * non_zeros
    non_zero_count = np.sum(non_zeros)
    adjust_mean = np.sum(Four_D_array)/non_zero_count
    Difference = (Four_D_array - adjust_mean) * non_zeros
    Difference = Difference * Difference
    adjust_var = np.sum(Difference)/non_zero_count
    adjust_std = math.sqrt(adjust_var)

    max = np.max(Four_D_array)

    List = list(np.reshape(Four_D_array, [-1,]))
    List.sort(reverse=True)
    print(List[0:100])
    adjust_median = List[int(non_zero_count/2)]
    print(adjust_median)
    print(non_zero_count)
    print(adjust_mean)
    print(adjust_std)
    print(max, max/adjust_mean, max/math.sqrt(adjust_var), max/adjust_median)


def TP_TN_FP_FN(predict, ground_truth, data, swap=True):
    # please input 2D image
    if np.min(predict) < 0:
        print('value error, predictions must > 0')
        return None
    predict = predict/(np.max(predict)+0.00000001)
    predict = predict > 0.5
    predict = np.array(predict, 'float32')
    shape = np.shape(predict)
    out_image = np.zeros([shape[0], shape[1], 3], 'float32')
    out_image[:, :, 0] = data
    out_image[:, :, 1] = data
    out_image[:, :, 2] = data

    for x in range(shape[0]):
        for y in range(shape[1]):
            if predict[x, y] == ground_truth[x, y] and predict[x, y] == 1:   # TP
                out_image[x, y, 0] = 0
                out_image[x, y, 2] = 0
                out_image[x, y, 1] = 1
            if not predict[x, y] == ground_truth[x, y] and ground_truth[x, y] == 1: # FN
                out_image[x, y, 1] = 0
                out_image[x, y, 2] = 0
                out_image[x, y, 0] = 1
            if not predict[x, y] == ground_truth[x, y] and ground_truth[x, y] == 0: # FP
                out_image[x, y, 1] = 0
                out_image[x, y, 0] = 0
                out_image[x, y, 2] = 1
    if swap:
        return np.swapaxes(out_image, 0, 1)
    return out_image


def signal_distribution(input_array, max_value, min_value):

    distribution = np.zeros([max_value - min_value + 1], 'int32')

    for x in range(min_value, max_value + 1):
        distribution[x - min_value] = np.sum((input_array >= x) & (input_array < (x + 1)))

    return distribution


def load_picture(path):  # read image and linearly cast the value into [0, 100)
    img = mpimg.imread(path)

    max_value = np.max(img)
    #image_show(np.array(Sigmoid(img,20,max_value),'int32'))
    min_value = np.min(img)
    img = img - min_value
    img = img * 99.99 / max_value
    return np.array(img, 'int32')


def poly_fit_and_show(x, y, order, save_path=None, show=True):

    z1 = np.polyfit(x, y, order)
    p1 = np.poly1d(z1)
    print(p1)  # 在屏幕上打印拟合多项式
    if show:
        yvals = p1(x)  # 也可以使用yvals=np.polyval(z1,x)
        plot1 = plt.plot(x, y, '*', label='original values')
        plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.legend(loc=4)  # 指定legend的位置
        plt.title('polyfitting')
        if save_path is None:
            plt.show()
        else:
            # plt.imshow()
            plt.savefig(save_path)
    return z1


def show_data_points(x, y, save_path=None, data_label='data points'):

    if data_label is not None:
        plot1 = plt.plot(x, y, 'r-', label='data points')
        plot1 = plt.plot(x, y, '*', label='data points')
    else:
        plot1 = plt.plot(x, y, 'r-')
        plot1 = plt.plot(x, y, '*')

    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.legend(loc=4)  # 指定legend的位置,读者可以自己help它的用法
    plt.title('scatter graph')
    if save_path is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(save_path)
        plt.close()


def value_of_polynomial_log_exp(poly_list, value):
    length = len(poly_list)
    degree = length - 1
    value_poly = 0
    for i in range(length):
        value_poly += poly_list[i] * pow(value, degree)
        degree = degree - 1
    if value_poly <= 0:
        value_poly = np.exp(value_poly)
    else:
        value_poly = np.log(np.log(value_poly))
    return value_poly


def value_of_polynomial(poly_list, value):
    length = len(poly_list)
    degree = length - 1
    value_poly = 0
    for i in range(length):
        value_poly += poly_list[i] * pow(value, degree)
        degree = degree - 1
    return value_poly


def value_of_this_strange_data(weight_list, value):
    # the model is f(x)sin(g(x)), f, g are two polynomial function with poly_list of weight_list[0::2] and [1::2]
    return 3*(math.exp(value_of_polynomial([0.2]+weight_list[0::], value))+1/7)*math.sin(3.1415926*2*value + 0.07)
    #return 3*(math.exp(value_of_polynomial(weight_list[2::], value))+weight_list[1])*math.sin(3.1415926*2*value + weight_list[0])
    #return value_of_polynomial(weight_list[::2], value) * math.sin(10*value_of_polynomial(weight_list[1::2], value))


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
        args[i] -= 2*h
        behind = func(args)
        args[i] += h
        return_list.append((ahead - behind)/2/h)

    return np.array(return_list, 'float32')


def show_two_data_sets(x1, y1, x2, y2, save_path=None, show=True, data_label=True):

    if show:
        if data_label:
            plot1 = plt.plot(x1, y1, '.', label='original values')
            plot2 = plt.plot(x2, y2, '*', label='predicted values', color='#FFA500')
        else:
            plot1 = plt.plot(x1, y1, '.')
            plot2 = plt.plot(x2, y2, '*', color='#FFA500')
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.legend(loc=4)  # 指定legend的位置,读者可以自己help它的用法
        plt.title('scatter graph')
        if save_path is None:
            plt.show()
        else:
            # plt.imshow()
            plt.savefig(save_path)
            plt.close()


def shuffle_array(input_array):
    slices = np.shape(input_array)[0]
    all_indices = np.arange(0, slices, 1)
    random.shuffle(all_indices)
    return_array = input_array[all_indices, :, :, :]
    return return_array


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
        image_show(picture)
    return picture
