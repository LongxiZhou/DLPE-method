import numpy as np
import matplotlib.pyplot as plt
import os
import pydicom
import SimpleITK as sitk


def get_father_dict():
    return os.path.abspath(os.path.join(os.getcwd(), '..'))


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


def recall(prediction, ground_truth):
    prediction = np.array(prediction > 0.5, 'float32')
    ground_truth = np.array(ground_truth > 0.5, 'float32')
    over_lap = np.sum(prediction * ground_truth)
    total_positive = np.sum(ground_truth)
    return over_lap / total_positive


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


def image_save(picture, path, gray=False):
    if not gray:
        plt.cla()
        plt.axis('off')
        plt.imshow(picture)
        plt.savefig(path, pad_inches=0.0, bbox_inches='tight')
    else:
        gray_img = np.zeros([np.shape(picture)[0], np.shape(picture)[1], 3], 'float32')
        gray_img[:, :, 0] = picture
        gray_img[:, :, 1] = picture
        gray_img[:, :, 2] = picture
        plt.cla()
        plt.imshow(gray_img)
        plt.savefig(path)
