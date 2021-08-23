"""
provide these functions. numpy is the standard format to process
dcm -> npy unrescaled
dcm -> npy signal rescaled
dcm -> npy spatial rescaled
dcm -> npy spatial and signal rescaled
mha -> npy
npy -> mha
npy spatial rescaled -> npy spatial unrescaled  (convert standard shape and resolution to original ones)
"""
import format_convert.read_in_CT as read_in_CT
from medpy import io
import SimpleITK as si
import numpy as np
import Tool_Functions.Functions as Functions
import format_convert.spatial_normalize as spatial_normalize
import pydicom
import os


def dcm_to_unrescaled(dcm_dict, save_path=None, show=True, return_resolution=False):
    """
    just stack dcm files together
    :param return_resolution:
    :param show:
    :param dcm_dict:
    :param save_path: the save path for stacked array
    :return: the stacked array in float32
    """
    array_stacked, resolution = read_in_CT.stack_dcm_files_simplest(dcm_dict, show=show)
    if save_path is not None:
        if show:
            print("save array to:", save_path)
        Functions.save_np_to_path(save_path, array_stacked)
    if return_resolution:
        return array_stacked, resolution
    return array_stacked


def dcm_to_signal_rescaled(dcm_dict, wc_ww=None, save_path=None, show=True):
    unrescaled_array = dcm_to_unrescaled(dcm_dict, save_path=None, show=show)
    if wc_ww is None:
        dcm_file_names = os.listdir(dcm_dict)
        wc, ww = Functions.wc_ww(os.path.join(dcm_dict, dcm_file_names[0]))
        if show:
            print("no wc_ww given, using default. wc:", wc, " ww:", ww)
    else:
        wc, ww = wc_ww
        if show:
            print("given wc_ww wc:", wc, " ww:", ww)
    signal_rescaled = (unrescaled_array - wc) / ww  # cast the wc_ww into -0.5 to 0.5
    if save_path is not None:
        if show:
            print("save array to:", save_path)
        Functions.save_np_to_path(save_path, signal_rescaled)
    return signal_rescaled


def dcm_to_spatial_rescaled(dcm_dict, target_resolution=(334/512, 334/512, 1), target_shape=(512, 512, 512),
                            save_path=None, show=True, tissue='lung'):
    unrescaled_array, resolution = dcm_to_unrescaled(dcm_dict, save_path=None, show=show, return_resolution=True)
    if tissue == 'lung':
        assert target_resolution == (334/512, 334/512, 1)
        assert target_shape == (512, 512, 512)
    spatial_rescaled = spatial_normalize.rescale_to_standard(unrescaled_array, resolution, target_resolution,
                                                             target_shape, tissue=tissue)
    if save_path is not None:
        if show:
            print("save array to:", save_path)
        Functions.save_np_to_path(save_path, spatial_rescaled)
    return spatial_rescaled


def dcm_to_spatial_signal_rescaled(dcm_dict, wc_ww=None, target_resolution=(334/512, 334/512, 1),
                                   target_shape=(512, 512, 512), tissue='lung', save_path=None, show=True):
    if tissue == 'lung':
        assert target_resolution == (334/512, 334/512, 1)
        assert target_shape == (512, 512, 512)
    if wc_ww is None:
        dcm_file_names = os.listdir(dcm_dict)
        wc, ww = Functions.wc_ww(os.path.join(dcm_dict, dcm_file_names[0]))
        if show:
            print("no wc_ww given, using default. wc:", wc, " ww:", ww)
    else:
        wc, ww = wc_ww
        if show:
            print("given wc_ww wc:", wc, " ww:", ww)
    spatial_rescaled = dcm_to_spatial_rescaled(dcm_dict, target_resolution, target_shape, None, show, tissue)
    spatial_signal_rescaled = (spatial_rescaled - wc) / ww
    if save_path is not None:
        if show:
            print("save array to:", save_path)
        Functions.save_np_to_path(save_path, spatial_signal_rescaled)
    return spatial_signal_rescaled


def read_in_mha(path):
    ar = si.ReadImage(path)
    mask = si.GetArrayFromImage(ar)
    mask = np.swapaxes(mask, 0, 2)
    mask = np.swapaxes(mask, 0, 1)
    mask = np.array(mask > 0.5, 'float32')
    return mask  # (x, y, z)


def save_np_as_mha(np_array, save_dict, file_name):
    # only for binary mask
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


def get_original_resolution(dcm_dict):
    dcm_file_names = os.listdir(dcm_dict)
    first_content = pydicom.read_file(os.path.join(dcm_dict, dcm_file_names[0]))
    resolution = first_content.PixelSpacing
    resolution.append(first_content.SliceThickness)
    return resolution


def undo_spatial_rescale(dcm_dict, spatial_rescaled_array, resolution_rescaled=(334/512, 334/512, 1), tissue='lung'):
    """
    align to the original dcm files, e.g. mask[:, :, slice_id] is for dcm file of slice_id
    :param dcm_dict:
    :param spatial_rescaled_array: the prediction is on the rescaled array
    :param resolution_rescaled: the resolution of the standard space
    :param tissue:
    :return: array that undo the spatial rescale
    """
    if tissue == 'lung':
        assert resolution_rescaled == (334/512, 334/512, 1)
    dcm_file_names = os.listdir(dcm_dict)
    num_slices = len(dcm_file_names)
    first_slice = Functions.load_dicom(os.path.join(dcm_dict, dcm_file_names[0]))[0]
    first_content = pydicom.read_file(os.path.join(dcm_dict, dcm_file_names[0]))
    resolution = first_content.PixelSpacing
    resolution.append(first_content.SliceThickness)
    rows, columns = first_slice.shape
    # the original shape should be [rows, columns, num_slices]
    original_shape = (rows, columns, num_slices)
    if tissue == 'lung' and original_shape[2] * resolution[2] > 450:
        resolution[2] = 450 / original_shape[2]
    return spatial_normalize.rescale_to_original(spatial_rescaled_array, resolution_rescaled, resolution,
                                                 original_shape)


if __name__ == '__main__':
    stacked = dcm_to_unrescaled("/home/zhoul0a/Desktop/pulmonary nodules/ct_and_gt/fjj-104/2020-03-10/Data/raw_data/",
                                None)
    Functions.image_show(stacked[:, :, 250])
