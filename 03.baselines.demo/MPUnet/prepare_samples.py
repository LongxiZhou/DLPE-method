# generate dataset

import nibabel as nib

import numpy as np

import Functions

import os

top_dict = Functions.get_current_dict()

patients = os.listdir(top_dict + '/arrays_raw')

for i in range(len(patients)):

    label_holder = np.zeros([512, 512, 512], 'float32')
    patient_id = patients[i]

    print('processing:', patient_id)

    patient = np.load(top_dict + '/arrays_raw/' + patient_id)  # ../unrescaled

    datas = np.array(patient[:, :, :])

    datas = (datas - np.min(datas)) * 255 / (np.max(datas) - np.min(datas))  # very important

    datas = np.stack((datas,) * 3, axis=-1)

    labels = label_holder

    datas = np.array(datas).astype(np.uint8)

    labels = np.array(labels).astype(np.uint8)

    imgs = nib.Nifti1Image(datas, np.eye(4))

    imgs.to_filename(os.path.join('test_dataset/images', '%s.nii.gz' % patient_id))

    # labels = nib.Nifti1Image(labels, np.eye(4))

    # labels.to_filename(os.path.join('test_dataset/labels', '%s.nii.gz' % patient_id))

    print(i + 1, '/', len(patients), 'Done!')
