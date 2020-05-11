# Hospital Deployable Version

This is a version of the model for automatically infection region segmentation.
Although it can process all CT lung scan, if the CT scan has the following parameters, it will perform better: 
```
Slice Thickness < 1.5; Pixel Spacing < 1
```



If we have a new CT scan, in which there are hundreds of DCM or DICOM files, we have two steps to get the prediction:

1. give the CT scan two names: patient_id, and collected_time, then save it to dict: './patients/patient_id/collected_time/'

   A patient can have many scans.
for example, if the patient have id '1', and the collected_time is '03-28', then you just save these DCM or DICOM files to './patients/1/03-28/'.


2. run predict_all() in predict_dcm.py

   this program automatically check the unpredicted files, then generate the prediction. 
each scan needs about 30s to predict

## FAQ

### Where is the prediction?

we give two types of predictions. The first is the prediction of the original shape; the other is the prediction of the normalized shape.

the prediction of the original shape is stored in './intermediate_arrays/patient_id/collected_time/'
under this dict, there are two files, the first is the data_array of simply stack DCM files. And the other is the predicted mask of this data_array.
see read_in_CT.py, line 37 to line 40 for details about the stacking.

the prediction of the normalized shape is stored in './standard/patient_id/collected_time/'
under this dict, there are two files, the first is the data_array of stack DCM files then normalize the signal and the shape. And the other is the prediction of this normalized data_array.
see function: rescale_to_standard in generate_raw_arrays.py for details of shape normalization
see line 48 & 49 in predict_dcm.py for details of signal normalization


### How can I visualize the prediction results?

In predict_dcm.py, there is a function: visualize(patient_id, collected_time)

run visualize(patient_id, collected_time)

then open './visualization/patient_id/collected_time/' to see the predictions (only slices with infections are showed!!!).

How can I repredict a CT scan?
You just delete any file in './standard/patient_id/collected_time/' 
then rerun run predict_all() in predict_dcm.py





