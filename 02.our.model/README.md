# Our Model

This is a version of the model for automatically infection region segmentation.
Although it can process all CT lung scan, if the CT scan has the following parameters, it will perform better: 
```
Slice Thickness < 1.5; Pixel Spacing < 1
```
1. Go into `./02.our.model/codes`; make sure the `model_infection` and `model_lung` is placed under this directory. 
2. run prediction_and_visualization.py
The program will do the preprocessing for the `.dcm` files and change it to the `.npy` arrays. The program segments the infection and lung and combines them together.
3. please go to `./02.our.model/standard/patient_id/time_1/` there is a `time_1_data.npy` which is the array after spatial and signal normalization.

## Additional Requirements
We need four more packages in addition to the environment specified in `01.introductory.demo`:
```
pip install bintrees
pip install pydicom
pip install SimpleITK
pip install opencv-python
```

## FAQ

### Where is the prediction?

we give two types of predictions. The first is the prediction of the original shape; the other is the prediction of the normalized shape.

the prediction of the original shape is stored in `./intermediate_arrays/patient_id/collected_time/`
under this dict, there are two files, the first is the data_array of simply stack DCM files. And the other is the predicted mask of this data_array.
see read_in_CT.py, line 37 to line 40 for details about the stacking.

the prediction of the normalized shape is stored in `./standard/patient_id/collected_time/`
under this dict, there are two files, the first is the data_array of stack DCM files then normalize the signal and the shape. And the other is the prediction of this normalized data_array.
see function: rescale_to_standard in generate_raw_arrays.py for details of shape normalization
see line 48 & 49 in `predict_dcm.py` for details of signal normalization


### How can I visualize the prediction results?
The program will automatically visualize all infected regions in the CT scan.
In `predict_dcm.py`, there is a function: visualize(patient_id, collected_time)

```
run visualize(patient_id, collected_time)
```
then open `./visualization/patient_id/collected_time/` to see the predictions (only slices with infections are showed!!!).

### How can I repredict a CT scan?
You just delete any file in `./standard/patient_id/collected_time/`
then rerun run `prediction_and_visualization.py`.


### How to output infection probabilities? 

Change line 129 of `U_net_predict.py`, from `>threshold` to `/3`.

