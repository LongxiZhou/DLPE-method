## Usage

### Install Dependencies



```
conda create -n 3dunet --file requirements.txt
```

### Pre-processing (transfer npy to h5)

```
python pre_process.py
```


### Train 3D Unet 

```
python pytorch3dunet/train.py --config ../resources/train_config_ce.yaml

```
The model will be saved to `pytorch3dunet/3dunet`.

### Test 3D Unet

```
python pytorch3dunet/predict.py --config ../resources/test_config_ce.yaml
```

The prediction will be saved to the folder containing test sets.

### Train 3D Vnet

```
python pytorch3dunet/train.py --config ../resources/train_config_dice.yaml
```
The model will be saved to `pytorch3dunet/3dunet`.

### Test 3D Vnet

```
python pytorch3dunet/predict.py --config ../resources/test_config_dice.yaml
```

### Post-processing (transfer h5 to npz for further evaluation)

```
python post_process.py
```

The configuration files (*.yaml) will be uploaded when the paper is published.

The codes are modified based on [/wolny/pytorch-3dunet](https://github.com/wolny/pytorch-3dunet).