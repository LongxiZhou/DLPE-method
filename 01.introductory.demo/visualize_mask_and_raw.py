import numpy as np
import os
import PIL


def visualize_mask_and_raw_array(save_dir,array_ct,pred_mask,gt_mask=None,direction='Z'):
    assert np.allclose(array_ct.shape,pred_mask.shape)
    if type(gt_mask) == np.ndarray:
        assert np.allclose(array_ct.shape,gt_mask.shape)
    
    array_cut = np.clip(array_ct, -0.5, 0.5) + 0.5

    X_size,Y_size,Z_size=array_ct.shape

    if type(gt_mask) == np.ndarray:
        merge = [np.zeros([X_size, Y_size , Z_size, 3], 'float32') for _ in range(4)]
        merge[0][:, :, :, 0] = array_cut
        merge[0][:, :, :, 1] = array_cut
        merge[0][:, :, :, 2] = array_cut

        merge[1][:, :, :, 0] = array_cut
        merge[1][:, :, :, 1] = array_cut - pred_mask
        merge[1][:, :, :, 2] = array_cut - pred_mask

        merge[2][:, :, :, 0] = array_cut
        merge[2][:, :, :, 1] = array_cut - gt_mask
        merge[2][:, :, :, 2] = array_cut - gt_mask

        TP=pred_mask*gt_mask
        FP=pred_mask*(1-gt_mask)
        FN=(1-pred_mask)*gt_mask

        merge[3][:, :, :, 0] = array_cut - TP - FP
        merge[3][:, :, :, 1] = array_cut - FN - FP
        merge[3][:, :, :, 2] = array_cut - TP - FN

    else:
        merge=[np.zeros([X_size, Y_size , Z_size, 3], 'float32') for _ in range(2)]
        merge[0][:, :, :, 0] = array_cut
        merge[0][:, :, :, 1] = array_cut
        merge[0][:, :, :, 2] = array_cut

        merge[1][:, :, :, 0] = array_cut
        merge[1][:, :, :, 1] = array_cut - pred_mask
        merge[1][:, :, :, 2] = array_cut - pred_mask

    merge = np.clip(merge, 0, 1)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # import ipdb; ipdb.set_trace()

    if direction=="X":
        for i in range(X_size):
            image_save(np.concatenate([merge[j][i, :, :] for j in range(len(merge))],axis=1), save_dir + str(i) +".png", gray=False)
    elif direction=="Y":
        for i in range(Y_size):
            image_save(np.concatenate([merge[j][:, i, :] for j in range(len(merge))],axis=1), save_dir + str(i) +".png", gray=False)
    elif direction=="Z":
        for i in range(Z_size):
            image_save(np.concatenate([merge[j][:, :, i] for j in range(len(merge))],axis=1), save_dir + str(i) +".png", gray=False)
    else:
        assert False

def image_save(picture,path,gray=False):
    if not gray:
        pil_img=PIL.Image.fromarray(np.uint8(picture*255))
        pil_img.save(path)
    else:
        gray_img = np.zeros([np.shape(picture)[0], np.shape(picture)[1], 3], 'float32')
        gray_img[:, :, 0] = picture
        gray_img[:, :, 1] = picture
        gray_img[:, :, 2] = picture
        pil_img=PIL.Image.fromarray(np.uint8(gray_img*255))
        pil_img.save(path)



