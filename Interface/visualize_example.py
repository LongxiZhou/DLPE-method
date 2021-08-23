import numpy as np
import Tool_Functions.Functions as Functions
import visualization.visualize_3d.highlight_semantics as highlight


def generate_slice(rescaled_ct, enhance_ct, airway_mask, blood_vessel_mask, slice_z, show=False):
    """

    :param rescaled_ct:
    :param enhance_ct: DLPE enhanced effect
    :param airway_mask:
    :param blood_vessel_mask:
    :param slice_z: range from 0 to 511
    :param show:
    :return: image
    """
    assert 0 <= slice_z <= 511
    final_image = np.zeros([512, 512 * 3, 3], 'float32')

    rescaled_ct = rescaled_ct[:, :, slice_z: slice_z + 1]
    rescaled_ct = np.clip(rescaled_ct + 0.5, 0, 1)
    enhance_ct = enhance_ct[:, :, slice_z: slice_z + 1]
    enhance_ct = np.clip(enhance_ct + 0.025, 0, 0.2) * 5

    airway_mask = airway_mask[:, :, slice_z: slice_z + 1]
    blood_vessel_mask = blood_vessel_mask[:, :, slice_z: slice_z + 1]
    output = highlight.highlight_mask(blood_vessel_mask, rescaled_ct, 'R', further_highlight=False)
    output = highlight.highlight_mask(airway_mask, output, 'B', further_highlight=True)[:, 512::, 0, :]

    final_image[:, 0: 512, 0] = rescaled_ct[:, :, 0]
    final_image[:, 0: 512, 1] = rescaled_ct[:, :, 0]
    final_image[:, 0: 512, 2] = rescaled_ct[:, :, 0]

    final_image[:, 512: 1024, :] = output

    final_image[:, 1024::, 0] = enhance_ct[:, :, 0]
    final_image[:, 1024::, 1] = enhance_ct[:, :, 0]
    final_image[:, 1024::, 2] = enhance_ct[:, :, 0]

    if show:
        return Functions.image_show(final_image)
    return final_image
