import os
import os.path as osp
import numpy as np
from PIL import Image
import SimpleITK as sitk
import argparse

def conver_png_to_nrrd(source_floder, save_floder):
    '''
    : param source_floder: source floder path
    : param save_floder: save floder path

    convert png to nrrd
    '''
    for root, dirs, files in os.walk(source_floder, topdown=False):
        for name in dirs:
            if name == 'cdfi':
                continue
            os.makedirs(osp.join(root, name).replace(source_floder, save_floder), exist_ok=True)

    for root, dirs, files in os.walk(source_floder, topdown=False):
        for name in files:
            if 'cdfi' in root:
                continue
            file = osp.join(root, name)
            img = np.asarray(Image.open(file).convert('L'))
            img = np.expand_dims(img, axis=2)
            img = sitk.GetImageFromArray(img) 
            sitk.WriteImage(img, file.replace(source_floder, save_floder).replace('png', 'nrrd'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='radiomics tools')

    parser.add_argument('source_floder', help='source floder path', type=str)
    parser.add_argument('save_floder', help='save floder path', type=str)
    args = parser.parse_args()
    conver_png_to_nrrd(args.source_floder, args.save_floder)

    # example python radiomics\utils\extract_radiomics_feature.py MDLRN\data\origin MDLRN\data\nrrd