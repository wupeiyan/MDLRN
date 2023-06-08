import os
import os.path as osp
import pandas as pd
from radiomics import featureextractor
import argparse

def radiomics_feature_extractor(img_path, roi_path, yaml_path, save_path):
    '''
    : param img_path: image floder path
    : param roi_path: roi floder path
    : param yaml_path: pyradiomics config file path
    : param save_path: save csv path
    '''
    img_list = [osp.join(img_path, i) for i in os.listdir(img_path)]
    roi_path = [osp.join(roi_path, i) for i in os.listdir(roi_path)]

    feature_df = pd.DataFrame()
    names = []
    labels = []

    for img, roi in zip(img_list, roi_path):
        extractor = featureextractor.RadiomicsFeatureExtractor(yaml_path)
        featureVector = extractor.execute(img, roi)
        feature_item = pd.DataFrame.from_dict([featureVector.values()])
        feature_item.columns = featureVector.keys()
        feature_df = pd.concat([feature_df, feature_item])
        name = osp.basename(img).split('.')[0]
        names.append(name)
        if 'po' in name:
            labels.append(1)
        else:
            labels.append(0)
    
    feature_df = feature_df.iloc[:, 37:] # delete first 37 columns
    feature_df.insert(0, "label", labels)
    feature_df.insert(0, "names", names)

    os.makedirs(osp.dirname(save_path), exist_ok=True)

    feature_df.to_csv(save_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='radiomics tools')

    parser.add_argument('img_path', help='image floder path', type=str)
    parser.add_argument('roi_path', help='roi floder path', type=str)
    parser.add_argument('yaml_path', help='pyradiomics config file path', type=str)
    parser.add_argument('save_path', help='save csv path', type=str)
    args = parser.parse_args()
    radiomics_feature_extractor(args.img_path, args.roi_path, args.yaml_path, args.save_path)

    # radiomics_feature_extractor('data/nrrd/PC/bus', 'data/nrrd/PC/roi', 'radiomics/config.yaml', 'radiomics/radiomics_pc.csv')
    # radiomics_feature_extractor('data/nrrd/VC/bus', 'data/nrrd/VC/roi', 'radiomics/config.yaml', 'radiomics/radiomics_vc.csv')
    # radiomics_feature_extractor('data/nrrd/TC1/bus', 'data/nrrd/TC1/roi', 'radiomics/config.yaml', 'radiomics/radiomics_tc1.csv')
    # radiomics_feature_extractor('data/nrrd/TC2/bus', 'data/nrrd/TC2/roi', 'radiomics/config.yaml', 'radiomics/radiomics_tc2.csv')
    # example python radiomics\utils\extract_radiomics_feature.py MDLRN\data\nrrd\PC\bus MDLRN\data\nrrd\PC\roi MDLRN\radiomics\config.yaml radiomics_pc.csv