from mrmr import mrmr_classif
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LassoCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp

def min_redundancy_max_relevance(dataframe, feature_nums=30):
    '''
    params dataframe: dataframe of features
    '''
    X = dataframe.iloc[:, 1:]
    y = dataframe.iloc[:, 0]
    feature_index = mrmr_classif(X=X, y=y, K=feature_nums)
    feature_index.insert(0, 'label')
    return feature_index


def lasso_selection(dataframe, lasso_alphas=[-4, 0, 50]):
    X = dataframe.iloc[:, 1:]
    Y = dataframe['label']

    columns = X.columns
    X = StandardScaler().fit_transform(X)
    X = pd.DataFrame(X)

    X.columns = columns
    alphas = np.logspace(lasso_alphas[0], lasso_alphas[1], lasso_alphas[2])
    lasso_model = LassoCV(alphas=alphas, max_iter=1000000000).fit(X, Y)
    coef = pd.Series(lasso_model.coef_, index=columns)

    print('Lasso picked : {}'.format(sum(coef != 0)))

    feature_index = coef[coef != 0].index
    feature_coef = coef[coef != 0]

    bias = lasso_model.intercept_

    x_values = np.arange(len(feature_index))
    y_values = coef[coef != 0]
    values_sort = sorted(zip(y_values.keys(), y_values.values), key=lambda x:x[1])
    feature_index, y_values = zip(*values_sort)

    
    coefs = lasso_model.path(X, Y, alphas=alphas, max_iter=100000)[1].T
    MSEs = lasso_model.mse_path_
    MSEs_mean = np.apply_along_axis(np.mean, 1, MSEs)
    MSEs_std = np.apply_along_axis(np.std, 1, MSEs)

    

    # lasso特征权重图
    fontsize = 12
    fontdict = {
        'family' : 'sans-serif',
        'fontsize': fontsize
    }
    lw = 2
    fig = plt.figure(1, figsize=(20,15), dpi=900)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.2)
    fig.tight_layout()
    grid = plt.GridSpec(2,4)
    ax = plt.subplot(grid[1, 1:3])

    ax.barh(
        x_values,    
        y_values, 
        color='lightblue', 
        edgecolor='black', 
        alpha=0.8, 
        yerr=0.0001
    )

    ax.set_yticks(
        x_values, 
        feature_index, 
        # rotation='90',
        ha='right', 
        va='top',
        fontdict=fontdict
    )
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize) 
    ax.set_xlabel("weight", fontdict=fontdict)  # 横轴名称
    ax.set_ylabel("feature", fontdict=fontdict)  # 纵轴名称
    ax.spines['bottom'].set_linewidth(lw)
    ax.spines['left'].set_linewidth(lw)
    ax.spines['top'].set_linewidth(lw)
    ax.spines['right'].set_linewidth(lw)

    ax = plt.subplot(grid[0, 0:2])
    ax.spines['bottom'].set_linewidth(lw)
    ax.spines['left'].set_linewidth(lw)
    ax.spines['top'].set_linewidth(lw)
    ax.spines['right'].set_linewidth(lw)

    plt.errorbar(
        lasso_model.alphas_, 
        MSEs_mean, 
        yerr=MSEs_std,  # y误差范围
        fmt="o",        # 数据点标记
        ms=3,           # 数据点大小
        mfc="r",        # 数据点颜色
        mec="r",        # 数据点边缘颜色
        ecolor="lightblue",     # 误差棒颜色
        elinewidth=2,   # 误差棒线宽
        capsize=4,      # 误差棒边界线长度
        capthick=1      # 误差棒边界线厚度
    )  
    plt.semilogx()
    plt.axvline(lasso_model.alpha_, color='black', ls="--")
    plt.xlabel('Lambda', fontdict=fontdict)
    plt.ylabel('MSE', fontdict=fontdict)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize) 


    ax = plt.subplot(grid[0,2:])
    ax.spines['bottom'].set_linewidth(lw)
    ax.spines['left'].set_linewidth(lw)
    ax.spines['top'].set_linewidth(lw)
    ax.spines['right'].set_linewidth(lw)
    plt.semilogx(lasso_model.alphas_, coefs, '-')
    plt.axvline(lasso_model.alpha_, color='black', ls="--")
    plt.xlabel('Lambda', fontdict=fontdict)
    plt.ylabel('Coefficients', fontdict=fontdict)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12) 
    # plt.show()
    plt.savefig('../lasso.svg', format='svg', dpi=900)


    feature_index = list(feature_index)
    feature_index.insert(0,'label')
    return feature_index, feature_coef, bias


def feature_selection(dataframe, mrmr_feature_nums, lasso_alphas):
    # mrmr
    mrmr_feature_index = min_redundancy_max_relevance(dataframe, mrmr_feature_nums)
    dataframe = dataframe[mrmr_feature_index]

    # lasso
    lasso_feature_index, lasso_feature_coef, lasso_bias = lasso_selection(dataframe, lasso_alphas)

    return lasso_feature_index, lasso_feature_coef, lasso_bias


def compute_rad_score(dataframe, coef, bias, save_path):
    '''
    计算lasso选出的特征乘以及对应的权重
    '''

    colunms = ['rad_score']
    X = dataframe.iloc[:, 1:]
    X = StandardScaler().fit_transform(X)  # 将数值标准化
    label = dataframe.iloc[:,0]
    w = coef.to_numpy()

    score = ((w * X).sum(axis=1))
    score = score + bias

    rad_score = pd.DataFrame(score)
    rad_score.columns = colunms
    label = pd.DataFrame(label)

    rad_score.insert(0, 'label', label)

    os.makedirs(osp.dirname(save_path), exist_ok=True)

    rad_score.to_csv(osp.join(save_path), index=None)
    return rad_score


if __name__ == '__main__':
    radiomics_pc = '../radiomics_pc.csv'
    radiomics_vc = '../radiomics_vc.csv'
    radiomics_tc1 = '../radiomics_tc1.csv'
    radiomics_tc2 = '../radiomics_tc2.csv'

    pc_rad_score_save_path = '../pc_rad_score.csv'
    vc_rad_score_save_path = '../vc_rad_score.csv'
    tc1_rad_score_save_path = '../tc1_rad_score.csv'
    tc2_rad_score_save_path = '../tc2_rad_score.csv'

    pc_df = pd.read_csv(radiomics_pc)
    vc_df = pd.read_csv(radiomics_vc)
    tc1_df = pd.read_csv(radiomics_tc1)
    tc2_df = pd.read_csv(radiomics_tc2)

    mrmr_feature_nums = 30
    lasso_alphas = [-4, 0, 50]

    lasso_feature_index, lasso_feature_coef, lasso_bias = feature_selection(pc_df, mrmr_feature_nums, lasso_alphas)
    compute_rad_score(pc_df[lasso_feature_index], lasso_feature_coef, lasso_bias, pc_rad_score_save_path)
    compute_rad_score(vc_df[lasso_feature_index], lasso_feature_coef, lasso_bias, vc_rad_score_save_path)
    compute_rad_score(tc1_df[lasso_feature_index], lasso_feature_coef, lasso_bias, tc1_rad_score_save_path)
    compute_rad_score(tc2_df[lasso_feature_index], lasso_feature_coef, lasso_bias, tc2_rad_score_save_path)