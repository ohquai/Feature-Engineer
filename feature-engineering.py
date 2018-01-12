# -*- coding: utf-8 -*-
"""
Feature Engineering for Data Mining or Machine Learning
These methods are written in Python2.7
"""

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.linear_model import Lasso
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import pearsonr


class FeatureEngineeringCommon(object):
    """
    common feature engineering methods
    """

    @staticmethod
    def balance_bi_label(df, max_pro):
        """
        control the maximum degree of unbalance
        对数据进行欠采样，应对严重不均衡的数据标签分布情况
        :param df: input dataframe
        :param max_pro: max proportion of the unbalanced data
        :return:
        """
        df1 = df[df.label == 1]
        df0 = df[df.label == 0]
        len1 = len(df1)
        len0 = len(df0)
        if len1 >= len0:
            df1 = shuffle(df1)
            df1 = df1.iloc[0:len0 * max_pro]
            df = df1.append(df0)
        else:
            df0 = shuffle(df0)
            df0 = df0.iloc[0:len1 * max_pro]
            df = df0.append(df1)

        return df

    @staticmethod
    def get_feature_trend(df, part=0.5):
        """
        get the vary of means between latest data and earlier data to have a better recognition of data
        如果数据是按照时间先后顺序放置的，此方法可求各特征的均值变化趋势，通过对比前n%和后（1-n）%的数据计算
        此方法针对特征分布随时间变化的情况，具有一定的作用
        :param df: input dataframe
        :param part: the proportion of earlier data in whole data
        :return:
        """
        length = df.shape[0]

        early_data = df[:int(length * part)]
        last_data = df[int(length * part):]
        early_mean = early_data.describe().loc['mean']
        last_mean = last_data.describe().loc['mean']
        mean_trend = last_mean/early_mean

        df_desc = pd.DataFrame(mean_trend.reshape(1, len(early_mean)), columns=df.columns, index=['coef'])

        return df_desc

    @staticmethod
    def del_insignificant_column(df):
        """
        delete all-none columns or all-same columns
        删除全空、值全部相同的特征，减少数据维度
        :param df: input dataframe
        :return:
        """
        print("length of columns: {0}".format(len(df.columns)))
        del_col = []
        for c in df.columns:
            if np.isnan(df[c]).all() is True or len(pd.value_counts(df[c])) == 1:
                print("delete all-none or all-same column %s" % c)
                del_col.append(c)
        df.drop(del_col, axis=1, inplace=True)
        print("length of columns: {0}".format(len(df.columns)))
        return df

    @staticmethod
    def lasso_feature_selection(df, exclude_col, label, n_features=100, alpha=0.001, max_iter=10000):
        """
        select columns with Lasso which use l1 regularization
        根据l1正则进行特征筛选，尤其适用于线性回归或逻辑回归等算法
        用户可指定保留的特征数、阈值、循环次数等，也可添加不参与计算的特征
        :param df: input dataframe
        :param exclude_col: columns excluded from selecting
        :param label: label in dataframe
        :param n_features: number of features remaining
        :param alpha: Lasso parameter
        :param max_iter: Lasso parameter
        :return:
        """
        df_x = df.drop(exclude_col, axis=1)
        df_y = df[label]

        sc = StandardScaler()
        train_x = sc.fit_transform(df_x)
        clf = Lasso(max_iter=max_iter, alpha=alpha)
        clf.fit(train_x, df_y)
        print("feature columns are ".format(df_x.columns.tolist()))
        print("feature coefficients are ".format(clf.coef_.tolist()))

        dict_coef = dict(zip(df_x.columns.tolist(), clf.coef_.tolist()))
        dict_sorted = sorted(dict_coef.items(), key=lambda x: x[1], reverse=True)
        col_important = [dict_sorted[:n_features][i][0] for i in range(n_features)]

        col_important.extend(exclude_col)
        df = df[col_important]
        return df

    @staticmethod
    def calculate_vif_(df, thresh=10.0, mode='all', name_start=None, name_end=None, name_contain=None):
        """
        calculate VIF for all columns or indicated by prefix or suffix
        此方法目的为应对特征的多重共线性
        可以通过mode选择潜在存在共线性的特征群，然后在特定特征群中进行筛选
        为了快速，vif计算只计算一次，然后通过对比阈值进行筛选，因此有一定误差，只是为了在效果和效率之间做个妥协
        （正确的方法应为：每次计算所有特征的VIF，删除最高的特征，再重新计算一遍VIF，以此类推，直至全部小于阈值）
        :param df: input dataframe
        :param thresh: threshold of VIF, 5/10 recommended
        :param mode: all columns or indicated columns
        :param name_start: list of prefix
        :param name_end: list of suffix
        :param name_contain: list of inner word
        :return:
        """
        if 'constant' not in df.columns:
            df['constant'] = 1
        mat = df.as_matrix(columns=None)
        col_name = df.columns.tolist()
        print("original columns are {0}".format(col_name))

        high_vif_cols = []
        for i in range(mat.shape[1]):
            if mode == 'all':
                one_vif = variance_inflation_factor(mat, i)
                print(col_name[i], one_vif)
                if one_vif > thresh:
                    high_vif_cols.append(col_name[i])
            elif mode == 'startwith':
                for name in name_start:
                    if col_name[i].startswith(name):
                        one_vif = variance_inflation_factor(mat, i)
                        print(col_name[i], one_vif)
                        if one_vif > thresh:
                            high_vif_cols.append(col_name[i])
            elif mode == 'endwith':
                for name in name_end:
                    if col_name[i].endswith(name):
                        one_vif = variance_inflation_factor(mat, i)
                        print(col_name[i], one_vif)
                        if one_vif > thresh:
                            high_vif_cols.append(col_name[i])
            elif mode == 'in':
                for name in name_contain:
                    if name in col_name[i]:
                        one_vif = variance_inflation_factor(mat, i)
                        print(col_name[i], one_vif)
                        if one_vif > thresh:
                            high_vif_cols.append(col_name[i])
            else:
                raise Exception("invalid mode {0}, with valid mode in: 'all','startwith','endwith' and 'in'".format(mode))

        print("high vif columns are {0}".format(high_vif_cols))
        df.drop(high_vif_cols, axis=1, inplace=True)
        df.drop(['constant'], axis=1, inplace=True)
        print('Remaining columns are {0}'.format(df.columns.tolist()))

        return df

    @staticmethod
    def feature_combine(df, col_headers, col_levels, mode='min', drop=True, dummy_nan=-1):
        """
        combine columns with same prefix and different suffix
        可以将小部分特征合为1个
        方法为：在同一个sample中，求这些特征的值的和/最大值/最小值，并且空值或指定的空值替代值不参与计算
        :param df: input dataframe
        :param col_headers: list of prefix
        :param col_levels: list of suffix
        :param mode: mode to combine the value, within ["min", "max", "sum"]
        :param drop: if drop the original columns
        :param dummy_nan: value that need to be considered as NaN
        :return:
        """
        def f_min(x):
            return np.nanmin(x)

        def f_max(x):
            return np.nanmax(x)

        def f_sum(x):
            return np.nansum(x)

        for col_header in col_headers:
            cols = [col_header+l for l in col_levels if col_header+l in df.columns]
            print("columns to be combine: {0}".format(cols))
            for col in cols:
                df.loc[df[col] == dummy_nan, col] = np.NaN
            if len(cols) > 0:
                if mode == 'min':
                    df.loc[:, col_header+'ttl'] = df[cols].apply(f_min, axis=1)
                elif mode == 'max':
                    df.loc[:, col_header + 'ttl'] = df[cols].apply(f_max, axis=1)
                elif mode == 'sum':
                    df.loc[:, col_header + 'ttl'] = df[cols].apply(f_sum, axis=1)
                else:
                    raise Exception("invalid mode {0}, with valid mode in: 'min','max' and 'sum'".format(mode))
                df.loc[:, col_header+'ttl'] = df[col_header+'ttl'].fillna(dummy_nan)
                if drop:
                    df.drop(cols, axis=1, inplace=True)

        return df

    @staticmethod
    def one_hot_encoder(df, cols, label, if_drop=True):
        """
        one-hot encode for categorical features
        为特征进行one-hot编码，可以指定新生成的特征的名称
        :param df: input dataframe
        :param cols: list of columns to be dealt
        :param label: label for label binarizer
        :param if_drop: if drop the original columns
        :return:
        """
        one_hot_coder = LabelBinarizer()
        one_hot_coder.fit(label)
        one_hot_col_name = []

        for col in cols:
            new_col_names = [col + '_'+str(l) for l in label]
            one_hot_element = one_hot_coder.transform(df[col])
            if 'one_hot_nadarry' not in locals().keys():
                one_hot_nadarry = one_hot_element
                one_hot_col_name = new_col_names
            else:
                one_hot_array = one_hot_element
                one_hot_nadarry = np.hstack((one_hot_nadarry, one_hot_array))
                one_hot_col_name.extend(new_col_names)

        if 'one_hot_nadarry' in locals().keys():
            df_encoder = pd.DataFrame(one_hot_nadarry, columns=one_hot_col_name)
            for col in df_encoder.columns:
                df[col] = df_encoder[col]
        else:
            raise Exception("no column in {0} is valid".format(cols))

        if 'one_hot_nadarry' in locals().keys() and if_drop:
            df.drop(cols, axis=1, inplace=True)

        return df

    @staticmethod
    def del_col_pearsonr(df, label, correlation=True, corr_thres=0.001, pv=False, p_value=0.05):
        """
        drop columns with low correlation or high p-value with target
        通过计算单个特征和目标（label）之间的线性相关性，进行特征筛选
        可以通过相关系数或/和p-value进行筛选，默认只开启相关系数，可调节
        :param df: input dataframe
        :param label: target column
        :param correlation: if use correlation filter
        :param corr_thres: the threshold of minimum correlation
        :param pv: if use p-value filter
        :param p_value: the threshold of maximum p-value
        :return:
        """
        # 删除相关度低和p-value高的特征
        col_del = []
        for col in df.columns:
            corr, pvalue = pearsonr(df[col], df[label])
            if correlation:
                if np.isnan(corr) is True or abs(corr) <= corr_thres:
                    print(col, pearsonr(df[col], df[label])[0])
                    print("low correlation")
                    col_del.append(col)
            if pv:
                if pvalue > p_value:
                    print(col, pearsonr(df[col], df[label])[1])
                    print("high p-value")
                    col_del.append(col)

        if len(col_del) > 0:
            df.drop(col_del, axis=1, inplace=True)
        return df
