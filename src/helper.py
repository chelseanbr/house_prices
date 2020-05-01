import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from graphviz import Source
from sklearn.tree import export_graphviz
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

# Pandas df exploration functions

def print_null_ct(df):
    # print how many null values each column has, if any
        print('Count of Null Values per Column, if any:\n\n{}'.format(df.isnull().sum()[df.isnull().sum() > 0]))

def print_unique_ct(df):
    # print how many unique values each column has
    print('Count of Unique Values per Column:\n')
    for col in df.columns:
        print('{}: {}'.format(col, len(df[col].unique())))

def get_cols_of_type(df, type):
    # print names of columns of given type
    cols = list(df.select_dtypes(type).columns)
    print('{} Columns ({}): \n{}'.format(type, len(cols), cols))
    return cols


# Pipeline/prep functions

# def ohe_obj(ohe, col, df):
#     col_name = col + '_is_'

#     # Get object column, fill NaNs with 'None'
#     df_obj = df[col].fillna('None')
#     obj_arr = df_obj.to_numpy().reshape(-1,1)

#     # Create one-hot encoder
#     # ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
#     ohe.fit(obj_arr)    # Assume for simplicity all features are categorical.

#     # Apply the one-hot encoder
#     obj_encoded = ohe.transform(obj_arr)

#     # Store one-hot encoding in dataframe
#     cols = [col_name + cat for cat in ohe.categories_[0]]
#     obj_encoded_df = pd.DataFrame(obj_encoded, columns=cols)
#     return obj_encoded_df

def ohe_obj(ohe, col, train_df, test_df):
    col_name = col + '_is_'

    # Get object column, fill NaNs with 'None'
    train_df_obj = train_df[col].fillna('None')
    train_obj_arr = train_df_obj.to_numpy().reshape(-1,1)

    test_df_obj = test_df[col].fillna('None')
    test_obj_arr = test_df_obj.to_numpy().reshape(-1,1)

    # Create one-hot encoder
    # ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    ohe.fit(train_obj_arr)    # Assume for simplicity all features are categorical.

    # Apply the one-hot encoder
    train_obj_encoded = ohe.transform(train_obj_arr)
    test_obj_encoded = ohe.transform(test_obj_arr)

    # Store one-hot encoding in dataframe
    cols = [col_name + cat for cat in ohe.categories_[0]]

    train_obj_encoded_df = pd.DataFrame(train_obj_encoded, columns=cols)
    test_obj_encoded_df = pd.DataFrame(test_obj_encoded, columns=cols)

    return train_obj_encoded_df, test_obj_encoded_df

def pipeline(train_df, test_df):
    updated_train_df = train_df.copy()
    updated_test_df = test_df.copy()

    # Ohe on object columns
    obj_cols = get_cols_of_type(train_df, 'object')
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    for col in obj_cols:
        train_obj_encoded_df, test_obj_encoded_df = ohe_obj(ohe, col, train_df, test_df)

        updated_train_df = updated_train_df.merge(train_obj_encoded_df, how='left', on=updated_train_df.index)
        updated_train_df.drop(columns=[col, 'key_0'], inplace=True)

        updated_test_df = updated_test_df.merge(test_obj_encoded_df, how='left', on=updated_test_df.index)
        updated_test_df.drop(columns=[col, 'key_0'], inplace=True)

    # Impute nulls in number cols with mean
    imp = SimpleImputer(strategy='mean')

    dfs = [updated_train_df, updated_test_df]
    updated_dfs = []
    for df in dfs:
        idf=pd.DataFrame(imp.fit_transform(df))
        idf.columns=df.columns
        idf.index=df.index
        updated_dfs.append(idf)
    [updated_train_df, updated_test_df] = updated_dfs

    return updated_train_df, updated_test_df

# def pipeline(df):
#     updated_df = df.copy()

#     obj_cols = get_cols_of_type(df, 'object')
#     ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
#     for col in obj_cols:
#         train_obj_encoded_df = ohe_obj(ohe, col, df)
#         updated_df = updated_df.merge(train_obj_encoded_df, how='left', on=updated_df.index)
#         updated_df.drop(columns=[col, 'key_0'], inplace=True)

#     return updated_df

def split_Xy_from_df(target, df):
    X = df.drop(columns=[target]).to_numpy()
    y = df[target].to_numpy()
    return X, y

def split_y_from_df(target, df):
    y = df[target].to_numpy()
    return y

def split_X_from_df(target, df):
    X = df.drop(columns=[target]).to_numpy()
    return X


# Plotting functions

def plot_pie(series, fig, ax):
    # fig, ax = plt.subplots(figsize=(8,8))

    series.value_counts().plot.pie(ax=ax, autopct='%1.2f%%')

    plt.rcParams['font.size'] = 18
    
    fig.tight_layout()
    return fig, ax

def plot_counts_bygroup(df, features, groupby, fig, axs):
    # fig, axs = plt.subplots(6, 4, figsize=(14,18))

    for feature, ax in zip(features, axs.flatten()[:len(features)]):
        ax = sns.countplot(data=df, x=feature, hue=groupby, ax=ax)
        ax.legend_.remove()

    fig.tight_layout()
    return fig, axs

def plot_topN_features(feature_importances, feature_list, N):
    # Plot the feature importance
    idxes = np.argsort(-feature_importances)
    feature_list[idxes]
    rev_sort_feature_importances = feature_importances[idxes]
    rev_sort_feature_cols = feature_list[idxes]

    feat_scores = pd.DataFrame({'Fraction of Samples Affected' : rev_sort_feature_importances[:N]},
                               index=rev_sort_feature_cols[:N])
    feat_scores = feat_scores.sort_values(by='Fraction of Samples Affected')
    feat_scores.plot(kind='barh')
    
    plt.title('Feature Importances', size=25)
    plt.ylabel('Features', size=25)
    return plt, rev_sort_feature_cols

def plot_tree(tree, feature_list, out_file=None):
    # Source(plot_tree(tree, feature_list, out_file=None)) to print in Jupyter nb
    return export_graphviz(tree, out_file=out_file, feature_names=feature_list)

# Modeling
def fit_pred_score_Nfold(model, X_train, y_train, X_test, test_idx, target_col, N=10, model_name=None, csv=None):
    # Fit model
    model.fit(X_train, y_train)
    # Predict
    y_pred = model.predict(X_test)
    # Create submission if csv arg passed
    if csv is not None:
        y_pred_df = pd.DataFrame(y_pred, index=test_idx, columns=[target_col])
        y_pred_df.head()
        y_pred_df.to_csv('submissions/' + csv + '.csv')
    # Get N-fold Cross-Validation RMSE score
    if model_name is None:
        model_name=model.__class__.__name__
    rmse = np.mean(np.sqrt(-cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=N)))
    print(model_name + ' RMSE, {}-fold CV on Train Data: {:0.3f}'.format(N, rmse))