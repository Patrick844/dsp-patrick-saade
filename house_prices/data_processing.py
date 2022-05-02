
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_log_error
pd.options.mode.chained_assignment = None


def split_train_test(df):

    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    xtrain, xtest, ytrain, ytest = train_test_split(X,
                                                    Y,
                                                    test_size=0.30,
                                                    random_state=42)
    return xtrain, xtest, ytrain, ytest


def feature_selection(df):

    feature_selected = ["Neighborhood",
                        "LotArea",
                        "Utilities",
                        "OverallQual",
                        "YearBuilt",
                        "GrLivArea",
                        "ExterCond",
                        "1stFlrSF",
                        "TotRmsAbvGrd",
                        "KitchenQual"]

    df = df[feature_selected]
    return df


def divide_by_type(df):

    categorical_features = [features for features in df.columns
                            if df[features].dtype == "O"]
    numerical_features = [features for features in df.columns
                          if df[features].dtype != "O"]
    date_features = [features for features in df.columns if "Yr" in features
                     or "Year" in features
                     or "Mo" in features]
    features = []
    for feature in numerical_features:
        if feature not in date_features:
            features.append(feature)
    numerical_features = features
    return categorical_features, numerical_features, date_features


def divide_ordinal_features(df, categorical_features, numerical_features):

    num_max = df[numerical_features].max()
    ordinal_numerical_features = num_max[num_max <= 15].index.tolist()
    ordinal_features = [features for features in df.columns
                        if re.search('Qu$', features)
                        or re.search('QC', features)
                        or re.search('Qual$', features)
                        or re.search('Cond$', features)]

    ordinal_categorical_features = [features for features in ordinal_features
                                    if df[features].dtype == "O"]
    return (ordinal_features,
            ordinal_numerical_features,
            ordinal_categorical_features)


def update_categorical_and_numerical_features(numerical_features,
                                              categorical_features,
                                              features_to_remove):

    update_numerical = []
    for feature in numerical_features:
        if feature not in (features_to_remove):
            update_numerical.append(feature)

    update_categorical = []
    for feature in categorical_features:
        if feature not in features_to_remove:
            update_categorical.append(feature)

    return update_numerical, update_categorical


def fill_numerical_missing_values(df, numerical_features):
    df_numerical = df[numerical_features].fillna(0)
    return df_numerical


def fit_scaler_min_max(df, numerical_features):
    scaler = MinMaxScaler()
    scaler.fit(df[numerical_features])
    pickle.dump(scaler, open('../models/MinMax_Numerical_scaler.pickle', 'wb'))
    return scaler


def transform_scaler_min_max(df, numerical_features):

    scaler = pickle.load(open('../models/MinMax_Numerical_scaler.pickle',
                              'rb'))
    df[numerical_features] = scaler.transform(df[numerical_features])

    return df


def fill_missing_categorical_values(df, categorical_features):

    for feature in categorical_features:
        if df[feature].isnull().sum() == 1:
            df[feature] = df[feature].fillna(df[feature].mode())
        else:
            df[feature] = df[feature].fillna("Missing")
    return df[categorical_features]


def fit_one_hot_encoding(df, categorical_features):

    enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    enc.fit(df[categorical_features])
    pickle.dump(enc, open('../models/One_Hot_Encoder.pickle', 'wb'))

    return enc


def transform_one_hot(df, categorical_features):

    enc = pickle.load(open('../models/One_Hot_Encoder.pickle', 'rb'))
    values = enc.transform(df[categorical_features])
    names = enc.get_feature_names_out(df[categorical_features].columns)
    df1 = pd.DataFrame(columns=names)
    df = pd.concat([df, df1], axis=1)
    df[names] = values
    df = df.drop(categorical_features, axis=1)
    return df


def fill_missing_ordinal_numericals_values(df, ordinal_numerical_features):
    df_num_ordinal = df[ordinal_numerical_features]
    if np.sum(df_num_ordinal.isnull().sum() > 0):
        df_num_ordinal = df_num_ordinal.fillna(0)
    return df[ordinal_numerical_features]


def fit_scaler_ordinal_numerical(df, ordinal_numerical_features):
    scaler = MinMaxScaler()
    scaler.fit(df[ordinal_numerical_features])
    pickle.dump(scaler, open('../models/MinMax_Ordinal_Numerical_scaler.pickle',
                              'wb'))
    return scaler

def transform_ordina_numerical_features(df, ordinal_numerical_features):


    scaler = pickle.load(open('../models/MinMax_Ordinal_Numerical_scaler.pickle',  mode = 'rb'))
    df[ordinal_numerical_features] = scaler.transform(df[ordinal_numerical_features])
    return df

def fill_missing_ordinal_categorical_values(df, ordinal_categorical_features):
    
    for feature in ordinal_categorical_features:
        if df[feature].isnull().sum()  == 1: 
            df[feature] = df[feature].fillna(df[feature].mode())
        else:
            df[feature] = df[feature].fillna("Missing")
    return df[ordinal_categorical_features]


def fit_ordinal_categorical(df, ordinal_categorical_features):
    enc = OrdinalEncoder(
                       handle_unknown = "use_encoded_value", 
                       unknown_value = 6
    )
    enc.fit(df[ordinal_categorical_features])
    pickle.dump(enc,  open('../models/Ordinal_Encoder.pickle', "wb"))
    return enc

def transform_ordinal_categorical_features(df, ordinal_categorical_features):
    enc = pickle.load( open('../models/Ordinal_Encoder.pickle', "rb"))
    df[ordinal_categorical_features] = enc.transform(df[ordinal_categorical_features])
    return df

def fit_scaler_ordinal_categorical(df, ordinal_categorical_features):
    scaler = MinMaxScaler()
    scaler.fit(df[ordinal_categorical_features])
    pickle.dump(scaler,  open('../models/MinMax_Ordinal_Categorical_scaler.pickle',  mode = 'wb'))
    return scaler

def transform_scaler_ordinal_categorical(df, ordinal_categorical_features):
    scaler = pickle.load( open('../models/MinMax_Ordinal_Categorical_scaler.pickle',  mode = 'rb'))
    df[ordinal_categorical_features] = scaler.transform(df[ordinal_categorical_features])
    return df

def fill_missing_dates_values(df, date_features):
    
    for features in date_features:
        df[features] =  df[features].fillna(df[features].mode())
    
    return df[date_features]


def fit_scaler_dates(df, date_features):
    scaler = MinMaxScaler()
    scaler.fit(df[date_features])
    pickle.dump(scaler,  open('../models/MinMax_dates_scaler.pickle',  mode = 'wb'))
    return scaler

def transform_dates(df, date_features):
    scaler = pickle.load(open('../models/MinMax_dates_scaler.pickle',  mode = 'rb'))
    df[date_features] = scaler.transform(df[date_features])
    return df

def pipeline_train_numerical_features(df, numerical_features):
    
    numerical_null = fill_numerical_missing_values(df, numerical_features)
    df.loc[:, numerical_features] = numerical_null
    scaler = fit_scaler_min_max(df, numerical_features)
    df = transform_scaler_min_max(df, numerical_features)
    return df


def pipeline_train_categorical_feature(df, categorical_features):
    
    df[categorical_features] = fill_missing_categorical_values(df, categorical_features)
    enc = fit_one_hot_encoding(df, categorical_features)
    df = transform_one_hot(df, categorical_features)
    return df
    
def pipeline_train_ordinal_numerical_features(df, ordinal_numerical_features):
    
    df[ordinal_numerical_features] = fill_missing_ordinal_numericals_values(df, ordinal_numerical_features)
    scaler = fit_scaler_ordinal_numerical(df, ordinal_numerical_features)
    df = transform_ordina_numerical_features(df, ordinal_numerical_features)
    return df
    
def pipeline_train_ordinal_categorical_features(df, ordinal_categorical_features):
    
    df[ordinal_categorical_features] = fill_missing_ordinal_categorical_values(df
                                                                             , ordinal_categorical_features)
    enc = fit_ordinal_categorical(df, ordinal_categorical_features)
    df = transform_ordinal_categorical_features(df, ordinal_categorical_features)
    scaler = fit_scaler_ordinal_categorical(df, ordinal_categorical_features)
    df = transform_scaler_ordinal_categorical(df, ordinal_categorical_features)
    return df

def pipeline_train_dates_features(df, date_features):
    
    df[date_features] = fill_missing_dates_values(df, date_features)
    scaler = fit_scaler_dates(df, date_features)
    df = transform_dates(df, date_features)
    return df
    
def splitting_types(df):
    df = feature_selection(df)
    categorical_features, numerical_features, date_features = divide_by_type(df)
    ordinal_features,  ordinal_numerical_features,  ordinal_categorical_features = divide_ordinal_features(df, categorical_features, numerical_features)
    features_to_remove = ordinal_categorical_features+ordinal_numerical_features
    numerical_features, categorical_features = update_categorical_and_numerical_features(numerical_features, 
                                                                                    categorical_features, 
                                                                                   features_to_remove)
    return numerical_features, categorical_features, ordinal_features, ordinal_numerical_features, ordinal_categorical_features, date_features, df



def pipeline_train(df):
    numerical_features, categorical_features, ordinal_features, ordinal_numerical_features, ordinal_categorical_features, date_features, df = splitting_types(df)
    df = pipeline_train_numerical_features(df, numerical_features)
    df = pipeline_train_categorical_feature(df, categorical_features)
    df = pipeline_train_ordinal_numerical_features(df, ordinal_numerical_features)
    df = pipeline_train_ordinal_categorical_features(df, ordinal_categorical_features)
    df = pipeline_train_dates_features(df, date_features)
    return df

# Test

def pipeline_test_numerical_features(df, numerical_features):
    
    numerical_null = fill_numerical_missing_values(df, numerical_features)
    df.loc[:, numerical_features] = numerical_null
    df = transform_scaler_min_max(df, numerical_features)
    return df
    
def pipeline_test_categorical_feature(df, categorical_features):
    
    df[categorical_features] = fill_missing_categorical_values(df, categorical_features)
    df = transform_one_hot(df, categorical_features)
    return df

def pipeline_test_ordinal_numerical_features(df, ordinal_numerical_features):
    
    df[ordinal_numerical_features] = fill_missing_ordinal_numericals_values(df, ordinal_numerical_features)
    df = transform_ordina_numerical_features(df, ordinal_numerical_features)
    return df
    
def pipeline_test_ordinal_categorical_features(df, ordinal_categorical_features):
    
    df[ordinal_categorical_features] = fill_missing_ordinal_categorical_values(df
                                                                             , ordinal_categorical_features)
    df = transform_ordinal_categorical_features(df, ordinal_categorical_features)
    df = transform_scaler_ordinal_categorical(df, ordinal_categorical_features)
    return df
    
def pipeline_test_dates_features(df, date_features):
    
    df[date_features] = fill_missing_dates_values(df, date_features)
    df = transform_dates(df, date_features)
    return df


def pipeline_test(df):
    numerical_features, categorical_features, ordinal_features, ordinal_numerical_features, ordinal_categorical_features, date_features, df = splitting_types(df)
    df = pipeline_test_numerical_features(df, numerical_features)
    df = pipeline_test_categorical_feature(df, categorical_features)
    df = pipeline_test_ordinal_numerical_features(df, ordinal_numerical_features)
    df = pipeline_test_ordinal_categorical_features(df, ordinal_categorical_features)
    df = pipeline_test_dates_features(df, date_features)
    return df


def compute_rmsle(y_test: np.ndarray,  y_pred: np.ndarray,  precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test,  y_pred))
    return round(rmsle,  precision)
