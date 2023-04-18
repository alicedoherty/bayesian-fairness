"""
Script to load in folktables dataset using some custom 
preprocessing to ensure number of features (data shapes) 
are always consistent. 

Requirements:
    numpy
    pandas
    folktables
    sklearn

"""
import os
import sys
import copy
sys.path.append('..')
import random
import numpy as np
import pandas as pd
import folktables
from folktables import generate_categories
from folktables import ACSDataSource, ACSIncome, ACSEmployment
from sklearn.model_selection import train_test_split

SEED = 0
random.seed(SEED)
np.random.seed(SEED)


ACS_categories = {
    "COW": {
        1.0: (
            "Employee of a private for-profit company or"
            "business, or of an individual, for wages,"
            "salary, or commissions"
        ),
        2.0: (
            "Employee of a private not-for-profit, tax-exempt,"
            "or charitable organization"
        ),
        3.0: "Local government employee (city, county, etc.)",
        4.0: "State government employee",
        5.0: "Federal government employee",
        6.0: (
            "Self-employed in own not incorporated business,"
            "professional practice, or farm"
        ),
        7.0: (
            "Self-employed in own incorporated business,"
            "professional practice or farm"
        ),
        8.0: "Working without pay in family business or farm",
        9.0: "Unemployed and last worked 5 years ago or earlier or never worked",
    },
    "SCHL": {
        1.0: "No schooling completed",
        2.0: "Nursery school, preschool",
        3.0: "Kindergarten",
        4.0: "Grade 1",
        5.0: "Grade 2",
        6.0: "Grade 3",
        7.0: "Grade 4",
        8.0: "Grade 5",
        9.0: "Grade 6",
        10.0: "Grade 7",
        11.0: "Grade 8",
        12.0: "Grade 9",
        13.0: "Grade 10",
        14.0: "Grade 11",
        15.0: "12th grade - no diploma",
        16.0: "Regular high school diploma",
        17.0: "GED or alternative credential",
        18.0: "Some college, but less than 1 year",
        19.0: "1 or more years of college credit, no degree",
        20.0: "Associate's degree",
        21.0: "Bachelor's degree",
        22.0: "Master's degree",
        23.0: "Professional degree beyond a bachelor's degree",
        24.0: "Doctorate degree",
    },
    "MAR": {
        1.0: "Married",
        2.0: "Widowed",
        3.0: "Divorced",
        4.0: "Separated",
        5.0: "Never married or under 15 years old",
    },
    "SEX": {1.0: "Male", 2.0: "Female"},
    "RAC1P": {
        1.0: "White alone",
        2.0: "Black or African American alone",
        3.0: "American Indian alone",
        4.0: "Alaska Native alone",
        5.0: (
            "American Indian and Alaska Native tribes specified;"
            "or American Indian or Alaska Native,"
            "not specified and no other"
        ),
        6.0: "Asian alone",
        7.0: "Native Hawaiian and Other Pacific Islander alone",
        8.0: "Some Other Race alone",
        9.0: "Two or More Races",
    },
}
# def generate_categories(features, definition_df):
#     """Generates a categories dictionary using the provided definition dataframe. Does not create a category mapping
#     for variables requiring the 2010 Public use microdata area code (PUMA) as these need an additional definition
#     file which are not unique without the state code.
#     Args:
#         features: list (list of features to include in the categories dictionary, numeric features will be ignored)
#         definition_df: pd.DataFrame (received from ```ACSDataSource.get_definitions()''')
#     Returns:
#         categories: nested dict with columns of categorical features
#             and their corresponding encodings (see examples folder)."""
#     categories = {}
#     for feature in features:
#         if 'PUMA' in feature:
#             continue

#         # extract definitions for this feature
#         coll_definition = definition_df[(definition_df[0] == 'VAL') & (definition_df[1] == feature)]

#         # extracts if the feature is numeric or categorical --> 'N' == numeric
#         coll_type = coll_definition.iloc[0][2]
#         if coll_type == 'N':
#             # do not add to categories
#             continue

#         # transform to numbers as downloaded definitions are in string format.
#         # -99999999999999.0 is used as a placeholder value for NaN
#         # as multiple NaN values are seen as different keys in a dictionary, a placeholder is needed
#         mapped_col = pd.to_numeric(coll_definition[4], errors='coerce').fillna(-99999999999999.0)
#         mapping_dict = dict(zip(mapped_col.tolist(), coll_definition[6].tolist()))

#         # add default value when not already available from definitions
#         if -99999999999999.0 not in mapping_dict:
#             mapping_dict[-99999999999999.0] = 'N/A'
#         # transform placeholder value back to NaN ensuring a single NaN key instaid of multiple
#         mapping_dict[float('nan')] = mapping_dict[-99999999999999.0]
#         del mapping_dict[-99999999999999.0]

#         categories[feature] = mapping_dict
#     return 

def custom_df_to_pandas(cust, df, categories=None, dummies=False):
    """Filters and processes a DataFrame (received from ```ACSDataSource''').

    Args:
        df: pd.DataFrame (received from ```ACSDataSource''')
        categories: nested dict with columns of categorical features
            and their corresponding encodings (see examples folder)
        dummies: bool to indicate the creation of dummy variables for
            categorical features (see examples folder)

    Returns:
        pandas.DataFrame."""

    df = cust._preprocess(df)
    variables = df[cust.features]
    #for i in categories:
    #    try:
    #        variables[i] = pd.Categorical(variables[i], dtype=pd.CategoricalDtype(categories=categories[i]))
    #    except:
    #        continue
            
    if categories:
        variables = variables.replace(categories)
    
    if dummies:
        variables = pd.get_dummies(variables) #added dummy_na=True to standardize the dfs from this function
         
    variables = pd.DataFrame(cust._postprocess(variables.to_numpy()),
                             columns=variables.columns)

    if cust.target_transform is None:
        target = df[cust.target]
    else:
        target = cust.target_transform(df[cust.target])

    target = pd.DataFrame(target).reset_index(drop=True)

    if cust._group:
        group = cust.group_transform(df[cust.group])
        group = pd.DataFrame(group).reset_index(drop=True)
    else:
        group = pd.DataFrame(0, index=np.arange(len(target)), columns=["group"])
    return variables, target, group
    

def get_dataset(dataset, year='2015', state='CA', keep_group=True):
    """ Returns the test, train, and validation labels for a folktables dataset
    for the particular state and year. 
    *IMPORTANT USAGE NOTES* 
    - This returns the dataset with gender as the protected attribute. 
    - When `keep_group' is true, the last data feature indicates the group label (male/female)
    """
    data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
    if(dataset == "Folk"):
        CustomIncome = folktables.BasicProblem(
            features=['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX',
                #'RAC1P',
            ],
            target='PINCP',
            target_transform=lambda x: x > 25000,    
            group='SEX',
            preprocess=folktables.adult_filter,
            postprocess=lambda x: np.nan_to_num(x, -1),
        )
        ca_data = data_source.get_data(states=[state],  download=True)
        #ca_features, ca_labels, grp = CustomIncome.df_to_pandas(ca_data, categories=ACS_categories, dummies=True)
        ca_features, ca_labels, grp = custom_df_to_pandas(CustomIncome, ca_data, categories=ACS_categories, dummies=True)
        
    elif(dataset == "Employ"):
        CustomEmployment = folktables.BasicProblem(
            features=[
                'AGEP', 'SCHL', 'MAR', 'RELP', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC', 'NATIVITY',
                'DEAR', 'DEYE', 'DREM',
                #'RAC1P',
                'SEX',
            ],
            target='ESR',
            target_transform=lambda x: x == 1,
            group='SEX',
            preprocess=lambda x: x,
            postprocess=lambda x: np.nan_to_num(x, -1),
        )
        ca_data = data_source.get_data(states=[state],  download=True)
        #ca_features, ca_labels, grp = CustomEmployment.df_to_pandas(ca_data, categories=ACS_categories, dummies=True) 
        ca_features, ca_labels, grp = custom_df_to_pandas(CustomEmployment, ca_data, categories=ACS_categories, dummies=True)
    elif(dataset == "Insurance"):
        CustomInsurance = folktables.BasicProblem(
            features=['AGEP', 'SCHL', 'MAR', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC', 'NATIVITY',
                      'DEAR', 'DEYE', 'DREM', 'PINCP', 'ESR', 'ST', 'FER', 'SEX',
            ],
            target='HINS2',
            target_transform=lambda x: x == 1,
            group='SEX',
            preprocess=lambda x: x,
            postprocess=lambda x: np.nan_to_num(x, -1),
        )
        ca_data = data_source.get_data(states=[state],  download=True)
        #ca_features, ca_labels, grp = CustomInsurance.df_to_pandas(ca_data, categories=ACS_categories, dummies=True)
    elif(dataset == "Coverage"):
        def public_coverage_filter(data):
            """
            Filters for the public health insurance prediction task; focus on low income Americans, and those not eligible for Medicare
            """
            df = data
            df = df[df['AGEP'] < 65]
            df = df[df['PINCP'] <= 30000]
            return df

        CustomCoverage = folktables.BasicProblem(
            features=['AGEP', 'SCHL', 'MAR', 'DIS', 'ESP', 'CIT', 'MIG','MIL', 'ANC',
                      'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'PINCP', 'ESR', 'ST', 'FER',
                      'SEX', #'RAC1P',
            ],
            target='PUBCOV',
            target_transform=lambda x: x == 1,
            group='SEX',
            preprocess=public_coverage_filter,
            postprocess=lambda x: np.nan_to_num(x, -1),
        )
        ca_data = data_source.get_data(states=[state],  download=True)
        ca_features, ca_labels, grp = CustomCoverage.df_to_pandas(ca_data, categories=ACS_categories, dummies=True)
    bin_indexes = []
    con_indexes = []
    i = 0
    for column in ca_features:
        if("_" in column):
            bin_indexes.append(i)
        else:
            con_indexes.append(i)
        i+=1
    ca_features = ca_features.to_numpy()
    ca_labels = ca_labels.to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(ca_features, ca_labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    y_val = np.squeeze(y_val)

    from sklearn.preprocessing import StandardScaler
    from sklearn.base import BaseEstimator, TransformerMixin
    class CustomScaler(BaseEstimator,TransformerMixin): 
        # note: returns the feature matrix with the binary columns ordered first  
        def __init__(self,bin_vars_index,cont_vars_index,copy=True,with_mean=True,with_std=True):
            self.scaler = StandardScaler()#(copy,with_mean,with_std)
            self.bin_vars_index = bin_vars_index
            self.cont_vars_index = cont_vars_index
        def fit(self, X, y=None):
            self.scaler.fit(X[:,self.cont_vars_index], y)
            return self

        def transform(self, X, y=None, copy=None):
            X_tail = self.scaler.transform(X[:,self.cont_vars_index],y)
            return np.concatenate((X_tail, X[:,self.bin_vars_index]), axis=1)

    scaler = CustomScaler(bin_indexes, con_indexes)
    scaled_data = scaler.fit_transform(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    y_train = y_train.astype(int)
    y_val   = y_val.astype(int)
    y_test  = y_test.astype(int)
    
    X_train = X_train[:,:-1]
    X_test = X_test[:,:-1]
    X_val = X_val[:,:-1]
    
    if(not keep_group):
        X_train = X_train[:,:-1]
        X_test = X_test[:,:-1]
        X_val = X_val[:,:-1]
    return X_train, X_test, X_val, y_train, y_test, y_val
    #return torch.tensor(X_train).float(), torch.tensor(X_test).float(), torch.tensor(X_val).float(), torch.Tensor(y_train).long(), torch.Tensor(y_test).long(), torch.Tensor(y_val).long()



    