import pandas as pd
import numpy as np
import pickle
#import xgboost
import joblib
import lightgbm
import json
import sqlalchemy
from sqlalchemy import types
#import cx_Oracle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
plt.style.use('seaborn-whitegrid')
sns.set_style("white")
import matplotlib.dates as mdates
from  matplotlib.dates import MonthLocator, DateFormatter,DayLocator
import matplotlib.ticker as ticker
months = mdates.DayLocator()  # every month
months_fmt = mdates.DateFormatter("%d.%m")

import pyodbc

conn_str = (
    "DRIVER={PostgreSQL Unicode};"
    "DATABASE=db;"
    "UID=user_login;"
    "PWD=;"
    "SERVER=server;"
    "PORT=port;"
)

cnxn = pyodbc.connect(conn_str)


sql = ("""
select * from tablename

""")

df = pd.read_sql(sql, cnxn)
df.columns = [*map(str.upper, df.columns)]
rename_dict = {'CURR_DELINQ_ALL_DBT_N':'CURR_DELINQ_ALL_DBT',
               'IN_SALDO_DEBT_CURR':'IN_SALDO_DEBT_CURR_RESH_V_SUD',
               'EXPOSURE_CURRENT_DBT_N':'EXPOSURE_CURRENT_DBT',
               'DELINQ_DEBT_AMOUNT_3M_DBT_N':'DELINQ_DEBT_AMOUNT_3M_DBT',
               'EXPOSURE_CURRENT_DBT_N':'EXPOSURE_CURRENT_DBT',
               'EXPOSURE_DEL_DEBT_CURRENT_DBT_N':'EXPOSURE_DEL_DEBT_CURRENT_DBT',
               'CURR_DELINQ_CARD_DBT_N':'CURR_DELINQ_CARD_DBT',
               'IN_SALDO_DELAYED_DEBT_CURR':'RESH_V_SUD_OD',
               'PAYMENTFACT_SUM_DBT_N':'PAYMENTFACT_SUM_DBT',
               'MAX_DELINQ_DEBT_12M_DBT_N':'MAX_DELINQ_DEBT_12M_DBT',
               'SUM_NEXT_PAYMENT_CONSUMER_DBT_N':'SUM_NEXT_PAYMENT_CONSUMER_DBT',
               'CC_MAX_MAIN_DEBT_3M_DBT_N':'CC_MAX_MAIN_DEBT_3M_DBT'}
df = df.rename(rename_dict, axis='columns')
df.loc[df.KALITA_PRODUCT_CODE.eq('CONSUMER'), 'IS_PK'] = 1
df.loc[df.KALITA_PRODUCT_CODE.eq('HOUSE_LOAN'), 'IS_PK'] = 0
model_rr1 = joblib.load('best_estimator_RR1.pkl')
model_rr0 = joblib.load('best_estimator_RR0.pkl')
feature_list_0 = ['MONTHS_LAST_DEL_PRC_MIN_DBT',
'CURR_DELINQ_ALL_DBT',
'NUM_DEAL_DELINQ_DBT',
'CC_AVG_UTIL_3M_DBT',
'FLAG_DECREASE_DELINQ_ALL_DBT',
'PAID_MAIN_DEBT_2_ADVANCE_DBT',
'IN_SALDO_DEBT_CURR_RESH_V_SUD',
'EXPOSURE_CURRENT',
'DELINQ_DEBT_AMOUNT_3M_DBT',
'EXPOSURE_DEL_DEBT_CURRENT_DBT',
'CURR_DELINQ_CARD_DBT',
'IS_PK',
'CC_UTIL_3M_2_UTIL_4_6M_DBT',
'RESH_V_SUD_OD',
'PAYMENTFACT_SUM_DBT',
'MAX_DELINQ_DEBT_12M_DBT',
'SUM_NEXT_PAYMENT_CONSUMER_DBT',
'DEL_DURATION_MAX_DBT',
'CC_MAX_MAIN_DEBT_3M_DBT']

feature_list_1 = [
'PAID_MAIN_DEBT_2_ADVANCE_DBT',
'MAINDEBT_2_MAINDEBT_1M_DBT',
'IS_PK',
'RESH_V_SUD_OD',
'MONTHS_LAST_DEL_PRC_MIN_DBT',
'CURR_DELINQ_CARD_DBT',
'CURR_DEL_DAYS_DBT', 
 'PAYMENTFACT_SUM_DBT',
'NUM_DEAL_DELINQ_DBT']
feature_list_0 or feature_list_1
df[feature_list_0 or feature_list_1].fillna(-999999999)
def predict(df):
    """
    function to aggregate calculation of proba and calibration
    """
    
    df = proba_calc(df)
    
    df = calibr_calc(df)
    
    return df

def proba_calc(df):
    """
    function to calculate probas from model
    """
    df['PROBA_RR0'] = model_rr0.predict(df[feature_list_0])
    df['PROBA_RR1'] = model_rr1.predict(df[feature_list_1])
    
    return df
    
def calibr_calc(df):
    """
    function to calculate calibration
    """
    df['SCORE_RR0']=np.log(df['PROBA_RR0'] / (1. - df['PROBA_RR0']))
    df['PROBA_CALIBR_RR0']=1/(1+np.exp(-( - 0.30413 + 0.99183  * df['SCORE_RR0'])))

    df['SCORE_RR1']=np.log(df['PROBA_RR1'] / (1. - df['PROBA_RR1']))
    df['PROBA_CALIBR_RR1']=1/(1+np.exp(-( 0.36553 + 1.06781  * df['SCORE_RR1'])))
    
    return df

df = predict(df)

rr_0_cutoff = 0.6
rr_1_cutoff = 0.16

segment_percentage_0 = 0.5
segment_percentage_1 = 0.5
segment_percentage_2 = 0.5

def cutoff_apply(df):
    """
    apply cutoff to probas
    """

    amnistion_selection = [
    (df['PROBA_CALIBR_RR0'] <= rr_0_cutoff),
    (df['PROBA_CALIBR_RR1'] >= rr_1_cutoff),
    (df['PROBA_CALIBR_RR0'].gt(rr_0_cutoff) & df['PROBA_CALIBR_RR1'].lt(rr_1_cutoff))]

    choices = [0,1,2]

    df['FL_TYPE_AMN_CR'] = np.select(amnistion_selection, choices)
    
    return df

def group_split(df):
    """
    function to group client and apply group selection
    """
    
    
    df_cl = df.groupby(['EPK_ID']).FL_TYPE_AMN_CR.min().reset_index().rename(columns = {'FL_TYPE_AMN_CR':'FL_TYPE_AMN_CL'})
    df = df.merge(df_cl, how = 'left', on = 'EPK_ID')
    
    choices = ['PILOT', 'CONTROL']
    df.loc[df.FL_TYPE_AMN_CL.eq(0), 'GROUP'] = np.random.choice(choices, len(df.loc[df.FL_TYPE_AMN_CL.eq(0)]), p = [segment_percentage_0, 1-segment_percentage_0])
    df.loc[df.FL_TYPE_AMN_CL.eq(1), 'GROUP'] = np.random.choice(choices, len(df.loc[df.FL_TYPE_AMN_CL.eq(1)]), p = [segment_percentage_1, 1-segment_percentage_1])
    df.loc[df.FL_TYPE_AMN_CL.eq(2), 'GROUP'] = np.random.choice(choices, len(df.loc[df.FL_TYPE_AMN_CL.eq(2)]), p = [segment_percentage_2, 1-segment_percentage_2])
    
    return df

df = group_split(cutoff_apply(df))
