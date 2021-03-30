import pandas as pd
import numpy as np


def fill_missing_values(df):
    '''
    input: a raw dataframe of mutual fund data
    output: a dataframe with missing values filled
    '''
    df.drop(df[df['Morningstar Risk']=='--'].index, inplace=True)
    df.drop('Average Moat Rating', axis=1, inplace=True)
    df.drop(df[df['ROE Last Year (%)']=='--'].index, inplace=True)
    pe_mean = df['Price/Earnings'].astype(float).mean()
    pcf_mean = df[df['Price/Cash Flow']!='--']['Price/Cash Flow'].astype(float).mean()
    pe_pcf_diff = pe_mean - pcf_mean
    
    df.loc[df['Price/Cash Flow']=='--','Price/Cash Flow'] = df.loc[df['Price/Cash Flow']=='--','Price/Earnings'].astype(float) - pe_pcf_diff
    
    df.drop(df[df['% Assets in Top 10 Holdings']=='--'].index, inplace=True)
    
    return df

def fill_missing_values2(df):
    '''
    input: a raw dataframe of mutual fund data
    output: a dataframe with missing values filled
    '''
    df.drop(df[df['3-year Return (%)']=='--'].index, inplace=True)
    df.drop(df[df['Dividend Yield (%)']=='--'].index, inplace=True)
    df.drop(df[df['Morningstar Sustainability Rating']=='--'].index, inplace=True)
    df.drop(df[df['Turnover Ratio (%)']=='--'].index, inplace=True)
    
    df['S&P500 3-year Return (%)'] = df['S&P500 3-year Return (%)'].replace(34.2, 10.3)

    return df

def convert_to_num(df):
    '''
    input: a dataframe with no missing values
    output: a dataframe with data types for numerical features converted to integers or floats 
    '''   
    df['Morningstar Sustainability Rating'] = df['Morningstar Sustainability Rating'].astype(int)
    df['3-year Return (%)'] = df['3-year Return (%)'].astype(float)
    df['Dividend Yield (%)'] = df['Dividend Yield (%)'].astype(float)
    df['Price/Earnings'] = df['Price/Earnings'].astype(float)
    df['Price/Cash Flow'] = df['Price/Cash Flow'].astype(float)
    df['Average Market Cap ($ mil)'] = df['Average Market Cap ($ mil)'].astype(int)
    df['ROE Last Year (%)'] = df['ROE Last Year (%)'].astype(float)
    df['Debt/Capital Last Year (%)'] = df['Debt/Capital Last Year (%)'].astype(float)
    df['% Assets in Top 10 Holdings'] = df['% Assets in Top 10 Holdings'].astype(float)
    df['Turnover Ratio (%)'] = df['Turnover Ratio (%)'].astype(int)
    df['Expense Ratio (%)'] = df['Expense Ratio (%)'].astype(float)
    df['Minimum Initial Purchase ($)'] = df['Minimum Initial Purchase ($)'].astype(int)
    df['% Assets in Top 10 Holdings'] = df['% Assets in Top 10 Holdings'].astype(float)
    
    return df

def add_outperform_underperform(df):
    '''
    input: a cleaned dataframe
    output: a cleaned dataframe with an outperform/underperform column added 
    '''  
    df['3-year Annualized Return vs. S&P500'] = df['3-year Return (%)'] - df['Expense Ratio (%)'] - df['S&P500 3-year Return (%)']
    df.drop(['3-year Return (%)','S&P500 3-year Return (%)'], axis=1,inplace=True)
    
    def outperform(return_diff):
        if return_diff > 0:
            return 'Outperform'
        elif return_diff < 0:
            return 'Underperform'

    df['Outperform / Underperform'] = df['3-year Annualized Return vs. S&P500'].apply(outperform)
    
    return df
    