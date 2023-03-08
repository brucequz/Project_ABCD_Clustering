"""
demographic_data.py

Preprocess demographic data for ABCD Study
"""

import numpy as np
import pandas as pd


def demographics(path) -> pd.DataFrame:
    demo_data=pd.read_csv(path,delimiter="\t",skiprows=(1,1))
    # Demographic data processing
    demo_data = summarize_columns(demo_data)
    demo_data = summarize_asian(demo_data)
    demo_data = summarize_other(demo_data)
    # Replace invalid values with NaN
    demo_data['race'] = demo_data.apply(lambda x: get_race(x), axis = 1)
    demo_data = demo_data.replace(777,np.nan)
    demo_data = demo_data.replace(666,np.nan)
    demo_data = demo_data.replace(66,np.nan)
    demo_data = demo_data.replace(77,np.nan)
    demo_data = demo_data.replace(999,np.nan)
    demo_data = demo_data.replace(99,np.nan)

    demo_data['demo_prnt_ed_v2'] = demo_data['demo_prnt_ed_v2'].replace(np.nan, -1)
    demo_data['demo_prtnr_ed_v2'] = demo_data['demo_prtnr_ed_v2'].replace(np.nan, -1)

    demo_data['highest_education_of_parent'] = demo_data.apply(lambda x: x['demo_prnt_ed_v2'] if x['demo_prnt_ed_v2']>= x['demo_prtnr_ed_v2'] else x['demo_prtnr_ed_v2'], axis = 1).astype(int)
    demo_data['education'] = demo_data.apply(lambda x: get_education(x), axis = 1)
    demo_data['ethnicity'] = demo_data.apply(lambda x: get_ethnicity(x), axis = 1)
    demo_data['demo_comb_income_v2'] = demo_data.apply(lambda x: get_income(x), axis = 1)

    demo_data = demo_data.drop(['demo_prnt_race_a_v2___10','demo_prnt_race_a_v2___11', 'demo_prnt_race_a_v2___18', 'demo_prnt_race_a_v2___19', 
                'demo_prnt_race_a_v2___20', 'demo_prnt_race_a_v2___21', 'demo_prnt_race_a_v2___22',
                'demo_prnt_race_a_v2___23', 'demo_prnt_race_a_v2___24', 'demo_prnt_race_a_v2___12',
                'demo_prnt_race_a_v2___13', 'demo_prnt_race_a_v2___14', 'demo_prnt_race_a_v2___15',
                'demo_prnt_race_a_v2___16', 'demo_prnt_race_a_v2___17', 'demo_prnt_race_a_v2___25', 
                'demo_prnt_race_a_v2___18', 'demo_prnt_race_a_v2___19', 
                'demo_prnt_race_a_v2___20', 'demo_prnt_race_a_v2___21', 'demo_prnt_race_a_v2___22',
                'demo_prnt_race_a_v2___23', 'demo_prnt_race_a_v2___24',
                'demo_prnt_race_a_v2___12','demo_prnt_race_a_v2___13', 'demo_prnt_race_a_v2___14', 'demo_prnt_race_a_v2___15',
                'demo_prnt_race_a_v2___16', 'demo_prnt_race_a_v2___17', 'demo_prnt_race_a_v2___25',
                'interview_date', 'collection_id', 'dataset_id', 'collection_title', 'study_cohort_name', 'pdem02_id', 
                'src_subject_id', 'sum_of_races', 'asian_sum', 'demo_prnt_race_a_v2___11', 'other_sum', 'demo_prnt_race_a_v2___10',
                'demo_prnt_race_a_v2___77', 'demo_prnt_race_a_v2___99',
                'demo_prnt_ed_v2', 'demo_prtnr_ed_v2', 'demo_ethn_v2'], axis=1)

    demo_features_keep = ['subjectkey', 'interview_age', 'sex', 'race', 'ethnicity', 'education', 'demo_comb_income_v2']

    demo_data = demo_data[demo_features_keep]
    demo_data = pd.get_dummies(demo_data, columns=['sex', 'race', 'ethnicity', 'education', 'demo_comb_income_v2'])

    return demo_data


def summarize_columns(data):
    data['sum_of_races'] = 0
    races = ['demo_prnt_race_a_v2___10','demo_prnt_race_a_v2___11', 'demo_prnt_race_a_v2___18', 'demo_prnt_race_a_v2___19', 
             'demo_prnt_race_a_v2___20', 'demo_prnt_race_a_v2___21', 'demo_prnt_race_a_v2___22',
             'demo_prnt_race_a_v2___23', 'demo_prnt_race_a_v2___24', 'demo_prnt_race_a_v2___12',
             'demo_prnt_race_a_v2___13', 'demo_prnt_race_a_v2___14', 'demo_prnt_race_a_v2___15',
             'demo_prnt_race_a_v2___16', 'demo_prnt_race_a_v2___17', 'demo_prnt_race_a_v2___25']
    for race in races:
        data['sum_of_races'] += data[race].astype(int)
    return data

def summarize_asian(data):
    data['asian_sum'] = 0
    asian_races = ['demo_prnt_race_a_v2___18', 'demo_prnt_race_a_v2___19', 
             'demo_prnt_race_a_v2___20', 'demo_prnt_race_a_v2___21', 'demo_prnt_race_a_v2___22',
             'demo_prnt_race_a_v2___23', 'demo_prnt_race_a_v2___24']
    for race in asian_races:
        data['asian_sum'] += data[race].astype(int)
    return data   

def summarize_other(data):
    data['other_sum'] = 0
    races = ['demo_prnt_race_a_v2___12','demo_prnt_race_a_v2___13', 'demo_prnt_race_a_v2___14', 'demo_prnt_race_a_v2___15',
             'demo_prnt_race_a_v2___16', 'demo_prnt_race_a_v2___17', 'demo_prnt_race_a_v2___25']
    for race in races:
        data['other_sum'] += data[race].astype(int)
    return data

def get_race(x):
    if x['sum_of_races']==3:
        return 'multiple_race'
    elif x['asian_sum']>=1:
        if x['demo_prnt_race_a_v2___11']==1:
            return 'multiple_race'
        elif x['other_sum']>=1:
            return 'multiple_race'
        elif x['demo_prnt_race_a_v2___10']==1:
            return 'asian'
        else:
            return 'asian'
    elif x['other_sum']>=2: 
        return 'multiple_race'
    elif x['other_sum']==1:
        if x['demo_prnt_race_a_v2___11']==1:
            return 'multiple_race'
        else:
            return 'other_race'
    else:
        if x['demo_prnt_race_a_v2___77']==1 or x['demo_prnt_race_a_v2___99']==1:
            return 'unsure_race'
        elif x['demo_prnt_race_a_v2___11']==1:
            return 'black'
        else:
            return 'white'

def get_education(x):
    if x['highest_education_of_parent']>=19:
        return 'Post Graduate Degree'
    elif x['highest_education_of_parent']==18:
            return 'Bachelors degree'
    elif x['highest_education_of_parent']>=15:
            return 'Some College'
    elif x['highest_education_of_parent']==14:
            return 'High School Diploma/GED'
    elif x['highest_education_of_parent']<=0:
            return 'No_answer'
    elif x['highest_education_of_parent']<=13:
            return 'Less_High School'
def get_ethnicity(x):
    if x['demo_ethn_v2']==1:
        return 'Hispanic/Latino'
    elif x['demo_ethn_v2']==2:
        return 'Not_Hispanic/Latino'
    elif x['demo_ethn_v2']=='NaN':
        return 'Unknown/Not Reported Eth'
    elif x['demo_ethn_v2']==999 or x['demo_ethn_v2']==777:
        return 'Unknown/Not Reported Eth'
    else:
        return 'Unknown/Not Reported Eth'

## 1= Less than $5,000; 2=$5,000 through $11,999; 3=$12,000 through $15,999; 4=$16,000 through $24,999; 5=$25,000 through $34,999; 6=$35,000 through $49,999; 7=$50,000 through $74,999; 8= $75,000 through $99,999; 9=$100,000 through $199,999; 10=$200,000 and greater.
def get_income(x):
    if x['demo_comb_income_v2'] == 999 or x['demo_comb_income_v2']==777:
        return 'Unknown/Not Reported income'
    elif x['demo_comb_income_v2'] == 1:
        return 'Less than $5000'
    elif x['demo_comb_income_v2'] == 2:
        return '$5000 through $11999'
    elif x['demo_comb_income_v2'] == 3:
        return '$12000 through $15999'
    elif x['demo_comb_income_v2'] == 4:
        return '$16000 through $24999'
    elif x['demo_comb_income_v2'] == 5:
        return '$25000 through $34999'
    elif x['demo_comb_income_v2'] == 6:
        return '$35000 through $49999'
    elif x['demo_comb_income_v2'] == 7:
        return '$50000 through $74999'
    elif x['demo_comb_income_v2'] == 8:
        return '$75000 through $99999'    
    elif x['demo_comb_income_v2'] == 9:
        return '$100000 through $199999'
    elif x['demo_comb_income_v2'] == 10:
        return '$200,000 and greater'