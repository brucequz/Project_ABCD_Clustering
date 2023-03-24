import pandas as pd

def mri_sdp(all_stress_subkey) -> pd.DataFrame:
    mri_sdp = pd.read_csv("/home/bruceq/ABCD/Research/Data/sMRI/abcd_mrisdp10201.txt", delimiter='\t', skiprows=(1,1))
    mri_sdp = mri_sdp.loc[:, "subjectkey":"mrisdp_151"]
    mri_sdp = mri_sdp.loc[:, "subjectkey":"mrisdp_151"].drop(["src_subject_id"], axis=1)
    mri_sdp = pd.get_dummies(mri_sdp, columns=['sex'])
    mri_sdp = mri_sdp.loc[mri_sdp['subjectkey'].isin(all_stress_subkey)].reset_index(drop=True)

    mri_sdp['interview_date'] = pd.to_datetime(mri_sdp['interview_date'], format = '%m/%d/%Y', errors='coerce')
    late_list = list(mri_sdp.loc[mri_sdp.interview_date >= pd.Timestamp(2020, 3, 1)].subjectkey)

    mri_sdp_1 = mri_sdp.loc[mri_sdp['eventname'] == 'baseline_year_1_arm_1'].drop(['eventname'], axis=1).reset_index(drop=True)
    mri_sdp_1 = mri_sdp_1.loc[mri_sdp_1.subjectkey.isin(late_list)]
    mri_sdp_2 = mri_sdp.loc[mri_sdp['eventname'] == '2_year_follow_up_y_arm_1'].drop(['eventname'], axis=1).reset_index(drop=True)
    mri_sdp_2 = mri_sdp_2.loc[mri_sdp_2.interview_date <= pd.Timestamp(2020, 3, 1)]


    mri_sdp_mean_1 = mri_sdp_1.mean(numeric_only=True)
    mri_sdp_mean_2 = mri_sdp_2.mean(numeric_only=True)
    mri_sdp_1 = mri_sdp_1.fillna(mri_sdp_mean_1)
    mri_sdp_2 = mri_sdp_2.fillna(mri_sdp_mean_2)
    mri_sdp_2 = mri_sdp_2.drop(['interview_date'], axis=1)
    mri_sdp_2 = mri_sdp_2.set_index('subjectkey')
    
    return mri_sdp_2