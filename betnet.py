import pandas as pd


def betnet(all_stress_subkey) -> pd.DataFrame:
    betnet = pd.read_csv("/home/bruceq/ABCD/Research/Data/abcd_betnet02.txt", delimiter='\t', skiprows=(1,1))
    col = list(betnet.loc[:, "rsfmri_c_ngd_ad_ngd_ad":"rsfmri_c_ngd_vs_ngd_vs"].columns)
    col.append("subjectkey")
    col.extend(["sex", "interview_age", "eventname", "interview_date"])


    betnet = betnet[col]
    betnet = pd.get_dummies(betnet, columns=['sex'])

    betnet_mean = betnet.mean(numeric_only=True)
    betnet = betnet.fillna(betnet_mean)

    betnet['interview_date'] = pd.to_datetime(betnet['interview_date'], format = '%m/%d/%Y', errors='coerce')
    late_list = list(betnet.loc[betnet.interview_date >= pd.Timestamp(2020, 3, 1)].subjectkey)

    betnet_1 = betnet.loc[betnet['eventname'] == 'baseline_year_1_arm_1'].drop(['eventname'], axis=1).reset_index(drop=True)
    betnet_1 = betnet_1.loc[betnet_1.subjectkey.isin(late_list)]
    betnet_2 = betnet.loc[betnet['eventname'] == '2_year_follow_up_y_arm_1'].drop(['eventname'], axis=1).reset_index(drop=True)
    betnet_2 = betnet_2.loc[betnet_2.interview_date <= pd.Timestamp(2020, 3, 1)] 

    betnet_1 = betnet_1.loc[betnet_1['subjectkey'].isin(all_stress_subkey)].reset_index(drop=True)
    betnet_2 = betnet_2.loc[betnet_2['subjectkey'].isin(all_stress_subkey)].reset_index(drop=True)

    # betnet_2 = pd.concat([betnet_1, betnet_2])

    betnet_2 = betnet_2.drop(['rsfmri_c_ngd_ad_ngd_n', 'rsfmri_c_ngd_cgc_ngd_n', 'rsfmri_c_ngd_ca_ngd_n', 'rsfmri_c_ngd_dt_ngd_n', 'rsfmri_c_ngd_dla_ngd_n', 'rsfmri_c_ngd_fo_ngd_n', 'rsfmri_c_ngd_n_ngd_ad', 'rsfmri_c_ngd_n_ngd_cgc', 'rsfmri_c_ngd_n_ngd_ca', 'rsfmri_c_ngd_n_ngd_dt', 'rsfmri_c_ngd_n_ngd_dla', 
    'rsfmri_c_ngd_n_ngd_fo', 'rsfmri_c_ngd_n_ngd_n', 'rsfmri_c_ngd_n_ngd_rspltp', 'rsfmri_c_ngd_n_ngd_smh', 'rsfmri_c_ngd_n_ngd_smm', 'rsfmri_c_ngd_n_ngd_sa', 'rsfmri_c_ngd_n_ngd_vta', 'rsfmri_c_ngd_n_ngd_vs', 'rsfmri_c_ngd_rspltp_ngd_n', 'rsfmri_c_ngd_smh_ngd_n',
    'rsfmri_c_ngd_smm_ngd_n', 'rsfmri_c_ngd_sa_ngd_n', 'rsfmri_c_ngd_vta_ngd_n', 'rsfmri_c_ngd_vs_ngd_n', 'interview_date'], axis=1)

    betnet_2 = betnet_2.set_index('subjectkey')

    return betnet_2