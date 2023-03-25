"""


"""
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import metrics
from craig.lazy_greedy import *

def stress() -> pd.DataFrame:
    stress_features = ['subjectkey', 'pstr_unable_control_cv', 'pstr_confidence_p_cv',
                        'pstr_way_p_cv', 'pstr_overcome_p_cv']

    stress_time_patterns = []

    for i in range(6):
        # Iteratively read in the .csv files as temporary dataframes
        readin = pd.read_csv('/Users/brucequ/Documents/BigML/Project_ABCD_Clustering/Data/abcd_covid_s%d.csv'\
        % (i+1))
        
        # Extract the wanted columns
        stress_columns = readin[stress_features]
        
        # Append a single timestamp to the list
        stress_time_patterns.append(stress_columns)

    stress_data = stress_time_patterns[0]

    # Iteratively merge the dataframes
    for i in range(len(stress_time_patterns)-1):
        stress_data = stress_data.merge(stress_time_patterns[i+1], on='subjectkey',
                                        how='outer',
                                        suffixes=('_%d' % (i+1), '_%d' % (i+2)))

    # Stress Data imputation
    stress_data = stress_data.dropna(thresh=12).reset_index(drop=True)
    imp_mean = IterativeImputer(random_state=0, max_iter=100)
    temp_cols = stress_data.drop(["subjectkey"], axis=1).columns
    temp_sub_keys = stress_data["subjectkey"].copy().reset_index(drop=True)
    stress_data = pd.DataFrame(imp_mean.fit_transform(stress_data.drop(["subjectkey"], axis=1)).round(decimals=0), columns=temp_cols)
    stress_data.insert(0, "subjectkey", temp_sub_keys)

    # Calculate total stress score
    for i in range(0, 6):
        stress_data['total_stress_score_%d' % (i+1)] = stress_data['pstr_unable_control_cv_%d' % (i+1)] + np.abs(stress_data['pstr_confidence_p_cv_%d' % (i+1)]-4) 
        + np.abs(stress_data['pstr_way_p_cv_%d' % (i+1)]-4) + stress_data['pstr_overcome_p_cv_%d' % (i+1)]
        
    # Keep total score columns
    stress_data = stress_data[["subjectkey", 'total_stress_score_1', 'total_stress_score_2', 'total_stress_score_3',
                                'total_stress_score_4', 'total_stress_score_5', 'total_stress_score_6']]
    return stress_data

def stress_cluster(stress_data:pd.DataFrame, cluster_id:int, num_centers=15) -> list:
    X = stress_data.drop(['subjectkey'], axis=1).to_numpy()

    # D is the similarity matrix between each two 
    D = metrics.pairwise.euclidean_distances(X, X)

    # Define the similarity score
    S = np.max(D) - D
    S_stress = S.copy()

    n = S.shape[0]
    V = []
    for i in range(0, n):
        V.append(i)

    # Create a FacilityLocation Object
    F = FacilityLocation(S, V)

    # Initialize the number of centers
    B = num_centers

    sset, vals = lazy_greedy_heap(F, V, B)
    sset_stress = sset.copy()

    reduced_s_matrix = S_stress[sset_stress, :]
    target_matrix = reduced_s_matrix.argmax(axis=0)
    cluster_points_ids = np.where(target_matrix == cluster_id - 1)[0]
    center_key = stress_data["subjectkey"].loc[cluster_points_ids]


    def get_stress_cluster(cluster_id, S_stress=S_stress, sset_stress=sset_stress):

        reduced_s_matrix = S_stress[sset_stress, :]
        target_matrix = reduced_s_matrix.argmax(axis=0)

        cluster_points_ids = np.where(target_matrix == cluster_id - 1)[0]

        center_key = stress_data["subjectkey"].loc[cluster_points_ids]

        return center_key.tolist()

    interesting_stress_clusters = [18, 6]
    overall_stress_extreme = []

    for i in interesting_stress_clusters:
        overall_stress_extreme = overall_stress_extreme + get_stress_cluster(i)

    low_stress_clusters = [9, 4]
    overall_low_stress = []
    for j in low_stress_clusters:
        overall_low_stress = overall_low_stress + get_stress_cluster(j)


    all_stress_subkey = stress_data['subjectkey']
    high_stress_subkey = overall_stress_extreme
    low_stress_subkey = overall_low_stress

    return [all_stress_subkey, high_stress_subkey, low_stress_subkey]
