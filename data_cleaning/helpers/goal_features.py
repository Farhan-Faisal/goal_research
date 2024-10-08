import pandas as pd
import numpy as np
import os

def get_goal_labels(working_dir, file_name):
    run1_labels_path = os.path.join(working_dir, file_name)

    keep_cols = ['ParticipantIdentifier', 'ResultIdentifier', 'trial_date', 'label']
    
    run1_labels_long = pd.read_csv(run1_labels_path)[keep_cols].drop_duplicates(
        subset=['ParticipantIdentifier', 'ResultIdentifier', 'trial_date']
    )

    run1_labels_wide = run1_labels_long.pivot(
        values='label', 
        index=['ParticipantIdentifier', 'trial_date'], 
        columns='ResultIdentifier'
    ).reset_index().dropna()
    
    run1_labels_wide = run1_labels_wide.rename(columns={
        'DAILY_goal1_set': 'DAILY_goal1_label', 
        'DAILY_goal2_set': 'DAILY_goal2_label'
    })

    return run1_labels_wide