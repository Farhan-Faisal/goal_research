import pandas as pd
import numpy as np
import os


run1_col = ['ParticipantIdentifier',
 'trial_date',
 'DAILY_goal1_confidence',
 'DAILY_goal1_consequences',
 'DAILY_goal1_effort',
 'DAILY_goal1_importance',
 'DAILY_goal2_confidence',
 'DAILY_goal2_consequences',
 'DAILY_goal2_effort',
 'DAILY_goal2_importance',
 'DAILY_goal1_set',
 'DAILY_goal2_set',
 'DAILY_goal1_report',
 'DAILY_goal2_report',
 'DAILY_goal2_interaction_eachOther',
 'DAILY_goal1_motivationExternal',
 'DAILY_goal1_motivationInternal',
 'DAILY_goal2_motivationExternal',
 'DAILY_goal2_motivationInternal'
]

positive_evening_cols = ['affect_pos_amused', 'affect_pos_appreciated', 'affect_pos_excited', 'affect_pos_relaxedCalm',
       'affect_pos_focused', 'affect_pos_happy', 'affect_pos_hopeful', 'affect_pos_motivated']

negative_evening_cols = ['affect_neg_angry','affect_neg_ashamed', 'affect_neg_bored', 'affect_neg_depressed',
       'affect_neg_embarrassed', 'affect_neg_frustrated', 'affect_neg_guilty', 'affect_neg_lazy', 
       'affect_neg_lonelyIsolated', 'affect_neg_nervousAnxious', 'affect_neg_sad', 'affect_neg_stressed',]

def clean_run1_survey(working_dir, file_name):
    run1_self_report_path = os.path.join(working_dir, file_name)

    keep_cols = ['ParticipantIdentifier', 'ResultIdentifier', 'Answers', 'trial_date']
    run1_long = pd.read_csv(run1_self_report_path)[keep_cols].drop_duplicates(subset=[
        'ParticipantIdentifier', 
        'ResultIdentifier', 
        'trial_date'
        ])
    
    
    run1_wide = run1_long.pivot(
        values='Answers', 
        index=['ParticipantIdentifier', 'trial_date'], 
        columns='ResultIdentifier'
        ).reset_index()[run1_col]
    
    return run1_wide


def clean_run2_survey(working_dir, file_name):
    run2_self_report_path = os.path.join(working_dir, file_name)

    run2_self_report = pd.read_csv(run2_self_report_path)
    run2_self_report.columns = run2_self_report.columns.str.replace('^sr_', '', regex=True)
    run2_self_report = run2_self_report[run1_col]
    
    return run2_self_report

def get_evening_affect_run1(working_dir, file_name):
    run1_affect_path = os.path.join(working_dir, file_name)

    keep_cols = ['ParticipantIdentifier', 'ResultIdentifier', 'Answers', 'trial_date']
    run1_long = pd.read_csv(run1_affect_path)[keep_cols].drop_duplicates(subset=[
        'ParticipantIdentifier', 
        'ResultIdentifier', 
        'trial_date'
        ])
    
    return_cols = ["ParticipantIdentifier", "trial_date"]
    return_cols += positive_evening_cols + negative_evening_cols
    
    run1_affect = run1_long.pivot(
        values='Answers', 
        index=['ParticipantIdentifier', 'trial_date'], 
        columns='ResultIdentifier'
        ).reset_index()[return_cols]
    
    for col in positive_evening_cols + negative_evening_cols:
        run1_affect[col] = pd.to_numeric(run1_affect[col], errors='coerce')
    
    run1_affect['mean_pos_evening'] = run1_affect[positive_evening_cols].mean(axis=1)
    run1_affect['mean_neg_evening'] = run1_affect[negative_evening_cols].mean(axis=1)
    run1_affect['mean_affect_diff'] = run1_affect['mean_pos_evening'] - run1_affect['mean_neg_evening']
    
    return run1_affect


def get_evening_affect_run2(working_dir, file_name):
    run2_affect = pd.read_csv(os.path.join(working_dir, file_name))

    run2_affect['mean_pos_evening'] = run2_affect[positive_evening_cols].mean(axis=1)
    run2_affect['mean_neg_evening'] = run2_affect[negative_evening_cols].mean(axis=1)
    run2_affect['mean_affect_diff'] = run2_affect['mean_pos_evening'] - run2_affect['mean_neg_evening']

    return_cols = ["ParticipantIdentifier", "trial_date", "mean_pos_evening", "mean_neg_evening", "mean_affect_diff"]
    return_cols += positive_evening_cols + negative_evening_cols
    return run2_affect[return_cols]
