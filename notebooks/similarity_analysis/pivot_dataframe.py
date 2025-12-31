import pandas as pd
import numpy as np

# STEP 1: Pivot df1 to get one row per participant-date with both today goals
def pivot_today_goals(df1):
    """
    Pivot df1 so we have one row per participant-date with both today's goals
    """
    # Separate goal_1 and goal_2 rows
    goal_1_rows = df1[df1['Identifier'] == 'goal_1'].copy()
    goal_2_rows = df1[df1['Identifier'] == 'goal_2'].copy()
    
    # Rename columns for goal_1
    goal_1_rows = goal_1_rows.rename(columns={
        'today_goal': 'today_goal_1',
        'sim_with_tom_1': 'goal1_sim_with_tom1',
        'sim_with_tom_2': 'goal1_sim_with_tom2',
        'same_as_tomorrow_1': 'openai_goal1_same_as_tom1',
        'same_as_tomorrow_2': 'openai_goal1_same_as_tom2'
    })
    
    # Rename columns for goal_2
    goal_2_rows = goal_2_rows.rename(columns={
        'today_goal': 'today_goal_2',
        'sim_with_tom_1': 'goal2_sim_with_tom1',
        'sim_with_tom_2': 'goal2_sim_with_tom2',
        'same_as_tomorrow_1': 'openai_goal2_same_as_tom1',
        'same_as_tomorrow_2': 'openai_goal2_same_as_tom2'
    })
    
    # Merge them together
    df1_pivot = goal_1_rows[['ParticipantIdentifier', 'trial_date', 
                              'today_goal_1', 'tomorrow_goal_1', 'tomorrow_goal_2',
                              'goal1_sim_with_tom1', 'goal1_sim_with_tom2',
                              'openai_goal1_same_as_tom1', 'openai_goal1_same_as_tom2']].merge(
        goal_2_rows[['ParticipantIdentifier', 'trial_date', 
                     'today_goal_2', 'goal2_sim_with_tom1', 'goal2_sim_with_tom2',
                     'openai_goal2_same_as_tom1', 'openai_goal2_same_as_tom2']],
        on=['ParticipantIdentifier', 'trial_date'],
        how='outer'
    )
    
    return df1_pivot


# STEP 2: Pivot df2 to get one row per participant-date with both goals' metrics
def pivot_today_metrics(df2):
    """
    df2 already has one row per participant-date with both goals, so just rename for clarity
    """
    df2_renamed = df2.rename(columns={
        'DAILY_goal1_set': 'today_goal_1_set',
        'DAILY_goal2_set': 'today_goal_2_set',
        'DAILY_goal1_report': 'today_completion_1',
        'DAILY_goal2_report': 'today_completion_2',
        'DAILY_goal1_effort': 'today_effort_1',
        'DAILY_goal2_effort': 'today_effort_2',
        'DAILY_goal1_importance': 'today_importance_1',
        'DAILY_goal2_importance': 'today_importance_2'
    })
    
    return df2_renamed


# STEP 3: Expand to have 2 rows per day - one for each tomorrow goal
def expand_to_tomorrow_goals(df_merged):
    """
    Create 2 rows per participant-date: one for tomorrow_goal_1, one for tomorrow_goal_2
    """
    rows = []
    
    for _, row in df_merged.iterrows():
        # Row 1: For tomorrow_goal_1
        row1 = {
            'ParticipantIdentifier': row['ParticipantIdentifier'],
            'trial_date': row['trial_date'],
            'tomorrow_goal': row['tomorrow_goal_1'],
            'today_goal_1': row['today_goal_1'],
            'today_goal_2': row['today_goal_2'],
            'today_completion_1': row['today_completion_1'],
            'today_completion_2': row['today_completion_2'],
            'today_effort_1': row['today_effort_1'],
            'today_effort_2': row['today_effort_2'],
            'today_importance_1': row['today_importance_1'],
            'today_importance_2': row['today_importance_2'],
            'sim_tomorrow_with_today1': row['goal1_sim_with_tom1'],
            'sim_tomorrow_with_today2': row['goal2_sim_with_tom1'],
            'openai_same_as_tomorrow_today1': row['openai_goal1_same_as_tom1'],
            'openai_same_as_tomorrow_today2': row['openai_goal2_same_as_tom1'],
            'which_tomorrow_goal': 'goal_1'
        }
        rows.append(row1)
        
        # Row 2: For tomorrow_goal_2
        row2 = {
            'ParticipantIdentifier': row['ParticipantIdentifier'],
            'trial_date': row['trial_date'],
            'tomorrow_goal': row['tomorrow_goal_2'],
            'today_goal_1': row['today_goal_1'],
            'today_goal_2': row['today_goal_2'],
            'today_completion_1': row['today_completion_1'],
            'today_completion_2': row['today_completion_2'],
            'today_effort_1': row['today_effort_1'],
            'today_effort_2': row['today_effort_2'],
            'today_importance_1': row['today_importance_1'],
            'today_importance_2': row['today_importance_2'],
            'sim_tomorrow_with_today1': row['goal1_sim_with_tom2'],
            'sim_tomorrow_with_today2': row['goal2_sim_with_tom2'],
            'openai_same_as_tomorrow_today1': row['openai_goal1_same_as_tom2'],
            'openai_same_as_tomorrow_today2': row['openai_goal2_same_as_tom2'],
            'which_tomorrow_goal': 'goal_2'
        }
        rows.append(row2)
    
    return pd.DataFrame(rows)


# STEP 4: Add max similarity and corresponding completion
def add_max_similarity_features(df):
    """
    Add max_sim and the completion/effort/importance of the most similar goal
    """
    df = df.copy()
    
    # Max similarity
    df['max_sim_tomorrow_with_today'] = df[
        ['sim_tomorrow_with_today1', 'sim_tomorrow_with_today2']
    ].max(axis=1)
    
    # Which goal was most similar?
    df['most_similar_is_goal1'] = (
        df['sim_tomorrow_with_today1'] >= df['sim_tomorrow_with_today2']
    )
    
    # Same as tomorrow for the most similar goal
    df['max_sim_same_as_tomorrow'] = np.where(
        df['most_similar_is_goal1'],
        df['openai_same_as_tomorrow_today1'],
        df['openai_same_as_tomorrow_today2']
    )
    
    # Completion of most similar goal
    df['max_sim_goal_completion_today'] = np.where(
        df['most_similar_is_goal1'],
        df['today_completion_1'],
        df['today_completion_2']
    )
    
    # Effort of most similar goal
    df['max_sim_goal_effort_today'] = np.where(
        df['most_similar_is_goal1'],
        df['today_effort_1'],
        df['today_effort_2']
    )
    
    # Importance of most similar goal
    df['max_sim_goal_importance_today'] = np.where(
        df['most_similar_is_goal1'],
        df['today_importance_1'],
        df['today_importance_2']
    )
    
    return df


# STEP 5: Add tomorrow's completion (shift data by 1 day)
def add_tomorrow_completion(df, df2_original):
    """
    Add tomorrow's goal completion by matching with next day's data
    
    Logic:
    - df has rows for TODAY (trial_date) trying to predict TOMORROW's goals
    - We need to get TOMORROW's actual completion from the NEXT day's data
    - We shift df2's dates back by 1 day, so when we merge on the same date,
      we're actually getting tomorrow's data
    
    Example:
    - df row: trial_date = Feb 1, tomorrow_goal = "study bio"
    - df2_original: trial_date = Feb 2, DAILY_goal1_report = 75
    - df2_tomorrow shifted: trial_date = Feb 1, DAILY_goal1_report = 75
    - After merge: Feb 1 row gets Feb 2's completion (75)
    
    IMPORTANT: We use df2_ORIGINAL (not df2_pivot) because we need the 
    DAILY_goal1_report column names, not today_completion_1 which is already in df
    """
    df = df.copy()
    df['trial_date'] = pd.to_datetime(df['trial_date'])
    
    # Create tomorrow's data (shift dates back by 1 day)
    # Use ORIGINAL df2 column names to avoid conflicts
    df2_tomorrow = df2_original.copy()
    df2_tomorrow['trial_date'] = pd.to_datetime(df2_tomorrow['trial_date']) - pd.Timedelta(days=1)
    
    # Merge to get tomorrow's metrics
    # Use ORIGINAL column names from df2 (DAILY_goal1_report, not today_completion_1)
    df = df.merge(
        df2_tomorrow[['ParticipantIdentifier', 'trial_date', 
                      'DAILY_goal1_report', 'DAILY_goal2_report',
                      'DAILY_goal1_effort', 'DAILY_goal2_effort',
                      'DAILY_goal1_importance', 'DAILY_goal2_importance']],
        on=['ParticipantIdentifier', 'trial_date'],
        how='left'
    )
    
    # Assign tomorrow's completion based on which_tomorrow_goal
    # which_tomorrow_goal tells us if we're predicting goal_1 or goal_2 for tomorrow
    df['tomorrow_goal_completion'] = np.where(
        df['which_tomorrow_goal'] == 'goal_1',
        df['DAILY_goal1_report'],
        df['DAILY_goal2_report']
    )
    
    df['tomorrow_goal_effort'] = np.where(
        df['which_tomorrow_goal'] == 'goal_1',
        df['DAILY_goal1_effort'],
        df['DAILY_goal2_effort']
    )
    
    df['tomorrow_goal_importance'] = np.where(
        df['which_tomorrow_goal'] == 'goal_1',
        df['DAILY_goal1_importance'],
        df['DAILY_goal2_importance']
    )
    
    # Drop the intermediate columns
    df = df.drop(columns=['DAILY_goal1_report', 'DAILY_goal2_report',
                          'DAILY_goal1_effort', 'DAILY_goal2_effort',
                          'DAILY_goal1_importance', 'DAILY_goal2_importance'])
    
    return df


# MAIN FUNCTION: Put it all together
def restructure_goal_data(df1, df2):
    """
    Complete restructuring pipeline
    
    Parameters:
    -----------
    df1 : DataFrame with goal text and similarities
    df2 : DataFrame with completion/effort/importance (ORIGINAL, not renamed)
    
    Returns:
    --------
    df_final : Restructured dataframe with 2 rows per day per participant
    """
    print("Step 1: Pivoting df1 to get both today goals per row...")
    df1_pivot = pivot_today_goals(df1)
    print(f"  Shape after pivot: {df1_pivot.shape}")
    
    print("\nStep 2: Renaming df2 columns...")
    df2_pivot = pivot_today_metrics(df2)
    print(f"  Shape: {df2_pivot.shape}")
    
    print("\nStep 3: Merging df1 and df2...")
    df_merged = df1_pivot.merge(
        df2_pivot[['ParticipantIdentifier', 'trial_date',
                   'today_completion_1', 'today_completion_2',
                   'today_effort_1', 'today_effort_2',
                   'today_importance_1', 'today_importance_2']],
        on=['ParticipantIdentifier', 'trial_date'],
        how='inner'
    )
    print(f"  Shape after merge: {df_merged.shape}")
    
    print("\nStep 4: Expanding to 2 rows per day (one per tomorrow goal)...")
    df_expanded = expand_to_tomorrow_goals(df_merged)
    print(f"  Shape after expansion: {df_expanded.shape}")
    
    print("\nStep 5: Adding max similarity features...")
    df_with_max = add_max_similarity_features(df_expanded)
    
    print("\nStep 6: Adding tomorrow's completion...")
    # IMPORTANT: Pass ORIGINAL df2, not df2_pivot!
    df_final = add_tomorrow_completion(df_with_max, df2)
    print(f"  Final shape: {df_final.shape}")
    
    # Remove rows where tomorrow_goal_completion is missing (last day of study)
    df_final = df_final.dropna(subset=['tomorrow_goal_completion'])
    print(f"  Shape after removing missing tomorrow completions: {df_final.shape}")
    
    return df_final


# Usage:
# df_new = restructure_goal_data(df1, df2)