import pandas as pd
from datetime import timedelta

def print_participant_general_states(df):
    """
    Prints general statistics about participants in the dataframe.
    """
    num_rows = len(df)

    # Number of unique participants
    num_participants = df['ParticipantIdentifier'].nunique()

    # Average number of entries per participant
    avg_entries = df.groupby('ParticipantIdentifier').size().mean()

    print(f"Total rows: {num_rows}")
    print(f"Unique participants: {num_participants}")
    print(f"Average entries per participant: {avg_entries:.2f}")


def filter_for_participant_counts(df, count):
    '''
    Filters a dataframe to only include participants with more than `count` entries.
    '''

    counts = df['ParticipantIdentifier'].value_counts()

    participants_gt = counts[counts > count].reset_index()['ParticipantIdentifier'].to_list()

    df = df[df['ParticipantIdentifier'].isin(participants_gt)].copy().dropna()

    print_participant_general_states(df)

    return df



def filter_for_regularity(df, day_mean, day_var):
    """
    Filters a dataframe to only include participants with mean day difference <= day_mean
    and variance of day difference <= day_var.
    """
    gap_stats = (
        df.groupby("ParticipantIdentifier")["day_diff"]
        .agg(["mean", "var", "max", "count"])
        .reset_index()
    )

    filtered_ids = gap_stats.loc[
        (gap_stats["mean"] <= day_mean) & (gap_stats["var"] <= day_var),
        "ParticipantIdentifier"
    ]

    df = df[df["ParticipantIdentifier"].isin(filtered_ids)].copy().dropna()

    print_participant_general_states(df)

    return df



def filter_for_goal_str_length(df, min_length):
    """
    Filters a dataframe to only include participants whose goal strings have at least `min_length` words.
    """

    df = df[
        (df['DAILY_goal1_set'].str.split().str.len() >= min_length) |
        (df['DAILY_goal2_set'].str.split().str.len() >= min_length)
    ].copy().dropna()

    print_participant_general_states(df)
        
    return df


def align_goal_with_day_they_were_done(df):
    """
    Aligns goals set on one day with the effort, importance, and completion recorded the next day.
    """

    goal_cols = [
        "ParticipantIdentifier", 
        "trial_date",
        "DAILY_goal1_set",
        "DAILY_goal2_set",
    ]

    other_cols = [
        "ParticipantIdentifier", 
        "trial_date",
        "DAILY_goal1_effort",
        "DAILY_goal1_importance",
        "DAILY_goal2_effort",
        "DAILY_goal2_importance",
        "DAILY_goal1_report",
        "DAILY_goal2_report",
        "DAILY_goal1_effort",
        "DAILY_goal2_effort",
        "day_diff"
    ]

    goal_temp = df[goal_cols].copy()
    otherDF = df[other_cols].copy()

    goal_temp["trial_date"] = pd.to_datetime(goal_temp["trial_date"]) + timedelta(days=1)
    otherDF["trial_date"] = pd.to_datetime(otherDF["trial_date"])
    mergedDF = pd.merge(
        goal_temp, 
        otherDF, 
        on=["ParticipantIdentifier", "trial_date"], 
        how="inner"
    ).dropna()

    print_participant_general_states(mergedDF)

    return mergedDF