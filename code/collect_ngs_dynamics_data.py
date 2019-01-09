#
# Script to process NGS data for all players involved on punt plays. The
# primary purpose here is to aggregate a bunch of data (by player role on
# punts) to determine what might be considered "typical" speeds/accelerations.
#
# Author: Charlie Bonfield
# Last Modified: 1/2019

## IMPORTS
import os
import glob
import pandas as pd

from preprocess_small_data import load_data

## VARIABLES
DDIR = '/Users/cbonfield/Projects/kaggle/nfl_punts/data/wdynamics/'
ODIR = '/Users/cbonfield/Projects/kaggle/nfl_punts/data/sumdynamics/'
REL_COLS = ['Season_Year', 'GameKey', 'PlayID', 'GSISID']

## FUNCTIONS
def extract_summary_statistics(ngs_df):
    """
    Given a player's NGS data, extract a few summary statistics. We're really
    only interesting in speeds/accelerations here.

    Parameters:
        ngs_df: pd.DataFrame
            NGS data for a single player/play. Note that this should only contain
            data relevant to the punt/subsequent motion.
    """

    sum_dict = {}
    YD_TO_M = 0.9144 # multiplicative factor for converting yards to meters

    sum_dict['Season_Year'] = ngs_df.Season_Year.values[0]
    sum_dict['GameKey'] = ngs_df.GameKey.values[0]
    sum_dict['PlayID'] = ngs_df.PlayID.values[0]
    sum_dict['GSISID'] = ngs_df.GSISID.values[0]
    sum_dict['Role'] = ngs_df.Role.tolist()[0]

    sum_dict['max_vx'] = ngs_df.vx.max()*YD_TO_M
    sum_dict['max_vy'] = ngs_df.vy.max()*YD_TO_M
    sum_dict['max_s'] = ngs_df.s.max()*YD_TO_M
    sum_dict['max_ax'] = ngs_df.ax.max()*YD_TO_M
    sum_dict['max_ay'] = ngs_df.ay.max()*YD_TO_M
    sum_dict['max_a'] = ngs_df.a.max()*YD_TO_M

    return sum_dict


def strip_ngs_data(ngs_df):
    """
    Given a set of NGS data for a player/play, strip out all of the data that's
    not relevant (motion before snap/after whistle).

    Parameters:
        ngs_df: pd.DataFrame
            NGS data for a single player/play.
    """

    # Figure out where the ball snap occurred and get the index so that we can
    # discard all data prior to that instant.
    try:
        play_st = ngs_df.loc[ngs_df.Event == 'punt'].index[0]
    except IndexError:
        try:
            play_st = ngs_df.loc[ngs_df.Event == 'ball_snap'].index[0]
        except IndexError:
            play_st = ngs_df.index.min()

    # Figure out where the play "ended" so that we can discard all data after
    # that. For simplicity, we assume that any concussion event would have occured
    # prior to a penalty flag being thrown or within five seconds of a tackle.
    try:
        play_ei = ngs_df.loc[ngs_df.Event == 'penalty_flag'].index[0]
    except IndexError:
        try:
            play_ei = ngs_df.loc[ngs_df.Event == 'tackle'].index[0] + 50
            play_ps = ngs_df.loc[ngs_df.Event == 'play_submit'].index[0]

            while play_ei > play_ps:
                play_ei -= 10
        except IndexError:
            play_ei = ngs_df.index.max()

    # Slice out the data that we actually need.
    play_df = ngs_df.iloc[play_st:play_ei]
    play_df.reset_index(drop=True, inplace=True)

    return play_df



## MAIN
if __name__ == '__main__':

    # Load in all small datasets.
    data_dict = load_data()
    punt_role = data_dict['play_role']
    punt_role.rename(index=str, columns={'PlayId':'PlayID'}, inplace=True)

    # Load smallest set of data for processing.
    #FILE = 'NGS-2017-post.csv'
    files = glob.glob(f'{DDIR}*.csv')

    for file in files:
        ngs_data = pd.read_csv(file)

        # Stick player roles onto NGS data.
        ngs_data = ngs_data.merge(punt_role, how='inner', left_on=REL_COLS, right_on=REL_COLS)

        # Do a bumch of preprocessing to make our lives easier.
        player_df = ngs_data.drop_duplicates(REL_COLS).reset_index(drop=True)
        player_df = player_df.loc[:, REL_COLS]
        player_df.loc[:, 'playerIndex'] = player_df.index.values
        ngs_data = ngs_data.merge(player_df, how='inner', left_on=REL_COLS, right_on=REL_COLS)

        # Step through all players.
        ss_list = []

        for play_idx in range(ngs_data.playerIndex.max()):
            play_df = ngs_data.loc[ngs_data.playerIndex == play_idx].reset_index(drop=True)

            if 'fair_catch' in play_df.Event.unique():
                continue
            else:
                pass

            strip_play = strip_ngs_data(play_df)
            try:
                sumstats = extract_summary_statistics(strip_play)
                ss_list.append(sumstats)
            except:
                continue

        # Stick in a DataFrame, save.
        OFILE = os.path.basename(file).split('.csv')[0]+'-summary.csv'
        stats_df = pd.DataFrame(ss_list)
        stats_df.to_csv(f'{ODIR}{OFILE}', index=False)
