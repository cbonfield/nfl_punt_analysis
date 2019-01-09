#
# Script for trimming NGS data. Here, we pluck out NGS data only for the punt
# plays in the injury/control sets.
#
# Author: Charlie Bonfield
# Last Modified: 12/2018

## IMPORTS
import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocess_small_data import load_data


## VARIABLES
WDIR = '/Users/cbonfield/Projects/kaggle/nfl_punts/'
DDIR = f'{WDIR}data/wdynamics/'


## FUNCTIONS
def process_video_data(dict_key=None):
    """
    Clean up the injury data so that it's easy to pull out the NGS data for the
    players of interest.

    Parameters:
        dict_key: str (default None)
            Options: video_injury, video_control
    """

    ddict = load_data()
    vid_df = ddict[dict_key]

    # Pull out columns relevant for player.
    play_df = vid_df.loc[:, ['Season_Year', 'GameKey', 'PlayId', 'GSISID']]
    play_df.loc[:, 'Identifier'] = 'PLAYER'

    # Pull out columns relevant for primary partner.
    part_df = vid_df.loc[:, ['Season_Year', 'GameKey', 'PlayId', 'Primary_Partner_GSISID']]
    part_df.rename(index=str, columns={'Primary_Partner_GSISID':'GSISID'}, inplace=True)
    part_df.loc[:, 'Identifier'] = 'PARTNER'

    # Concatenate, sort, return.
    both_df = pd.concat([play_df, part_df], ignore_index=True)
    both_df.sort_values(by=['Season_Year', 'GameKey', 'PlayId', 'Identifier'], inplace=True)
    both_df.reset_index(drop=True, inplace=True)

    return both_df


## MAIN
if __name__ == '__main__':

    # Load in smaller datasets.
    inj_df = process_video_data(dict_key='video_injury')
    inj_df.rename(index=str, columns={'PlayId':'PlayID'}, inplace=True)

    # Step through entire set of NGS data.
    ngs_dfs = []
    mer_cols = ['Season_Year', 'GameKey', 'PlayID', 'GSISID']
    ngs_files = glob.glob(f'{DDIR}*.csv')

    for nfil in ngs_files:
        print(os.path.basename(nfil))
        ngs_df = pd.read_csv(nfil)
        tmp_df = inj_df.merge(ngs_df, how='inner', left_on=mer_cols, right_on=mer_cols)
        ngs_dfs.append(tmp_df)

    # Save dataset.
    out_df = pd.concat(ngs_dfs, ignore_index=True)
    out_df.sort_values(by=mer_cols, inplace=True)
    out_df.reset_index(drop=True, inplace=True)

    out_df.to_csv(f'{WDIR}injury_ngs_data.csv', index=False)

    # Load in set of NGS data for testing.
    #data = pd.read_csv(f'{DDIR}NGS-2017-pre.csv')
    #mer_cols = ['Season_Year', 'GameKey', 'PlayID', 'GSISID']
    #test = inj_df.merge(data, how='inner', left_on=mer_cols, right_on=mer_cols)
