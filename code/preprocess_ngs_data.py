#
# Script for preprocessing NGS data.
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

## VARIABLES
WDIR = '/Users/cbonfield/Dropbox/nfl_punts/raw/'
ODIR = '/Users/cbonfield/Projects/kaggle/nfl_punts/data/wdynamics/'

## FUNCTIONS
def compute_dynamics(data):
    """
    Compute velocity/acceleration given the NGS data (we have x and y positions
    as well as time).

    Parameters:
        data: pd.DataFrame
            DataFrame containing NGS data.
    """

    MER_COLS = ['Season_Year', 'GameKey', 'PlayID', 'GSISID']

    # Calculate time/position differences to estimate velocity/speed.
    grp_df = data.groupby(MER_COLS).apply(lambda x: x.sort_values('t').reset_index(drop=True))
    x_diff = data.groupby(MER_COLS).apply(lambda x: x.sort_values('t').reset_index(drop=True)).x.diff()
    y_diff = data.groupby(MER_COLS).apply(lambda x: x.sort_values('t').reset_index(drop=True)).y.diff()
    t_diff = data.groupby(MER_COLS).apply(lambda x: x.sort_values('t').reset_index(drop=True)).t.diff()

    # Tack on dx, dy, and dt (adding column in place works better than an actual
    # merge here).
    grp_df = grp_df.reset_index(drop=True)
    grp_df.loc[:, 'dx'] = x_diff.values
    grp_df.loc[:, 'dy'] = y_diff.values
    grp_df.loc[:, 'dt'] = t_diff.values
    #grp_df.fillna(0.0, inplace=True)

    # Add vx(t), vy(t) to grp_df.
    grp_df.loc[:, 'vx'] = grp_df.dx / grp_df.dt
    grp_df.loc[:, 'vy'] = grp_df.dy / grp_df.dt
    grp_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    #grp_df.fillna(0.0, inplace=True)

    # Add speed (sqrt(vx^2+vy^2)) as a function of time, too.
    vx = grp_df.vx.values
    vy = grp_df.vy.values
    speed = np.sqrt(np.square(vx)+np.square(vy))
    grp_df.loc[:, 's'] = speed

    # Calculate velocity differences to estimate acceleration.
    dvx_diff = grp_df.groupby(MER_COLS).apply(lambda x: x.sort_values('t').reset_index(drop=True)).vx.diff()
    dvy_diff = grp_df.groupby(MER_COLS).apply(lambda x: x.sort_values('t').reset_index(drop=True)).vy.diff()

    # Add ax(t), ay(t) to grp_df.
    grp_df.loc[:, 'dvx'] = dvx_diff.values
    grp_df.loc[:, 'dvy'] = dvy_diff.values
    grp_df.loc[:, 'ax'] = grp_df.dvx / grp_df.dt
    grp_df.loc[:, 'ay'] = grp_df.dvy / grp_df.dt
    grp_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Add magnitude of acceleration, too.
    ax = grp_df.ax.values
    ay = grp_df.ay.values
    acc_mag = np.sqrt(np.square(ax)+np.square(ay))
    grp_df.loc[:, 'a'] = acc_mag

    # Drop differential columns.
    diff_cols = ['dx', 'dy', 'dt', 'dvx', 'dvy']
    grp_df.drop(diff_cols, axis=1, inplace=True)

    return grp_df

def get_relative_times(file_name):
    """
    Given the name of an NGS dataset, compute relative time (with respect to the
    start of the play).

    Parameters:
        file_name: str
            Name of NGS dataset.
    """

    # Load in NGS dataset.
    ngs_df = pd.read_csv(f'{WDIR}{file_name}')

    # Get the play start times.
    play_starts = ngs_df.groupby(['Season_Year', 'GameKey', 'PlayID']).agg({'Time':'min'})
    play_starts.reset_index(inplace=True)
    play_starts.rename(index=str, columns={'Time':'Play_StartTime'}, inplace=True)

    # Stick the start times back on the rest of the NGS data.
    ngs_df = ngs_df.merge(play_starts, how='outer',
                          left_on=['Season_Year', 'GameKey', 'PlayID'],
                          right_on=['Season_Year', 'GameKey', 'PlayID'])
    ngs_df.loc[:, 'Time'] = pd.to_datetime(ngs_df.Time)
    ngs_df.loc[:, 'Play_StartTime'] = pd.to_datetime(ngs_df.Play_StartTime)
    ngs_df.loc[:, 'Relative_Time'] = ngs_df.Time - ngs_df.Play_StartTime
    ngs_df.loc[:, 't'] = [x.total_seconds() for x in ngs_df.Relative_Time.tolist()]
    ngs_df.drop(['Play_StartTime', 'Relative_Time'], axis=1, inplace=True)

    return ngs_df

if __name__ == '__main__':

    # Load in smallest set of NGS data for testing.
    FILE = 'NGS-2017-post.csv'
    fs_data = get_relative_times(FILE)
    ss_data = compute_dynamics(fs_data)

    # Step through all NGS data, adding velocity/acceleration.
    #files = glob.glob(f'{WDIR}NGS*.csv')
    #files = [os.path.basename(x) for x in files]

    #for fn in files:
    #    print(fn)
    #    fs_data = get_relative_times(fn)
    #    ss_data = compute_dynamics(fs_data)

    #    ss_data.to_csv(f'{ODIR}{fn}', index=False)
    #    print('Done!')
