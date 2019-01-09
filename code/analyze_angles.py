#
# This script is to be used to explore angles (orientation, direction) from the
# NGS data.
#
# Author: Charlie Bonfield
# Last Modified: 1/2019

## IMPORTS
import numpy as np
import pandas as pd

import plotly.io as pio
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

pd.set_option('display.max_rows', 5000)

## VARIABLES
WDIR = '/Users/cbonfield/Projects/kaggle/nfl_punts/data/'
YD_TO_M = 0.9144 # multiplicative factor for converting yards to meters


## FUNCTIONS
def calculate_pp_distance(ngs_df):
    """
    Given the NGS data for the injury set, calculate player-partner distance.

    Parameters:
        ngs_df: pd.DataFrame
            DataFrame containing NGS data for player/partner on plays with a
            concussion.
    """

    # Split player/partner data.
    play_df = ngs_df.loc[ngs_df.Identifier == 'PLAYER']
    part_df = ngs_df.loc[ngs_df.Identifier == 'PARTNER']
    play_df = play_df.drop(['Identifier'], axis=1)
    part_df = part_df.drop(['Identifier'], axis=1)

    # Rename columns for ease of merge.
    ren_cols = ['GSISID', 'x', 'y', 'dis', 'o', 'dir', 'vx', 'vy', 's', 'ax',
                'ay', 'a', 't']
    play_cols = {x:f'play_{x}' for x in ren_cols}
    part_cols = {x:f'part_{x}' for x in ren_cols}
    play_df.rename(index=str, columns=play_cols, inplace=True)
    part_df.rename(index=str, columns=part_cols, inplace=True)

    # Perform merge.
    mer_cols = ['Season_Year', 'GameKey', 'PlayID', 'Event',
                'eventIndex', 'Time']
    pp_df = play_df.merge(part_df, how='inner', left_on=mer_cols, right_on=mer_cols)

    # Add extra columns that will assist with determining when tackle was made.
    pp_df.loc[:, 'diff_x'] = pp_df.part_x - pp_df.play_x
    pp_df.loc[:, 'diff_y'] = pp_df.part_y - pp_df.play_y
    pp_df.loc[:, 'pp_dis'] = np.sqrt(np.square(pp_df.diff_x.values) + np.square(pp_df.diff_y.values))

    return pp_df

def find_impact(pp_df):
    """
    Given a DataFrame containing player-parter NGS data (for a single play),
    identify the most likely time of impact and return all data from one second
    around that time of impact.

    Parameters:
        pp_df: pd.DataFrame
            NGS data for player/partner pair.
    """

    # Figure out where the ball snap occurred and get the index so that we can
    # discard all data prior to that instant.
    try:
        play_st = pp_df.loc[pp_df.Event == 'punt'].index[0]
    except IndexError:
        try:
            play_st = pp_df.loc[pp_df.Event == 'ball_snap'].index[0]
        except IndexError:
            play_st = pp_df.index.min()

    # Figure out where the play "ended" so that we can discard all data after
    # that. For simplicity, we assume that any concussion event would have occured
    # prior to a penalty flag being thrown or within five seconds of a tackle.
    try:
        play_ei = pp_df.loc[pp_df.Event == 'penalty_flag'].index[0]
    except IndexError:
        try:
            play_ei = pp_df.loc[pp_df.Event == 'tackle'].index[0] + 50

            play_ps = pp_df.loc[pp_df.Event == 'play_submit'].index[0]

            while play_ei > play_ps:
                play_ei -= 10

        except IndexError:
            play_ei = pp_df.index.max()

    # Slice out the data that we actually need.
    pp_df = pp_df.iloc[play_st:play_ei].reset_index(drop=True)

    # Find the DataFrame index corresponding to the minimum player-partner
    # distance.
    min_dis_index = int(pp_df.loc[pp_df.pp_dis == pp_df.pp_dis.min()].index[0])

    # Grab a few rows around that index.
    min_dis_df = pp_df.iloc[(min_dis_index-5):(min_dis_index+5)]
    min_dis_df.loc[min_dis_index,'impact'] = 1
    min_dis_df.fillna(0, inplace=True)
    min_dis_df.reset_index(drop=True, inplace=True)

    return min_dis_df

def make_radial_plot(plot_df, angle_opt):
    """
    Make a radial plot using a DataFrame built specifically for this purpose
    (i.e., expected column names are hard-coded).

    Parameters:
        plot_df: pd.DataFrame
            DataFrame containing data to be plotted.
        angle_opt: str
            Angle to plot ('o', 'dir', 'dir_tt').
    """

    if angle_opt == 'o':
        plt_col = 'pp_o_diff'
        color = 'orange'
    elif angle_opt == 'dir':
        plt_col = 'pp_dir_diff'
        color = 'green'
    elif angle_opt == 'dir_tt':
        plt_col = 'pp_dir_diff'

        impact_types = plot_df.Primary_Impact_Type.tolist()
        imp_col_dict = {'Helmet-to-helmet':'red', 'Helmet-to-body':'blue'}
        color = [imp_col_dict[x] for x in impact_types]

        #pa_types = plot_df.Player_Activity_Derived.tolist()
        #pa_col_dict = {
        #    'Blocking': 'red', 'Blocked': 'orange', 'Tackling': 'green',
        #    'Tackled': 'blue'
        #}
        #color = [pa_col_dict[x] for x in pa_types]
    else:
        raise ValueError('Not a valid option!')

    angles = plot_df.loc[:,plt_col].values

    def _fix_domain(ang):
        if ang < 0:
            return 360. - abs(ang)
        else:
            return ang

    radii = plot_df.acc_rank.astype(float).tolist()
    fix_angles = [_fix_domain(x) for x in angles]

    data = [
        go.Scatterpolar(
            r = radii,
            theta = fix_angles,
            mode = 'markers',
            marker = dict(
                color = color
            )
        )
    ]

    layout = go.Layout(
        showlegend = False,
        font=dict(
            family='sans serif',
            size=24,
            color='black'
        ),
        polar = dict(
            radialaxis = dict(
                showticklabels = False,
                showline=False,
                ticklen=0
            )
        )
    )

    fig = go.Figure(data=data,layout=layout)

    return fig

## MAIN
if __name__ == '__main__':

    # Load data.
    inj_df = pd.read_csv(f'{WDIR}injury_ngs_data.csv')

    # Add column for easy indexing.
    merge_cols = ['Season_Year', 'GameKey', 'PlayID']
    ind_df = inj_df.drop_duplicates(merge_cols).reset_index(drop=True)
    ind_df.loc[:, 'eventIndex'] = ind_df.index.values

    ind_df = ind_df.loc[:, ['eventIndex', 'Season_Year', 'GameKey', 'PlayID']]
    inj_df = inj_df.merge(ind_df, how='outer', left_on=merge_cols, right_on=merge_cols)

    # Get player-partner processed DataFrame.
    play_part_df = calculate_pp_distance(inj_df)

    # Step through each play, identifying the most likely point of impact and
    # grabbing a few rows around it.
    impacts = []

    for play_idx in range(len(inj_df.eventIndex.unique())):
        try:
            sp_df = play_part_df.loc[play_part_df.eventIndex == play_idx].reset_index(drop=True)

            # Find most probable time for impact between player/partner.
            impact_df = find_impact(sp_df)
            impacts.append(impact_df)
        except TypeError:
            continue

    pp_impact_df = pd.concat(impacts, ignore_index=True)

    # Add angle difference columns.
    pp_impact_df.loc[:, 'pp_dir_diff'] = pp_impact_df.play_dir - pp_impact_df.part_dir
    pp_impact_df.loc[:, 'pp_o_diff'] = pp_impact_df.play_o - pp_impact_df.part_o
    pp_impact_df.to_csv('/Users/cbonfield/Desktop/full_impact.csv', index=False)

    # Isolate most likely time step for impact.
    just_impact = pp_impact_df.loc[pp_impact_df.impact == 1]

    # Bring in summary stats for speed/acceleration (to be used on plot).
    spd_acc_ss = pd.read_csv(f'{WDIR}spd_acc_summary.csv')
    spd_acc_ss.rename(index=str, columns={'season_year': 'Season_Year', 'game_key': 'GameKey', 'play_id': 'PlayID'}, inplace=True)

    just_impact = just_impact.merge(spd_acc_ss, how='inner', left_on=merge_cols,
                                    right_on=merge_cols)

    # Bring in impact types/player activity.
    impact_type = pd.read_csv(f'{WDIR}video_review.csv')
    impact_type = impact_type.loc[:, ['Season_Year', 'GameKey', 'PlayID', 'Player_Activity_Derived', 'Primary_Impact_Type']]

    just_impact = just_impact.merge(impact_type, how='inner', left_on=merge_cols,
                                    right_on=merge_cols)

    # Pluck out the columns relevant for plotting.
    #keep_columns = ['max_play_a', 'pp_dir_diff', 'pp_o_diff']
    #plot_df = just_impact.loc[:, keep_columns].sort_values(by='max_play_a').reset_index(drop=True)
    plot_df = just_impact.sort_values(by='max_play_a').reset_index(drop=True)
    plot_df.loc[:, 'acc_rank'] = (plot_df.index.values+1)/(plot_df.index.max())

    plt_opt = 'dir_tt'
    figure = make_radial_plot(plot_df, plt_opt)

    ODIR = f'/Users/cbonfield/Projects/kaggle/nfl_punts/figures/rel_angles/'
    pio.write_image(figure, f'{ODIR}{plt_opt}_rel_angles.pdf')
