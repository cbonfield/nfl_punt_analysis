#
# Process NGS data for injured players.
#
# Author: Charlie Bonfield
# Last Modified: 12/2018

## IMPORTS
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.io as pio
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

pd.set_option('display.max_rows', 5000)

## VARIABLES
WDIR = '/Users/cbonfield/Projects/kaggle/nfl_punts/data/'
YD_TO_M = 0.9144 # multiplicative factor for converting yards to meters


## FUNCTIONS
def make_va_subplot(play_df, part_df, plt_option=None):
    """
    This function generates a plotly figure that can be used to display NGS data
    (see below for a list of supported options).

    Parameters:
        play_df: pd.DataFrame
            NGS data for player.
        part_df: pd.DataFrame
            NGS data for partner.
        plt_option: str
            Option for plotting. Supported values are:
                vel_acc: plot velocity/acceleration along each axis independently
                spd_acc: plot speed/acceleration (magnitudes)
                angles: plot orientation/direction as a function of time
                polar_angles: plot orientation/direction as a function of time
                              on a polar plot (r is time, theta is angle)
    """

    # Figure out where the ball snap occurred and get the index so that we can
    # discard all data prior to that instant.
    try:
        play_st = play_df.loc[play_df.Event == 'punt'].index[0]
        part_st = part_df.loc[part_df.Event == 'punt'].index[0]
    except IndexError:
        try:
            play_st = play_df.loc[play_df.Event == 'ball_snap'].index[0]
            part_st = part_df.loc[part_df.Event == 'ball_snap'].index[0]
        except IndexError:
            play_st = play_df.index.min()
            part_st = part_df.index.min()

    # Figure out where the play "ended" so that we can discard all data after
    # that. For simplicity, we assume that any concussion event would have occured
    # prior to a penalty flag being thrown or within five seconds of a tackle.
    try:
        play_ei = play_df.loc[play_df.Event == 'penalty_flag'].index[0]
        part_ei = play_df.loc[play_df.Event == 'penalty_flag'].index[0]
    except IndexError:
        try:
            play_ei = play_df.loc[play_df.Event == 'tackle'].index[0] + 50
            part_ei = part_df.loc[part_df.Event == 'tackle'].index[0] + 50

            play_ps = play_df.loc[play_df.Event == 'play_submit'].index[0]
            part_ps = part_df.loc[part_df.Event == 'play_submit'].index[0]

            while play_ei > play_ps:
                play_ei -= 10

            while part_ei > part_ps:
                part_ei -= 10
        except IndexError:
            play_ei = play_df.index.max()
            part_ei = part_df.index.max()

    # Slice out the data that we actually need.
    play_df = play_df.iloc[play_st:play_ei]
    part_df = part_df.iloc[part_st:part_ei]

    # Make traces for plotly figure, then plot! Note that we also return a
    # set of values for further analysis, too.
    data_dict = {}

    if plt_option == 'vel_acc':
        vel_play_x = go.Scatter(
                            x=play_df.t,
                            y=play_df.vx*YD_TO_M,
                            name = "Player",
                            line = dict(color = '#17BECF'),
                            opacity = 0.8)

        vel_part_x = go.Scatter(
                            x=part_df.t,
                            y=part_df.vx*YD_TO_M,
                            name = "Partner",
                            line = dict(color = '#7F7F7F'),
                            opacity = 0.8)

        acc_play_x = go.Scatter(
                            x=play_df.t,
                            y=play_df.ax*YD_TO_M,
                            name = "Player",
                            line = dict(color = '#17BECF'),
                            opacity = 0.8)

        acc_part_x = go.Scatter(
                            x=part_df.t,
                            y=part_df.ax*YD_TO_M,
                            name = "Partner",
                            line = dict(color = '#7F7F7F'),
                            opacity = 0.8)
        vel_play_y = go.Scatter(
                            x=play_df.t,
                            y=play_df.vy*YD_TO_M,
                            name = "Player",
                            line = dict(color = '#17BECF'),
                            opacity = 0.8)

        vel_part_y = go.Scatter(
                            x=part_df.t,
                            y=part_df.vy*YD_TO_M,
                            name = "Partner",
                            line = dict(color = '#7F7F7F'),
                            opacity = 0.8)

        acc_play_y = go.Scatter(
                            x=play_df.t,
                            y=play_df.ay*YD_TO_M,
                            name = "Player",
                            line = dict(color = '#17BECF'),
                            opacity = 0.8)

        acc_part_y = go.Scatter(
                            x=part_df.t,
                            y=part_df.ay*YD_TO_M,
                            name = "Partner",
                            line = dict(color = '#7F7F7F'),
                            opacity = 0.8)

        # Create a plotly figure, adding all of the traces.
        fig = tools.make_subplots(rows=4, cols=1, subplot_titles=('Velocity (x)', 'Acceleration (x)', 'Velocity (y)', 'Acceleration (y)'))

        fig.append_trace(vel_play_x, 1, 1)
        fig.append_trace(vel_part_x, 1, 1)
        fig.append_trace(acc_play_x, 2, 1)
        fig.append_trace(acc_part_x, 2, 1)
        fig.append_trace(vel_play_y, 3, 1)
        fig.append_trace(vel_part_y, 3, 1)
        fig.append_trace(acc_play_y, 4, 1)
        fig.append_trace(acc_part_y, 4, 1)

        # Pull out relevant data.
        data_dict['min_time'] = play_df.t.min()
        data_dict['min_play_vx'] = play_df.vx.min()*YD_TO_M
        data_dict['min_play_ax'] = play_df.ax.min()*YD_TO_M
        data_dict['max_play_vx'] = play_df.vx.max()*YD_TO_M
        data_dict['max_play_ax'] = play_df.ax.max()*YD_TO_M
        data_dict['min_part_vx'] = part_df.vx.min()*YD_TO_M
        data_dict['min_part_ax'] = part_df.ax.min()*YD_TO_M
        data_dict['max_part_vx'] = part_df.vx.max()*YD_TO_M
        data_dict['max_part_ax'] = part_df.ax.max()*YD_TO_M
        data_dict['min_play_vy'] = play_df.vy.min()*YD_TO_M
        data_dict['min_play_ay'] = play_df.ay.min()*YD_TO_M
        data_dict['max_play_vy'] = play_df.vy.max()*YD_TO_M
        data_dict['max_play_ay'] = play_df.ay.max()*YD_TO_M
        data_dict['min_part_vy'] = part_df.vy.min()*YD_TO_M
        data_dict['min_part_ay'] = part_df.ay.min()*YD_TO_M
        data_dict['max_part_vy'] = part_df.vy.max()*YD_TO_M
        data_dict['max_part_ay'] = part_df.ay.max()*YD_TO_M
    elif plt_option == 'spd_acc':
        vel_play = go.Scatter(
                        x=play_df.t,
                        y=play_df.s*YD_TO_M,
                        name = "Player",
                        line = dict(color = 'red'),
                        opacity = 0.8)

        vel_part = go.Scatter(
                        x=part_df.t,
                        y=part_df.s*YD_TO_M,
                        name = "Partner",
                        line = dict(color = 'blue'),
                        opacity = 0.8)

        acc_play = go.Scatter(
                        x=play_df.t,
                        y=play_df.a*YD_TO_M,
                        name = "Player",
                        line = dict(color = 'red'),
                        opacity = 0.8)

        acc_part = go.Scatter(
                        x=part_df.t,
                        y=part_df.a*YD_TO_M,
                        name = "Partner",
                        line = dict(color = 'blue'),
                        opacity = 0.8)

        fig = tools.make_subplots(rows=2, cols=1, subplot_titles=('Speed', 'Acceleration'))

        fig.append_trace(vel_play, 1, 1)
        fig.append_trace(vel_part, 1, 1)
        fig.append_trace(acc_play, 2, 1)
        fig.append_trace(acc_part, 2, 1)

        # Pull out relevant data (min/max speed/acceleration for
        # players/partners).
        data_dict['min_time'] = play_df.t.min()
        data_dict['min_play_s'] = play_df.s.min()*YD_TO_M
        data_dict['min_play_a'] = play_df.a.min()*YD_TO_M
        data_dict['max_play_s'] = play_df.s.max()*YD_TO_M
        data_dict['max_play_a'] = play_df.a.max()*YD_TO_M
        data_dict['min_part_s'] = part_df.s.min()*YD_TO_M
        data_dict['min_part_a'] = part_df.a.min()*YD_TO_M
        data_dict['max_part_s'] = part_df.s.max()*YD_TO_M
        data_dict['max_part_a'] = part_df.a.max()*YD_TO_M
    elif plt_option == 'angles':
        ang_play = go.Scatter(
                        x=play_df.t,
                        y=play_df.o,
                        name = "Player",
                        line = dict(color = '#17BECF'),
                        opacity = 0.8)

        ang_part = go.Scatter(
                        x=part_df.t,
                        y=part_df.o,
                        name = "Partner",
                        line = dict(color = '#7F7F7F'),
                        opacity = 0.8)

        dir_play = go.Scatter(
                        x=play_df.t,
                        y=play_df.dir,
                        name = "Player",
                        line = dict(color = '#17BECF'),
                        opacity = 0.8)

        dir_part = go.Scatter(
                        x=part_df.t,
                        y=part_df.dir,
                        name = "Partner",
                        line = dict(color = '#7F7F7F'),
                        opacity = 0.8)

        fig = tools.make_subplots(rows=2, cols=1, subplot_titles=('Orientation', 'Direction'))

        fig.append_trace(ang_play, 1, 1)
        fig.append_trace(ang_part, 1, 1)
        fig.append_trace(dir_play, 2, 1)
        fig.append_trace(dir_part, 2, 1)
    elif plt_option == 'polar_angles':
        ang_play = go.Scatterpolar(
                        name = 'Player (o)',
                        r=play_df.t,
                        theta=play_df.o,
                        mode = 'lines',
                        line = dict(color='#FF0000'))

        ang_part = go.Scatterpolar(
                        name = 'Partner (o)',
                        r=part_df.t,
                        theta=part_df.o,
                        mode = 'lines',
                        line = dict(color='#F6A000'),
                        subplot='polar3')

        dir_play = go.Scatterpolar(
                        name = 'Player (dir)',
                        r=play_df.t,
                        theta=play_df.dir,
                        mode = 'lines',
                        line = dict(color='#00EE35'),
                        subplot='polar2')

        dir_part = go.Scatterpolar(
                        name = 'Partner (dir)',
                        r=part_df.t,
                        theta=part_df.dir,
                        mode = 'lines',
                        line = dict(color='#030CF4'),
                        subplot='polar4')

        data = [ang_play, dir_play, ang_part, dir_part]

        layout = go.Layout(
                    title = 'Angular Positions',
                    polar = dict(

                        domain = dict(
                            x = [0, 0.46],
                            y = [0.56, 1]
                          ),
                          radialaxis = dict(
                            angle = 90
                          )
                        ),
                    polar2 = dict(
                      domain = dict(
                        x = [0, 0.46],
                        y = [0, 0.44]
                      ),
                      radialaxis = dict(
                        angle = 90
                      )
                    ),
                    polar3 = dict(
                      domain = dict(
                        x = [0.54, 1],
                        y = [0.56, 1]
                      ),
                      radialaxis = dict(
                        angle = 90
                      )
                    ),
                    polar4 = dict(
                      domain = dict(
                        x = [0.54, 1],
                        y = [0, 0.44]
                      ),
                      radialaxis = dict(
                        angle = 90
                      )
                    ),
                    legend=dict(orientation="h")
                )

        fig = go.Figure(data=data,layout=layout)
    else:
        raise ValueError('Not a valid option!')

    return fig, data_dict


## MAIN
if __name__ == '__main__':

    # Load data.
    inj_df = pd.read_csv(f'{WDIR}injury_ngs_data.csv')

    inj_df.head()

    # Add column for easy indexing.
    merge_cols = ['Season_Year', 'GameKey', 'PlayID']
    ind_df = inj_df.drop_duplicates(merge_cols).reset_index(drop=True)
    ind_df.loc[:, 'eventIndex'] = ind_df.index.values

    ind_df = ind_df.loc[:, ['eventIndex', 'Season_Year', 'GameKey', 'PlayID']]
    inj_df = inj_df.merge(ind_df, how='outer', left_on=merge_cols, right_on=merge_cols)

    # Iterate through each play, exporting a figure each time.
    plt_opt = 'polar_angles'
    dyn_info = []

    for play_idx in range(len(inj_df.eventIndex.unique())):
        try:
            sing_df = inj_df.loc[inj_df.eventIndex == play_idx]
            play_df = sing_df.loc[sing_df.Identifier == 'PLAYER'].reset_index(drop=True)
            part_df = sing_df.loc[sing_df.Identifier == 'PARTNER'].reset_index(drop=True)

            # Grab some stuff for labeling saved figure.
            sy = sing_df.Season_Year.values[0]
            gk = sing_df.GameKey.values[0]
            pi = sing_df.PlayID.values[0]

            figure, dd = make_va_subplot(play_df, part_df, plt_option=plt_opt)
            sys.exit()

            # Add data dictionary to list.
            dd['season_year'] = sy
            dd['game_key'] = gk
            dd['play_id'] = pi
            dyn_info.append(dd)

            # Save figure.
            ODIR = f'/Users/cbonfield/Projects/kaggle/nfl_punts/figures/{plt_opt}/'
            pio.write_image(figure, f'{ODIR}{plt_opt}_{sy}_{gk}_{pi}.pdf')
        except TypeError:
            continue

    # Export DataFrame containing dynamic information.
    #dd_df = pd.DataFrame(dyn_info)
    #dd_df.to_csv(f'{WDIR}{plt_opt}_summary.csv', index=False)

    #iplot(figure, filename = "injury_data_fig")
