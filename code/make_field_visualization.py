#
# Run this script to make a nice visualization of player/partner position
# data.
#
# Author: Charlie Bonfield
# Last Modified: 1/2019

## IMPORTS
import glob
import numpy as np
import pandas as pd

from scipy import interpolate

import plotly.io as pio
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


## VARIABLES
DDIR = '/Users/cbonfield/Projects/kaggle/nfl_punts/data/'


## FUNCTIONS
def trim_player_partner_data(ngs_df):
    """
    Given a DataFrame with NGS data for player/partner on punt play, cut out
    the relevant NGS data.

    Parameters:
        ngs_df: pd.DataFrame
            DataFrame containing NGS data.
    """

    # Isolate player/partner data.
    play_df = ngs_df.loc[ngs_df.Identifier == 'PLAYER'].dropna().reset_index(drop=True)
    part_df = ngs_df.loc[ngs_df.Identifier == 'PARTNER'].dropna().reset_index(drop=True)

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

    return play_df, part_df

def make_plot(ngs_df):
    """
    Given a DataFrame with NGS data for player/partner on punt play, make a
    snappy visualization.

    Parameters:
        ngs_df: pd.DataFrame
            DataFrame containing NGS data.
    """

    # Isolate player/partner data.
    play_df = ngs_df.loc[ngs_df.Identifier == 'PLAYER'].dropna().reset_index(drop=True)
    part_df = ngs_df.loc[ngs_df.Identifier == 'PARTNER'].dropna().reset_index(drop=True)

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

    play_trace = go.Scatter(
                     x = play_df.x.values,
                     y = play_df.y.values,
                     mode = 'markers',
                     marker = dict(
                        color=play_df.s.values,
                        colorbar=dict(x=1.0),
                        colorscale='Reds',
                        size=12
                     )
                 )

    part_trace = go.Scatter(
                     x = part_df.x.values,
                     y = part_df.y.values,
                     mode = 'markers',
                     marker = dict(
                        color=part_df.s.values,
                        colorbar=dict(x=1.1),
                        colorscale='Blues',
                        reversescale=True,
                        size=12
                     )
                 )

    yardline_trace = go.Scatter(
                        x=[20, 30, 40, 50, 60, 70, 80, 90, 100],
                        y=[1, 1, 1, 1, 1, 1, 1, 1, 1],
                        mode='text',
                        text=['10','20','30','40','50','40','30','20','10'],
                        textposition='top center',
                        textfont=dict(
                            family='sans serif',
                            size=20,
                            color='white'
                        )
                    )

    data = [play_trace, part_trace, yardline_trace]

    layout = go.Layout(
                autosize=True,
                showlegend=False,
                plot_bgcolor='#008000',
                xaxis=dict(
                    range=[0,120],
                    linecolor='black',
                    linewidth=2,
                    mirror=True,
                    showticklabels=False
                ),
                yaxis=dict(
                    range=[0,53.3],
                    linecolor='black',
                    linewidth=2,
                    mirror=True,
                    showticklabels=False
                ),
                annotations=[
                    dict(
                        x=0,
                        y=0.5,
                        showarrow=False,
                        text='HOME ENDZONE',
                        textangle=270,
                        xref='paper',
                        yref='paper',
                        font=dict(
                            family='sans serif',
                            size=24,
                            color='white'
                        )
                    ),
                    dict(
                        x=1,
                        y=0.5,
                        showarrow=False,
                        text='AWAY ENDZONE',
                        textangle=90,
                        xref='paper',
                        yref='paper',
                        font=dict(
                            family='sans serif',
                            size=24,
                            color='white'
                        )
                    ),
                    dict(
                        x=float(17./120.),
                        y=1,
                        showarrow=False,
                        text='10',
                        textangle=180,
                        xref='paper',
                        yref='paper',
                        font=dict(
                            family='sans serif',
                            size=20,
                            color='white'
                        )
                    ),
                    dict(
                        x=float(27./120.),
                        y=1,
                        showarrow=False,
                        text='20',
                        textangle=180,
                        xref='paper',
                        yref='paper',
                        font=dict(
                            family='sans serif',
                            size=20,
                            color='white'
                        )
                    ),
                    dict(
                        x=float(37./120.),
                        y=1,
                        showarrow=False,
                        text='30',
                        textangle=180,
                        xref='paper',
                        yref='paper',
                        font=dict(
                            family='sans serif',
                            size=20,
                            color='white'
                        )
                    ),
                    dict(
                        x=float(50./120.),
                        y=1,
                        showarrow=False,
                        text='40',
                        textangle=180,
                        xref='paper',
                        yref='paper',
                        font=dict(
                            family='sans serif',
                            size=20,
                            color='white'
                        )
                    ),
                    dict(
                        x=float(60./120.),
                        y=1,
                        showarrow=False,
                        text='50',
                        textangle=180,
                        xref='paper',
                        yref='paper',
                        font=dict(
                            family='sans serif',
                            size=20,
                            color='white'
                        )
                    ),
                    dict(
                        x=float(70./120.),
                        y=1,
                        showarrow=False,
                        text='40',
                        textangle=180,
                        xref='paper',
                        yref='paper',
                        font=dict(
                            family='sans serif',
                            size=20,
                            color='white'
                        )
                    ),
                    dict(
                        x=float(80./120.),
                        y=1,
                        showarrow=False,
                        text='30',
                        textangle=180,
                        xref='paper',
                        yref='paper',
                        font=dict(
                            family='sans serif',
                            size=20,
                            color='white'
                        )
                    ),
                    dict(
                        x=float(93./120.),
                        y=1,
                        showarrow=False,
                        text='20',
                        textangle=180,
                        xref='paper',
                        yref='paper',
                        font=dict(
                            family='sans serif',
                            size=20,
                            color='white'
                        )
                    ),
                    dict(
                        x=float(103./120.),
                        y=1,
                        showarrow=False,
                        text='10',
                        textangle=180,
                        xref='paper',
                        yref='paper',
                        font=dict(
                            family='sans serif',
                            size=20,
                            color='white'
                        )
                    )
                ],
                shapes=[
                    {
                        'type': 'line',
                        'x0': 10,
                        'y0': 0,
                        'x1': 10,
                        'y1': 53.3,
                        'line': {
                            'color': 'white',
                            'width': 2
                        },
                    },
                    {
                        'type': 'line',
                        'x0': 110,
                        'y0': 0,
                        'x1': 110,
                        'y1': 53.3,
                        'line': {
                            'color': 'white',
                            'width': 2
                        },
                    },
                    {
                        'type': 'line',
                        'x0': 20,
                        'y0': 0,
                        'x1': 20,
                        'y1': 53.3,
                        'line': {
                            'color': 'white',
                            'width': 1
                        },
                    },
                    {
                        'type': 'line',
                        'x0': 30,
                        'y0': 0,
                        'x1': 30,
                        'y1': 53.3,
                        'line': {
                            'color': 'white',
                            'width': 1
                        },
                    },
                    {
                        'type': 'line',
                        'x0': 40,
                        'y0': 0,
                        'x1': 40,
                        'y1': 53.3,
                        'line': {
                            'color': 'white',
                            'width': 1
                        },
                    },
                    {
                        'type': 'line',
                        'x0': 50,
                        'y0': 0,
                        'x1': 50,
                        'y1': 53.3,
                        'line': {
                            'color': 'white',
                            'width': 1
                        },
                    },
                    {
                        'type': 'line',
                        'x0': 60,
                        'y0': 0,
                        'x1': 60,
                        'y1': 53.3,
                        'line': {
                            'color': 'white',
                            'width': 1
                        },
                    },
                    {
                        'type': 'line',
                        'x0': 70,
                        'y0': 0,
                        'x1': 70,
                        'y1': 53.3,
                        'line': {
                            'color': 'white',
                            'width': 1
                        },
                    },
                    {
                        'type': 'line',
                        'x0': 80,
                        'y0': 0,
                        'x1': 80,
                        'y1': 53.3,
                        'line': {
                            'color': 'white',
                            'width': 1
                        },
                    },
                    {
                        'type': 'line',
                        'x0': 90,
                        'y0': 0,
                        'x1': 90,
                        'y1': 53.3,
                        'line': {
                            'color': 'white',
                            'width': 1
                        },
                    },
                    {
                        'type': 'line',
                        'x0': 100,
                        'y0': 0,
                        'x1': 100,
                        'y1': 53.3,
                        'line': {
                            'color': 'white',
                            'width': 1
                        },
                    },
                    {
                        'type': 'line',
                        'x0': 15,
                        'y0': 0,
                        'x1': 15,
                        'y1': 53.3,
                        'line': {
                            'color': 'white',
                            'width': 1,
                            'dash':'dot'
                        },
                    },
                    {
                        'type': 'line',
                        'x0': 25,
                        'y0': 0,
                        'x1': 25,
                        'y1': 53.3,
                        'line': {
                            'color': 'white',
                            'width': 1,
                            'dash':'dot'
                        },
                    },
                    {
                        'type': 'line',
                        'x0': 35,
                        'y0': 0,
                        'x1': 35,
                        'y1': 53.3,
                        'line': {
                            'color': 'white',
                            'width': 1,
                            'dash':'dot'
                        },
                    },
                    {
                        'type': 'line',
                        'x0': 45,
                        'y0': 0,
                        'x1': 45,
                        'y1': 53.3,
                        'line': {
                            'color': 'white',
                            'width': 1,
                            'dash':'dot'
                        },
                    },
                    {
                        'type': 'line',
                        'x0': 55,
                        'y0': 0,
                        'x1': 55,
                        'y1': 53.3,
                        'line': {
                            'color': 'white',
                            'width': 1,
                            'dash':'dot'
                        },
                    },
                    {
                        'type': 'line',
                        'x0': 65,
                        'y0': 0,
                        'x1': 65,
                        'y1': 53.3,
                        'line': {
                            'color': 'white',
                            'width': 1,
                            'dash':'dot'
                        },
                    },
                    {
                        'type': 'line',
                        'x0': 75,
                        'y0': 0,
                        'x1': 75,
                        'y1': 53.3,
                        'line': {
                            'color': 'white',
                            'width': 1,
                            'dash':'dot'
                        },
                    },
                    {
                        'type': 'line',
                        'x0': 85,
                        'y0': 0,
                        'x1': 85,
                        'y1': 53.3,
                        'line': {
                            'color': 'white',
                            'width': 1,
                            'dash':'dot'
                        },
                    },
                    {
                        'type': 'line',
                        'x0': 95,
                        'y0': 0,
                        'x1': 95,
                        'y1': 53.3,
                        'line': {
                            'color': 'white',
                            'width': 1,
                            'dash':'dot'
                        },
                    },
                    {
                        'type': 'line',
                        'x0': 105,
                        'y0': 0,
                        'x1': 105,
                        'y1': 53.3,
                        'line': {
                            'color': 'white',
                            'width': 1,
                            'dash':'dot'
                        },
                    }
                ]
             )

    fig = go.Figure(data=data, layout=layout)

    return fig

## MAIN
if __name__ == '__main__':

    # Load data from plays with concussions.
    ngs_data = pd.read_csv(f'{DDIR}injury_ngs_data.csv')

    # Add column for easy indexing.
    merge_cols = ['Season_Year', 'GameKey', 'PlayID']
    ind_df = ngs_data.drop_duplicates(merge_cols).reset_index(drop=True)
    ind_df.loc[:, 'eventIndex'] = ind_df.index.values
    play_indexes = ind_df.eventIndex.tolist()

    ind_df = ind_df.loc[:, ['eventIndex', 'Season_Year', 'GameKey', 'PlayID']]
    ngs_data = ngs_data.merge(ind_df, how='inner', left_on=merge_cols, right_on=merge_cols)

    # Testing.
    #sp_data = ngs_data.loc[ngs_data.eventIndex == 3].reset_index(drop=True)
    #figure = make_plot(sp_data)
    #iplot(figure, filename='field-viz')

    """
    # Generate set of plots as static files.
    for play_idx in range(len(ngs_data.eventIndex.unique())):
        try:
            sp_data = ngs_data.loc[ngs_data.eventIndex == play_idx].reset_index(drop=True)
            figure = make_plot(sp_data)

            # Grab some stuff for labeling saved figure.
            sy = sp_data.Season_Year.values[0]
            gk = sp_data.GameKey.values[0]
            pi = sp_data.PlayID.values[0]

            # Save figure.
            ODIR = f'/Users/cbonfield/Projects/kaggle/nfl_punts/figures/fields/'
            pio.write_image(figure, f'{ODIR}field_{sy}_{gk}_{pi}.pdf')
        except TypeError:
            continue
    """

    # Run to generate animated figure.
    init_notebook_mode(connected=True)

    figure = {
        'data': [],
        'layout': {},
        'frames': []
    }

    ## CUSTOM
    field_xaxis=dict(
            range=[0,120],
            linecolor='black',
            linewidth=2,
            mirror=True,
            showticklabels=False
    )
    field_yaxis=dict(
            range=[0,53.3],
            linecolor='black',
            linewidth=2,
            mirror=True,
            showticklabels=False
    )
    field_annotations=[
            dict(
                x=0,
                y=0.5,
                showarrow=False,
                text='HOME ENDZONE',
                textangle=270,
                xref='paper',
                yref='paper',
                font=dict(
                    family='sans serif',
                    size=24,
                    color='white'
                )
            ),
            dict(
                x=1,
                y=0.5,
                showarrow=False,
                text='AWAY ENDZONE',
                textangle=90,
                xref='paper',
                yref='paper',
                font=dict(
                    family='sans serif',
                    size=24,
                    color='white'
                )
            ),
            dict(
                x=float(17./120.),
                y=1,
                showarrow=False,
                text='10',
                textangle=180,
                xref='paper',
                yref='paper',
                font=dict(
                    family='sans serif',
                    size=20,
                    color='white'
                )
            ),
            dict(
                x=float(27./120.),
                y=1,
                showarrow=False,
                text='20',
                textangle=180,
                xref='paper',
                yref='paper',
                font=dict(
                    family='sans serif',
                    size=20,
                    color='white'
                )
            ),
            dict(
                x=float(37./120.),
                y=1,
                showarrow=False,
                text='30',
                textangle=180,
                xref='paper',
                yref='paper',
                font=dict(
                    family='sans serif',
                    size=20,
                    color='white'
                )
            ),
            dict(
                x=float(50./120.),
                y=1,
                showarrow=False,
                text='40',
                textangle=180,
                xref='paper',
                yref='paper',
                font=dict(
                    family='sans serif',
                    size=20,
                    color='white'
                )
            ),
            dict(
                x=float(60./120.),
                y=1,
                showarrow=False,
                text='50',
                textangle=180,
                xref='paper',
                yref='paper',
                font=dict(
                    family='sans serif',
                    size=20,
                    color='white'
                )
            ),
            dict(
                x=float(70./120.),
                y=1,
                showarrow=False,
                text='40',
                textangle=180,
                xref='paper',
                yref='paper',
                font=dict(
                    family='sans serif',
                    size=20,
                    color='white'
                )
            ),
            dict(
                x=float(80./120.),
                y=1,
                showarrow=False,
                text='30',
                textangle=180,
                xref='paper',
                yref='paper',
                font=dict(
                    family='sans serif',
                    size=20,
                    color='white'
                )
            ),
            dict(
                x=float(93./120.),
                y=1,
                showarrow=False,
                text='20',
                textangle=180,
                xref='paper',
                yref='paper',
                font=dict(
                    family='sans serif',
                    size=20,
                    color='white'
                )
            ),
            dict(
                x=float(103./120.),
                y=1,
                showarrow=False,
                text='10',
                textangle=180,
                xref='paper',
                yref='paper',
                font=dict(
                    family='sans serif',
                    size=20,
                    color='white'
                )
            )
    ]
    field_shapes=[
            {
                'type': 'line',
                'x0': 10,
                'y0': 0,
                'x1': 10,
                'y1': 53.3,
                'line': {
                    'color': 'white',
                    'width': 2
                },
            },
            {
                'type': 'line',
                'x0': 110,
                'y0': 0,
                'x1': 110,
                'y1': 53.3,
                'line': {
                    'color': 'white',
                    'width': 2
                },
            },
            {
                'type': 'line',
                'x0': 20,
                'y0': 0,
                'x1': 20,
                'y1': 53.3,
                'line': {
                    'color': 'white',
                    'width': 1
                },
            },
            {
                'type': 'line',
                'x0': 30,
                'y0': 0,
                'x1': 30,
                'y1': 53.3,
                'line': {
                    'color': 'white',
                    'width': 1
                },
            },
            {
                'type': 'line',
                'x0': 40,
                'y0': 0,
                'x1': 40,
                'y1': 53.3,
                'line': {
                    'color': 'white',
                    'width': 1
                },
            },
            {
                'type': 'line',
                'x0': 50,
                'y0': 0,
                'x1': 50,
                'y1': 53.3,
                'line': {
                    'color': 'white',
                    'width': 1
                },
            },
            {
                'type': 'line',
                'x0': 60,
                'y0': 0,
                'x1': 60,
                'y1': 53.3,
                'line': {
                    'color': 'white',
                    'width': 1
                },
            },
            {
                'type': 'line',
                'x0': 70,
                'y0': 0,
                'x1': 70,
                'y1': 53.3,
                'line': {
                    'color': 'white',
                    'width': 1
                },
            },
            {
                'type': 'line',
                'x0': 80,
                'y0': 0,
                'x1': 80,
                'y1': 53.3,
                'line': {
                    'color': 'white',
                    'width': 1
                },
            },
            {
                'type': 'line',
                'x0': 90,
                'y0': 0,
                'x1': 90,
                'y1': 53.3,
                'line': {
                    'color': 'white',
                    'width': 1
                },
            },
            {
                'type': 'line',
                'x0': 100,
                'y0': 0,
                'x1': 100,
                'y1': 53.3,
                'line': {
                    'color': 'white',
                    'width': 1
                },
            },
            {
                'type': 'line',
                'x0': 15,
                'y0': 0,
                'x1': 15,
                'y1': 53.3,
                'line': {
                    'color': 'white',
                    'width': 1,
                    'dash':'dot'
                },
            },
            {
                'type': 'line',
                'x0': 25,
                'y0': 0,
                'x1': 25,
                'y1': 53.3,
                'line': {
                    'color': 'white',
                    'width': 1,
                    'dash':'dot'
                },
            },
            {
                'type': 'line',
                'x0': 35,
                'y0': 0,
                'x1': 35,
                'y1': 53.3,
                'line': {
                    'color': 'white',
                    'width': 1,
                    'dash':'dot'
                },
            },
            {
                'type': 'line',
                'x0': 45,
                'y0': 0,
                'x1': 45,
                'y1': 53.3,
                'line': {
                    'color': 'white',
                    'width': 1,
                    'dash':'dot'
                },
            },
            {
                'type': 'line',
                'x0': 55,
                'y0': 0,
                'x1': 55,
                'y1': 53.3,
                'line': {
                    'color': 'white',
                    'width': 1,
                    'dash':'dot'
                },
            },
            {
                'type': 'line',
                'x0': 65,
                'y0': 0,
                'x1': 65,
                'y1': 53.3,
                'line': {
                    'color': 'white',
                    'width': 1,
                    'dash':'dot'
                },
            },
            {
                'type': 'line',
                'x0': 75,
                'y0': 0,
                'x1': 75,
                'y1': 53.3,
                'line': {
                    'color': 'white',
                    'width': 1,
                    'dash':'dot'
                },
            },
            {
                'type': 'line',
                'x0': 85,
                'y0': 0,
                'x1': 85,
                'y1': 53.3,
                'line': {
                    'color': 'white',
                    'width': 1,
                    'dash':'dot'
                },
            },
            {
                'type': 'line',
                'x0': 95,
                'y0': 0,
                'x1': 95,
                'y1': 53.3,
                'line': {
                    'color': 'white',
                    'width': 1,
                    'dash':'dot'
                },
            },
            {
                'type': 'line',
                'x0': 105,
                'y0': 0,
                'x1': 105,
                'y1': 53.3,
                'line': {
                    'color': 'white',
                    'width': 1,
                    'dash':'dot'
                },
            }
    ]


    # fill in most of layout
    figure['layout']['autosize'] = True
    figure['layout']['showlegend'] = False
    figure['layout']['plot_bgcolor'] = '#008000'

    figure['layout']['xaxis'] = field_xaxis
    figure['layout']['yaxis'] = field_yaxis
    figure['layout']['annotations'] = field_annotations
    figure['layout']['shapes'] = field_shapes

    figure['layout']['hovermode'] = 'closest'
    figure['layout']['sliders'] = {
        'args': [
            'transition', {
                'duration': 400,
                'easing': 'cubic-in-out'
            }
        ],
        'initialValue': 0,
        'plotlycommand': 'animate',
        'values': play_indexes,
        'visible': True
    }
    figure['layout']['updatemenus'] = [
        {
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 500, 'redraw': False},
                             'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                    'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }
    ]

    sliders_dict = {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 20},
            'prefix': 'Play Index: ',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 300, 'easing': 'cubic-in-out'},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': []
    }

    # Make data (for single play).
    plt_dicts = []
    pidx = 0

    sp_data = ngs_data.loc[ngs_data.eventIndex == pidx].reset_index(drop=True)

    # Grab some stuff for labeling saved figure.
    sy = sp_data.Season_Year.values[0]
    gk = sp_data.GameKey.values[0]
    pi = sp_data.PlayID.values[0]

    plt_dict = {}
    plt_dict['playIndex'] = pi
    plt_dict['seasonYear'] = sy
    plt_dict['gameKey'] = gk
    plt_dict['playID'] = pi
    plt_dicts.append(plt_dict)

    # Get player/partner data (reduced).
    rd_play_df, rd_part_df = trim_player_partner_data(sp_data)

    for i in range(2):
        if not i:
            ngs_dataset = rd_play_df.copy()
            plt_name = 'Player'
            color_scale = 'Reds'
            cb_loc = 1.0
        else:
            ngs_dataset = rd_part_df.copy()
            plt_name = 'Partner'
            color_scale = 'Blues'
            cb_loc = 1.1

        data_dict = {
            'x': list(ngs_dataset['x']),
            'y': list(ngs_dataset['y']),
            'mode': 'markers',
            'marker': {
                'color': list(ngs_dataset['s']),
                'colorbar': {'x':cb_loc},
                'colorscale':color_scale,
                'size':12
            },
            'name':plt_name
        }
        figure['data'].append(data_dict)

    # Add data for yardline trace.
    data_dict = {
        'x': [20, 30, 40, 50, 60, 70, 80, 90, 100],
        'y': [1, 1, 1, 1, 1, 1, 1, 1, 1],
        'mode': 'text',
        'text': ['10','20','30','40','50','40','30','20','10'],
        'textposition': 'top center',
        'textfont': {
            'family': 'sans serif',
            'size': 20,
            'color': 'white'
        }
    }
    figure['data'].append(data_dict)

    # Make frames.
    for pidx in play_indexes:
        frame = {'data': [], 'name': pidx}
        try:
            print(pidx)
            sp_data = ngs_data.loc[ngs_data.eventIndex == pidx].reset_index(drop=True)

            # Grab some stuff for labeling saved figure.
            sy = sp_data.Season_Year.values[0]
            gk = sp_data.GameKey.values[0]
            pi = sp_data.PlayID.values[0]

            plt_dict = {}
            plt_dict['playIndex'] = pi
            plt_dict['seasonYear'] = sy
            plt_dict['gameKey'] = gk
            plt_dict['playID'] = pi
            plt_dicts.append(plt_dict)

            # Get player/partner data (reduced).
            rd_play_df, rd_part_df = trim_player_partner_data(sp_data)

            for i in range(2):
                if not i:
                    ngs_dataset = rd_play_df.copy()
                    plt_name = 'Player'
                    color_scale = 'Reds'
                    cb_loc = 1.0
                else:
                    ngs_dataset = rd_part_df.copy()
                    plt_name = 'Partner'
                    color_scale = 'Blues'
                    cb_loc = 1.1

                data_dict = {
                    'x': list(ngs_dataset['x']),
                    'y': list(ngs_dataset['y']),
                    'mode': 'markers',
                    'marker': {
                        'color': list(ngs_dataset['s']),
                        'colorbar': {'x':cb_loc},
                        'colorscale':color_scale,
                        'size':12
                    },
                    'name':plt_name
                }
                frame['data'].append(data_dict)

            # Add data for yardline trace.
            data_dict = {
                'x': [20, 30, 40, 50, 60, 70, 80, 90, 100],
                'y': [1, 1, 1, 1, 1, 1, 1, 1, 1],
                'mode': 'text',
                'text': ['10','20','30','40','50','40','30','20','10'],
                'textposition': 'top center',
                'textfont': {
                    'family': 'sans serif',
                    'size': 20,
                    'color': 'white'
                }
            }
            frame['data'].append(data_dict)

            # Add frames.
            figure['frames'].append(frame)
            slider_step = {'args': [
                [pidx],
                {'frame': {'duration': 300, 'redraw': False},
                 'mode': 'immediate',
               'transition': {'duration': 300}}
             ],
             'label': pidx,
             'method': 'animate'}
            sliders_dict['steps'].append(slider_step)
        except TypeError:
            continue

    figure['layout']['sliders'] = [sliders_dict]

    iplot(figure)
