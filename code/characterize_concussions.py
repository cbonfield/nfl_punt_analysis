#
# Run script to characterize concussions. After collecting data from all punts
# where the ball was returned, we want to see if we can adopt some sort of
# definition for what can be considered normal dynamics.
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
SDIR = '/Users/cbonfield/Projects/kaggle/nfl_punts/data/sumdynamics/'
ODIR = f'/Users/cbonfield/Projects/kaggle/nfl_punts/figures/'

## FUNCTIONS
def ecdf(data):
    """
    Compute empirical cumulative distribution function for set of 1D data.

    Parameters:
        data: np.array
    Returns:
        x: np.array
            Sorted data.
        y: np.array
            ECDF(x).
    """

    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n

    return x, y

def make_histogram(ngs_df, col_to_plot, add_lines=False):
    """
    Generate plotly histogram using maximum accelerations experienced by
    players.

    Parameters:
        ngs_df: pd.DataFrame
            DataFrame containing NGS data.
        col_to_plot: str
            Name of column to plot.
        add_lines: bool (default False)
            Boolean indicating whether we wish to superpose lines for values
            from concussed players.
    """

    trace = (
        go.Histogram(
            x=ngs_df[col_to_plot].values,
            histnorm='probability',
            nbinsx=100
        )
    )

    layout = go.Layout(
                autosize=True,
                xaxis=dict(
                    range=[0,100]
                )
             )

    if add_lines:
        _ = 'placeholder'
    else:
        data = [trace]

    fig = go.Figure(data=data, layout=layout)

    return fig

def plot_ecdf(int_ecdf, inj_ecdf, max_x):
    """
    Plot ECDF from entire set of play data and from the subset where a concussion
    occurred.

    Parameters:
        int_ecdf: np.array (values: [quantity, ecdf])
            Values from interpolated ECDF from entire dataset.
        inj_ecdf: np.array (values: [quantity, ecdf])
            ECDF/values for players in the injury set.
        max_x: int
            Max x-value (used for plot range).
    """

    int_trace = go.Scatter(
                    x = int_ecdf[:,0],
                    y = int_ecdf[:,1],
                    mode = 'markers',
                    marker = dict(opacity=0.3)
                )

    inj_trace = go.Scatter(
                    x = inj_ecdf[:,0],
                    y = inj_ecdf[:,1],
                    mode = 'markers',
                    marker = dict(color='red')
                )

    data = [int_trace, inj_trace]

    layout = go.Layout(
                autosize=True,
                showlegend=False,
                xaxis=dict(
                    range=[0,max_x]
                )
             )

    fig = go.Figure(data=data, layout=layout)

    return fig

## MAIN
if __name__ == '__main__':

    # Load data.
    files = glob.glob(f'{SDIR}*.csv')
    flist = []

    for f in files:
        tmp_df = pd.read_csv(f)
        flist.append(tmp_df)

    sum_df = pd.concat(flist, ignore_index=True)

    # Drop unreasonable accelerations.
    sum_df = sum_df.loc[sum_df.max_a <= 150.]

    # Construct ECDF for players.
    srt_spds, spd_ecdf = ecdf(sum_df.max_s.values)
    int_s_ecdf = interpolate.interp1d(srt_spds, spd_ecdf)
    srt_accs, acc_ecdf = ecdf(sum_df.max_a.values)
    int_a_ecdf = interpolate.interp1d(srt_accs, acc_ecdf)

    # Load in set of summary statistics for players involved in concussions.
    inj_df = pd.read_csv(SDIR.split('data/')[0]+'data/spd_acc_summary.csv')

    ren_dict = {'season_year':'Season_Year', 'game_key':'GameKey',
                'play_id': 'PlayID'}
    inj_df.rename(index=str, columns=ren_dict, inplace=True)

    inj_df.head()

    # Add column for player/partner action.
    ppa_df = pd.read_csv(SDIR.split('data/')[0]+'data/video_review.csv')
    ppa_df = ppa_df.loc[:, ['Season_Year', 'GameKey', 'PlayID', 'Player_Activity_Derived', 'Primary_Partner_Activity_Derived']]

    mer_cols = ['Season_Year', 'GameKey', 'PlayID']
    inj_df = inj_df.merge(ppa_df, how='inner', left_on=mer_cols, right_on=mer_cols)

    def _identify_moving_pp(row):
        player_activity = row.Player_Activity_Derived

        if 'ing' in player_activity:
            return 1
        elif 'ed' in player_activity:
            return 0
        else:
            raise ValueError('Check derived activity!')

    def _moving_play_s(row, dyn_opt):
        pp_activity = row.PP_Activity

        if dyn_opt == 's':
            return pp_activity * row.max_play_s + abs(pp_activity - 1) * row.max_part_s
        elif dyn_opt == 'a':
            return pp_activity * row.max_play_a + abs(pp_activity - 1) * row.max_part_a
        else:
            raise ValueError('Check derived activity!')

    inj_df.loc[:, 'PP_Activity'] = inj_df.apply(_identify_moving_pp, axis=1)
    inj_df.loc[:, 'max_move_s'] = inj_df.apply(_moving_play_s, args=('s'), axis=1)
    inj_df.loc[:, 'max_move_a'] = inj_df.apply(_moving_play_s, args=('a'), axis=1)

    def _get_s_cdf(x):
        return int_s_ecdf(x)

    def _get_a_cdf(x):
        return int_a_ecdf(x)

    inj_df.loc[:, 's_cumprob'] = inj_df.max_play_s.apply(_get_s_cdf)
    inj_df.loc[:, 'a_cumprob'] = inj_df.max_play_a.apply(_get_a_cdf)

    """
    # Make figure (histogram), then plot.
    plotcol = 'max_a'

    figure = make_histogram(sum_df, plotcol)
    #iplot(figure, filename='acc-histogram')
    pio.write_image(figure, f'{ODIR}{plotcol}_hist.pdf')
    """

    # Make figure (acceleration ECDF), then plot.
    a_vals = np.linspace(min(srt_accs), max(srt_accs),200)
    a_ecdf = np.array([int_a_ecdf(x) for x in a_vals])
    a_ecdf = np.vstack([a_vals,a_ecdf]).T

    inj_a_data = inj_df.loc[:, ['max_move_a', 'a_cumprob']].values

    figure = plot_ecdf(a_ecdf, inj_a_data, 60)
    #iplot(figure, filename='acc-ecdf')
    pio.write_image(figure, f'{ODIR}acc-ecdf-mov.pdf')

    # Make figure (speed ECDF), then plot.
    s_vals = np.linspace(min(srt_spds), max(srt_spds),200)
    s_ecdf = np.array([int_s_ecdf(x) for x in s_vals])
    s_ecdf = np.vstack([s_vals,s_ecdf]).T

    inj_s_data = inj_df.loc[:, ['max_move_s', 's_cumprob']].values

    figure = plot_ecdf(s_ecdf, inj_s_data, 10)
    #iplot(figure, filename='spd-ecdf')
    pio.write_image(figure, f'{ODIR}spd-ecdf-mov.pdf')
