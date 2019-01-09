#
# Run this script to analyze characteristics of punt plays as a function of
# game condition. The primary intent of this script is to provide an additional
# source of data that can complement what we find from the NGS data.
#
# Author: Charlie Bonfield
# Last Modified: 1/2019

## IMPORTS
import pandas as pd
from scipy import stats
import preprocess_small_data as ppsd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)

import numpy as np
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

import plotly.io as pio
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


## FUNCTIONS
def perform_ks_test(pop_df, sam_df, col_of_interest):
    """
    Perform Kolmogorov-Smirnov test for population (all punts) and sample
    (punts with concussions) distribution of provided quantity (col_of_interest).

    Parameters:
        pop_df: pd.DataFrame
            DataFrame containing data from all selected punts (excluding the
            ones in the concussion set).
        sam_df: pd.DataFrame
            DataFrame containing data from concussion set.
        col_of_interest: str
            Name of column/quantity for which you'd like to conduct the KS test.
    """

    pdata = pop_df.loc[:, col_of_interest].values
    sdata = sam_df.loc[:, col_of_interest].values

    stat, p = stats.ks_2samp(pdata, sdata)

    return (stat, p)

def plot_distribution(pop_df, sam_df, col_of_interest, plot_hp):
    """
    Plot distribution of quantity (col_of_interest) for population/sample.

    Parameters:
        pop_df: pd.DataFrame
            DataFrame containing data from all selected punts (excluding the
            ones in the concussion set).
        sam_df: pd.DataFrame
            DataFrame containing data from concussion set.
        col_of_interest: str
            Name of column/quantity that you'd like to plot.
        plot_hp: tuple (ints/floats)
            Hyperparameter for plotting (bandwidth for KDE, number of bins for
            histogram). Index 0 contains population hyperparameter, Index 1
            contains sample hyperparameter.
    """

    pdata = pop_df.loc[:, col_of_interest].values
    sdata = sam_df.loc[:, col_of_interest].values

    # Uncomment for histograms.
    pop_trace = go.Histogram(
                    x=pdata,
                    opacity=0.75,
                    marker=dict(color='red'),
                    histnorm='probability',
                    nbinsx=plot_hp[0]
                )
    sam_trace = go.Histogram(
                    x=sdata,
                    opacity=0.75,
                    marker=dict(color='blue'),
                    histnorm='probability',
                    nbinsx=plot_hp[1]
                )

    """
    # Uncomment for KDEs.
    # Make KDE for each sample.
    pop_kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(pdata[:, np.newaxis])
    pop_plot = np.linspace(np.min(pdata), np.max(pdata), 1000)[:, np.newaxis]
    plog_dens = pop_kde.score_samples(pop_plot)

    sam_kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(sdata[:, np.newaxis])
    sam_plot = np.linspace(np.min(sdata), np.max(sdata), 1000)[:, np.newaxis]
    slog_dens = sam_kde.score_samples(sam_plot)

    pop_trace = go.Scatter(
                    x=pop_plot[:, 0],
                    y=np.exp(plog_dens),
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color='red', width=2)
                )

    sam_trace = go.Scatter(
                    x=sam_plot[:, 0],
                    y=np.exp(slog_dens),
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color='blue', width=2)
                )
    """

    # Make figure.
    fig = tools.make_subplots(rows=2,cols=1,shared_xaxes=True)
    fig.append_trace(pop_trace, 1, 1)
    fig.append_trace(sam_trace, 2, 1)
    fig['layout'].update(barmode='overlay')

    #data = [pop_trace, sam_trace]
    #layout = go.Layout(barmode='overlay')
    #fig = go.Figure(data=data, layout=layout)

    return fig

## MAIN
if __name__ == '__main__':

    # Load data.
    data_dict = ppsd.load_data()
    data_dict = ppsd.collect_outcomes(data_dict)
    data_dict = ppsd.expand_play_description(data_dict)

    # Focus on play information.
    play_info = data_dict['play_info']
    #rel_cols = ['Quarter', 'Punt_Outcome', 'Penalty_on_Punt', 'Punt_Distance',
    #            'Post_Punt_YardLine', 'Post_Punt_FieldSide', 'Post_Punt_Own_Territory',
    #            'Score_Differential']
    outcomes = ['return', 'downed', 'muffed punt', 'fair catch']
    #play_info = data_dict['play_info'].loc[:, rel_cols]
    play_info = play_info.loc[play_info.Punt_Outcome.isin(outcomes)].reset_index(drop=True)
    play_info.loc[:, 'playIndex'] = play_info.index.values

    # Split out the plays on which we had an identified concussion.
    inj_df = data_dict['video_injury']
    inj_df.loc[:, 'concussionPlay'] = 1
    drop_cols = ['Home_Team', 'Visit_Team', 'Qtr', 'PlayDescription', 'Week']
    inj_df.drop(drop_cols, axis=1, inplace=True)
    inj_df.rename(index=str, columns={'PlayId':'PlayID'}, inplace=True)

    # Join onto play_info.
    mer_cols = ['Season_Year', 'Season_Type', 'GameKey', 'PlayID']

    inj_play_info = play_info.merge(inj_df, how='inner', left_on=mer_cols,
                                    right_on=mer_cols)

    # Exclude plays from injury set from set used as population.
    play_info = play_info.loc[~play_info.playIndex.isin(inj_play_info.playIndex.tolist())].reset_index(drop=True)

    """
    # Generate plots/perform KS test.
    # Columns: ['Quarter', 'Score_Differential', 'Pre_Punt_RelativeYardLine',
    #           'Post_Punt_RelativeYardLine']
    #col_of_interest = 'Post_Punt_RelativeYardLine'
    cols_of_interest = ['Quarter', 'Score_Differential', 'Pre_Punt_RelativeYardLine',
                        'Post_Punt_RelativeYardLine']
    hp_dict = {'Quarter':(5,4), 'Score_Differential':(20,20),
               'Pre_Punt_RelativeYardLine': (20,20),
               'Post_Punt_RelativeYardLine': (20,20)}

    for col in cols_of_interest:

        # Drop some values.
        if col == 'Post_Punt_RelativeYardLine':
            pop_info = play_info.loc[play_info.Post_Punt_RelativeYardLine != -999]
        else:
            pop_info = play_info.copy()

        ks_stat, pval = perform_ks_test(pop_info, inj_play_info, col)
        print(col, pval)

        figure = plot_distribution(pop_info, inj_play_info, col, hp_dict[col])

        # Save figure.
        ODIR = f'/Users/cbonfield/Projects/kaggle/nfl_punts/figures/gc_dists/'
        pio.write_image(figure, f'{ODIR}{col}.pdf')
    """

    # Split score differential into bins.
    def _bin_score_differential(row):
        row_sd = row.Score_Differential

        if row_sd <= -21:
            return '< -21'
        elif (row_sd > -21) & (row_sd <= -14):
            return '-20 to -14'
        elif (row_sd > -14) & (row_sd <= -7):
            return '-13 to -7'
        elif (row_sd > -7) & (row_sd <= -1):
            return '-6 to -1'
        elif row_sd == 0:
            return 'TIE'
        elif (row_sd > 0) & (row_sd < 7):
            return '+1 to +6'
        elif (row_sd >= 7) & (row_sd < 14):
            return '+7 to +13'
        elif (row_sd >= 14) & (row_sd < 21):
            return '+14 to +20'
        elif row_sd >= 21:
            return '> +21'
        else:
            return 'ERROR'

    play_info.loc[:, 'binSD'] = play_info.apply(_bin_score_differential, axis=1)

    # Tally up outcomes as a function of score differential.
    reorder_indices = ['< -21', '-20 to -14', '-13 to -7', '-6 to -1', 'TIE',
                       '+1 to +6', '+7 to +13', '+14 to +20', '> +21']
    binned_outcomes = play_info.pivot_table(index='binSD', columns='Punt_Outcome',
                                            aggfunc='size', fill_value=0)
    binned_outcomes = binned_outcomes.div(binned_outcomes.sum(axis=1), axis=0)
    binned_outcomes = binned_outcomes.reindex(reorder_indices)

    """
    # Look at outcomes as a function of quarter.
    binned_outcomes = play_info.pivot_table(index='Quarter', columns='Punt_Outcome',
                                            aggfunc='size', fill_value=0)
    binned_outcomes = binned_outcomes.div(binned_outcomes.sum(axis=1), axis=0)
    """

    # Plot!
    trace = go.Heatmap(
                z=binned_outcomes.values.T,
                x=binned_outcomes.index.tolist(),
                y=binned_outcomes.columns.tolist())
    data=[trace]

    figure = go.Figure(data=data)
    ODIR = f'/Users/cbonfield/Projects/kaggle/nfl_punts/figures/gc_dists/'
    pio.write_image(figure, f'{ODIR}heatmap_v2.pdf')

    iplot(data)
