#
# Script for preprocessing everything except the NGS data.
#
# Author: Charlie Bonfield
# Last Modified: 12/2018

## IMPORTS
import numpy as np
import pandas as pd


## VARIABLES
WDIR = '/Users/cbonfield/Projects/kaggle/nfl_punts/data/'


## FUNCTIONS
def collect_outcomes(data):
    """
    Extract the punt outcome from the PlayDescription field.

    Parameters:
        data: dict (keys: labels, values: DataFrames)
            Data dictionary - likely the output from load_data().
    """

    play_info =  data['play_info']

    def _process_description(row):
        tmp_desc = row.PlayDescription
        outcome = ''

        if 'touchback' in tmp_desc.lower():
            outcome = 'touchback'
        elif 'fair catch' in tmp_desc.lower():
            outcome = 'fair catch'
        elif 'out of bounds' in tmp_desc.lower():
            outcome = 'out of bounds'
        elif 'muff' in tmp_desc.lower():
            outcome = 'muffed punt'
        elif 'downed' in tmp_desc.lower():
            outcome = 'downed'
        elif 'no play' in tmp_desc.lower():
            outcome = 'no play'
        elif 'blocked' in tmp_desc.lower():
            outcome = 'blocked punt'
        elif 'fumble' in tmp_desc.lower():
            outcome = 'fumble'
        elif 'pass' in tmp_desc.lower():
            outcome = 'pass'
        elif 'declined' in tmp_desc.lower():
            outcome = 'declined penalty'
        elif 'direct snap' in tmp_desc.lower():
            outcome = 'direct snap'
        elif 'safety' in tmp_desc.lower():
            outcome = 'safety'
        else:
            if 'punts' in tmp_desc.lower():
                outcome = 'return'
            else:
                outcome = 'SPECIAL'

        return outcome

    play_info.loc[:, 'Punt_Outcome'] = play_info.apply(_process_description, axis=1)

    def _identify_penalties(row):
        if 'penalty' in row.PlayDescription.lower():
            return 1
        else:
            return 0

    play_info.loc[:, 'Penalty_on_Punt'] = play_info.apply(_identify_penalties, axis=1)

    # Update dictionary to include additional set of features.
    data.update({'play_info': play_info})

    return data

def expand_play_description(data):
    """
    Expand the PlayDescription field in a standardized fashion. This function
    extracts a number of relevant additional features from PlayDescription,
    including punt distance, post-punt field location, and a few other derived
    features.

    Parameters:
        data: dict (keys: labels, values: DataFrames)
            Data dictionary - likely the output from load_data().
    """

    play_info = data['play_info']

    def _split_punt_distance(row):
        try:
            return int(row.PlayDescription.split('punts ')[1].split('yard')[0])
        except IndexError:
            return np.nan

    def _split_field_position(row):
        try:
            return row.PlayDescription.split(',')[0].split('to ')[1]
        except IndexError:
            return ''

    def _post_punt_territory(row):
        if row.Poss_Team == row.Post_Punt_FieldSide:
            return 0
        else:
            return 1

    def _start_punt_field_position(row):
        try:
            field_position = int(row.YardLine.split(' ')[1])
        except:
            print(row.YardLine)

        if row.Poss_Team in row.YardLine:
            return field_position
        else:
            return 100 - field_position

    def _field_position_punt(row):
        if 'end zone' in row.Post_Punt_YardLine:
            return 0
        elif '50' in row.Post_Punt_YardLine:
            return 50
        else:
            try:
                yard_line = int(row.Post_Punt_YardLine.split(' ')[1])
                own_field = int(row.Post_Punt_Own_Territory)

                if not own_field:
                    return 100 - yard_line
                else:
                    return yard_line
            except:
                return -999

    play_info.loc[:, 'Punt_Distance'] = play_info.apply(_split_punt_distance, axis=1)
    play_info.loc[:, 'Post_Punt_YardLine'] = play_info.apply(_split_field_position, axis=1)
    play_info.loc[:, 'Post_Punt_FieldSide'] = play_info.Post_Punt_YardLine.apply(lambda x: x.split(' ')[0])
    play_info.loc[:, 'Post_Punt_Own_Territory'] = play_info.apply(_post_punt_territory, axis=1)
    play_info.loc[:, 'Pre_Punt_RelativeYardLine'] = play_info.apply(_start_punt_field_position, axis=1)
    play_info.loc[:, 'Post_Punt_RelativeYardLine'] = play_info.apply(_field_position_punt, axis=1)

    # Extract additional information from play info (home team, away team, score
    # differential, home/away punt identifier).
    play_info.loc[:, 'Home_Team'] = play_info.Home_Team_Visit_Team.apply(lambda x: x.split('-')[0])
    play_info.loc[:, 'Away_Team'] = play_info.Home_Team_Visit_Team.apply(lambda x: x.split('-')[1])
    play_info.loc[:, 'Home_Points'] = play_info.Score_Home_Visiting.apply(lambda x: x.split('-')[0]).astype(int)
    play_info.loc[:, 'Away_Points'] = play_info.Score_Home_Visiting.apply(lambda x: x.split('-')[1]).astype(int)

    def _home_away_punt_bool(row):
        if row.Home_Team == row.Poss_Team:
            return 1
        else:
            return 0

    play_info.loc[:, 'Home_Visit_Team_Punt'] = play_info.apply(_home_away_punt_bool, axis=1)

    def _get_score_differential(row):
        if not row.Home_Visit_Team_Punt:
            return int(row.Away_Points - row.Home_Points)
        else:
            return int(row.Home_Points - row.Away_Points)

    play_info.loc[:, 'Score_Differential'] = play_info.apply(_get_score_differential, axis=1)

    # Update dictionary to include additional set of features.
    data.update({'play_info': play_info})

    return data

def load_data(raw_bool=False):
    """
    When called, this function will load in all of the data and do the relevant
    preprocessing (mainly just a series of merges to link the injury data with a
    few of the other data sources). The output of this function is a dictionary
    with key/value pairs that are labels/DataFrames, respectively.

    Parameters:
        raw_bool: bool (default False)
            Boolean indicating whether you wish to perform the necessary
            preprocessing steps (False) or not (True).
    """

    # Load data.
    game_data = pd.read_csv(f'{WDIR}game_data.csv')
    play_info = pd.read_csv(f'{WDIR}play_information.csv')
    play_role = pd.read_csv(f'{WDIR}play_player_role_data.csv')
    punt_data = pd.read_csv(f'{WDIR}player_punt_data.csv')

    video_injury = pd.read_csv(f'{WDIR}video_footage-injury.csv')
    video_review = pd.read_csv(f'{WDIR}video_review.csv')
    video_control = pd.read_csv(f'{WDIR}video_footage-control.csv')

    if raw_bool:
        pass
    else:
        # Rename columns to match format (between video_injury/video_control and
        # everything else).
        ren_dict = {
            'season': 'Season_Year',
            'Type': 'Season_Type',
            'Home_team': 'Home_Team',
            'gamekey': 'GameKey',
            'playid': 'PlayId'
        }

        video_injury.rename(index=str, columns=ren_dict, inplace=True)
        video_control.rename(index=str, columns=ren_dict, inplace=True)
        video_review.rename(index=str, columns={'PlayID':'PlayId'}, inplace=True)

        # Join video_review to video_injury.
        video_injury = video_injury.merge(video_review, how='outer',
                                          left_on=['Season_Year', 'GameKey', 'PlayId'],
                                          right_on=['Season_Year', 'GameKey', 'PlayId'])

        # Process punt_data - it's possible to have multiple numbers for the same
        # player, so we'll drop number to get rid of duplicates.
        punt_data.drop('Number', axis=1, inplace=True)
        punt_data.drop_duplicates(inplace=True)

        # Add player primary position to video_injury.
        video_injury = video_injury.merge(punt_data, how='inner', on=['GSISID'])
        video_injury.rename(index=str, columns={'Position':'Player_Position'},
                            inplace=True)

        # Fix a few values in Primary_Partner_GSISID that will cause the next
        # merge to barf (one nan, one 'Unclear').
        video_injury.replace(to_replace={'Primary_Partner_GSISID':'Unclear'},
                             value=99999, inplace=True)
        video_injury.replace(to_replace={'Primary_Partner_GSISID':np.nan},
                             value=99999, inplace=True)
        video_injury.loc[:, 'Primary_Partner_GSISID'] = video_injury.Primary_Partner_GSISID.astype(int)

        # Add primary partner primary position to video_injury.
        video_injury = video_injury.merge(punt_data, how='left',
                                          left_on=['Primary_Partner_GSISID'],
                                          right_on=['GSISID'])
        video_injury.drop('GSISID_y', axis=1, inplace=True)
        video_injury.rename(index=str, columns={'GSISID_x':'GSISID'}, inplace=True)
        video_injury.rename(index=str, columns={'Position':'Primary_Partner_Position'},
                            inplace=True)

        # Add punt specific play role for players to video_injury.
        play_role.rename(index=str, columns={'PlayID':'PlayId'}, inplace=True)
        video_injury = video_injury.merge(play_role, how='left',
                                          left_on=['Season_Year', 'GameKey', 'PlayId', 'GSISID'],
                                          right_on=['Season_Year', 'GameKey', 'PlayId', 'GSISID'])
        video_injury.rename(index=str, columns={'Role':'Player_Punt_Role'}, inplace=True)

        # Add punt specific play role for primary partners to video_injury.
        video_injury = video_injury.merge(play_role, how='left',
                                          left_on=['Season_Year', 'GameKey', 'PlayId', 'Primary_Partner_GSISID'],
                                          right_on=['Season_Year', 'GameKey', 'PlayId', 'GSISID'])
        video_injury.drop('GSISID_y', axis=1, inplace=True)
        video_injury.rename(index=str, columns={'GSISID_x':'GSISID'}, inplace=True)
        video_injury.rename(index=str, columns={'Role':'Primary_Partner_Punt_Role'},
                            inplace=True)

    # Stick everything in a dictionary to return as output.
    out_dict = {
        'game_data': game_data,
        'play_info': play_info,
        'play_role': play_role,
        'punt_data': punt_data,
        'video_injury': video_injury,
        'video_control': video_control,
        'video_review': video_review
    }

    return out_dict

def parse_penalties(play_info_df):
    """
    Extract penalty types for plays on which we had penalties.

    Parameters:
        play_info_df: pd.DataFrame
            DataFrame containing play information.
    """

    pen_df = play_info_df.loc[play_info_df.Penalty_on_Punt == 1].reset_index(drop=True)

    def _extract_penalty_type(row):
        try:
            tmp_desc = row.PlayDescription.lower()
            pen_suff = tmp_desc.split('penalty on ')[1]
            drop_plr = pen_suff.split(', ')[1]

            penalty = drop_plr.split(',')[0]
        except:
            penalty = 'EXCEPTION'

        return penalty

    pen_df.loc[:, 'Penalty_Type'] = pen_df.apply(_extract_penalty_type, axis=1)

    return pen_df

## MAIN
if __name__ == '__main__':

    # Load data. While we load in all of the relevant data sources (excluding
    # the stuff from NGS), we perform the necessary preprocessing to assemble
    # a relevant set of data for video_injury.
    data_dict = load_data()

    # Extract additional information from PlayDescription.
    data_dict = collect_outcomes(data_dict)
    data_dict = expand_play_description(data_dict)

    # Unpack the data dictionary into DataFrames that can be parsed.
    gd_df = data_dict['game_data']
    pi_df = data_dict['play_info']
    pr_df = data_dict['play_role']
    pd_df = data_dict['punt_data']
    vi_df = data_dict['video_injury']
    vc_df = data_dict['video_control']
    vr_df = data_dict['video_review']

    # Extract a separate DataFrame for plays on which we saw a penalty.
    pen_df = parse_penalties(pi_df)
