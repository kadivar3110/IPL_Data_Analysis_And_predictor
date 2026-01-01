import streamlit as st
import pandas as pd
import os
import plotly.express as px
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Debugging: Print file locations to help diagnose deployment issues
st.write(f"Script directory: {script_dir}")
st.write(f"Files in directory: {os.listdir(script_dir)}")

# Use the zipped file to save space and avoid GitHub limits
csv_path = os.path.join(script_dir, 'IPL.zip')

if not os.path.exists(csv_path):
    st.error(f"Error: IPL.zip not found at {csv_path}. Please ensure you uploaded it to GitHub.")
    st.stop()

df = pd.read_csv(csv_path)
import pickle

def preprocess(df):

    df_ipl_first_inning = df[df['innings'] == 1]

    total_runs_per_match = df_ipl_first_inning.groupby('match_id')['runs_total'].sum().reset_index()
    total_runs_per_match = total_runs_per_match.rename(columns={'runs_total': 'targeted_total_run'})

    df_ipl = df[['match_id', 'date', 'event_name','innings','batting_team', 'bowling_team', 'over', 'ball', 'batter', 'bowler', 'runs_extras', 'runs_total', 'runs_target', 'player_of_match',
    'match_won_by', 'toss_winner', 'season', 'superover_winner']]

    df_ipl = df_ipl.drop_duplicates(subset='match_id', keep='first')

    df_ipl = pd.merge(df_ipl, total_runs_per_match, on='match_id', how='left')

    df_ipl = df_ipl[df_ipl['innings'] == 1]

    df_ipl = df_ipl.drop(['over', 'ball', 'runs_extras', 'runs_total', 'runs_target', 'event_name', 'innings', 'batter', 'bowler'], axis=1)

    return df_ipl
def change_in_season_format(df_ipl):
    df_ipl['season'] = df_ipl['season'].astype(str)

    df_ipl['season'] = df_ipl['season'].replace({
    '2007/08': '2008',
    '2009/10': '2010',
    '2020/21': '2020'
    })
    df_ipl['season'] = df_ipl['season'].astype(int)
    seasons = df_ipl['season'].unique().tolist()
    seasons.sort()

    return seasons
# Load the pkl file
def rearenge_name(df_ipl):
    df_ipl['match_won_by'] = df_ipl['match_won_by'].replace({
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
    'Rising Pune Supergiants': 'Rising Pune Supergiant'
    })


    df_ipl = df_ipl[df_ipl['match_won_by'] != 'Unknown']

    df_ipl['toss_winner'] = df_ipl['toss_winner'].replace({
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
    'Rising Pune Supergiants': 'Rising Pune Supergiant'
    })


    df_ipl = df_ipl[df_ipl['toss_winner'] != 'Unknown']

    df_ipl['bowling_team'] = df_ipl['bowling_team'].replace({
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
    'Rising Pune Supergiants': 'Rising Pune Supergiant'
    })


    df_ipl = df_ipl[df_ipl['bowling_team'] != 'Unknown']

    df_ipl['batting_team'] = df_ipl['batting_team'].replace({
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
    'Rising Pune Supergiants': 'Rising Pune Supergiant'
    })


    df_ipl = df_ipl[df_ipl['batting_team'] != 'Unknown']

    df_ipl['superover_winner'] = df_ipl['superover_winner'].replace({
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
    'Rising Pune Supergiants': 'Rising Pune Supergiant'
    })


    df_ipl = df_ipl[df_ipl['superover_winner'] != 'Unknown']

    Team_names = df_ipl['batting_team'].unique().tolist()
    Team_names.sort()
    return df_ipl, Team_names

def return_by_selection(df: pd.DataFrame, team: str, seas) -> pd.DataFrame:
    df_team_season = df.copy()
    if (team == 'Overall') and (seas == 'Overall'):
        return df_team_season
    if (team != 'Overall') and (seas == 'Overall'):
        return df_team_season[df_team_season['match_won_by'] == team]
    if (team == 'Overall') and (seas != 'Overall'):
        return df_team_season[df_team_season['season'] == seas]
    return df_team_season[(df_team_season['match_won_by'] == team) & (df_team_season['season'] == seas)]
    
pkl_path = os.path.join(script_dir, 'preprocess.pkl')

if not os.path.exists(pkl_path):
    st.error(f"Error: preprocess.pkl not found at {pkl_path}. Please ensure you uploaded it to GitHub.")
    st.stop()

with open(pkl_path, 'rb') as f:
    preprocess_function = pickle.load(f)

df_ipl : pd.DataFrame = preprocess_function(df)
seasons = change_in_season_format(df_ipl)
df_ipl, Team_names = rearenge_name(df_ipl)
Team_names.insert(0, 'Overall')
seasons.insert(0, 'Overall')

st.sidebar.title("IPL Analysis")
option = st.sidebar.radio("Select Option", ["OverAll Analysis", "Statistical Analysis", 'Player Analysis', "City Analysis"])


if option == "OverAll Analysis":
    selected_season = st.sidebar.selectbox("Select Season", seasons)
    selected_team = st.sidebar.selectbox("Select Winner Team", Team_names)
    df_ipl = return_by_selection(df_ipl, selected_team, selected_season)

    if selected_team == 'Overall' and selected_season == 'Overall':
        st.title("Overall IPL Analysis")
    elif selected_team != 'Overall' and selected_season == 'Overall':
        st.title(f"Overall Analysis of {selected_team} Wins")
    elif selected_team == 'Overall' and selected_season != 'Overall':
        st.title(f"Overall Analysis of Season {selected_season}")
    else:
        st.title(f"Overall Analysis of {selected_team} Wins in Season {selected_season}")

    st.dataframe(df_ipl)

if option == "Statistical Analysis":
    st.title("Statistical Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Teams", len(df_ipl['batting_team'].unique()))
    col2.metric("Total Matches", len(df_ipl['match_id'].unique()))
    col3.metric("Total Seasons", len(df_ipl['season'].unique()))

    col1, col2 = st.columns(2)
    col1.metric("Total Runs by Batter", 355373)
    col2.metric("Total Wickets by Bowler", 12650)

    mean_by_season = df_ipl.groupby('season')['targeted_total_run'].mean().reset_index() # Convert Series to DataFrame
    mean_by_season = mean_by_season.rename(columns={'targeted_total_run':'Avrage Run'}) # Now 'columns' argument is valid
    fig = px.line(mean_by_season, x='season', y='Avrage Run', title='Average Run per Season')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)

    matches_per_season = df_ipl.groupby('season')['match_id'].count().reset_index()
    matches_per_season.rename(columns={'match_id': 'Total Matches'}, inplace=True)

    fig = px.line(matches_per_season, x='season', y='Total Matches',
             title='Total Number of Matches per Season',
             labels={'season': 'Season', 'Total Matches': 'Number of Matches'})

    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)

    uni = df_ipl.groupby('season')['batting_team'].nunique().reset_index()
    uni.rename(columns={'batting_team': 'Total Teams'}, inplace=True)
    fig = px.line(uni, x='season', y='Total Teams', title='Total Number of Teams per Season')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)



# Re-create team_stats_df
    team_stats = {
        'Team': [],
        'Matches Played': [],
        'Matches Won': [],
        'Win Ratio': []
    }

    # Get all unique team names from batting and bowling teams
    all_teams = pd.concat([df_ipl['batting_team'], df_ipl['bowling_team']]).unique()

    for team in all_teams:
        matches_played = (
            (df_ipl['batting_team'] == team) | 
            (df_ipl['bowling_team'] == team)
        ).sum()
        matches_won = (df_ipl['match_won_by'] == team).sum()
    
        win_ratio = matches_won / matches_played if matches_played > 0 else 0
    
        team_stats['Team'].append(team)
        team_stats['Matches Played'].append(matches_played)
        team_stats['Matches Won'].append(matches_won)
        team_stats['Win Ratio'].append(win_ratio)

    team_stats_df = pd.DataFrame(team_stats)
    team_stats_df = team_stats_df.sort_values(by='Win Ratio', ascending=False)

    # Create figure with secondary y-axis
    fig = make_subplots(rows=1, cols=1, specs=[[{'secondary_y': True}]])

    # Add traces for Matches Played and Matches Won on primary y-axis
    fig.add_trace(
        go.Bar(name='Matches Played', x=team_stats_df['Team'], y=team_stats_df['Matches Played'], marker_color='skyblue'),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(name='Matches Won', x=team_stats_df['Team'], y=team_stats_df['Matches Won'], marker_color='lightgreen'),
        secondary_y=False,
    )

    # Add trace for Win Ratio on secondary y-axis
    fig.add_trace(
        go.Scatter(name='Win Ratio', x=team_stats_df['Team'], y=team_stats_df['Win Ratio'], mode='lines+markers', marker_color='red'),
        secondary_y=True,
    )
    # Add figure title and labels
    fig.update_layout(
        title_text='Team Performance: Matches Played, Won, and Win Ratio',
        xaxis_title='Team',
        barmode='group' # Group the bars
    )

    # Set y-axes titles
    fig.update_yaxes(title_text='Number of Matches', secondary_y=False)
    fig.update_yaxes(title_text='Win Ratio', secondary_y=True, range=[0, 1])

    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig)

    max_wins = df_ipl.groupby(['season', 'match_won_by']).size().reset_index(name='wins')
    max_wins = max_wins.loc[max_wins.groupby('season')['wins'].idxmax()]



    fig = px.bar(max_wins, x='season', y='wins', color='match_won_by',
            title='Maximum Wins by a Single Team per Season',
             labels={'season': 'Season', 'wins': 'Maximum Wins', 'match_won_by': 'Winning Team'})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)
    df_ipl['toss_match_winner'] = (df_ipl['toss_winner'] == df_ipl['match_won_by'])

    toss_win = df_ipl['toss_match_winner'].value_counts().reset_index()
    toss_win.columns = ['Toss Winner Won Match', 'Count']

    fig = px.pie(toss_win, values='Count', names='Toss Winner Won Match',
             title='Relationship Between Toss Win and Match Win',
             labels={'Toss Winner Won Match': 'Toss Winner Won Match', 'Count': 'Number of Matches'})

    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig)

if option == "Player Analysis":
    st.title("Player Analysis")
    selected_Player_Type = st.sidebar.selectbox("Select Player Type", ['Bolwer', 'Batter'])
    if selected_Player_Type == 'Bolwer':

        bowler_df = df.groupby(['match_id', 'bowler'])['bowler_wicket'].sum().reset_index()
        bowler_df = pd.merge(bowler_df, df_ipl[['match_id', 'season']].drop_duplicates(), on='match_id',how='left')
        bowler_df = pd.merge(bowler_df, df[['match_id', 'stage']].drop_duplicates(), on='match_id', how='left')
        bowler_df['season'] = bowler_df['season'].astype('Int64')

        uni_bowler = bowler_df['bowler'].unique().tolist()

        Total_wk = bowler_df.groupby(['season', 'bowler'])['bowler_wicket'].sum().reset_index()
        Total_wk = Total_wk.rename(columns={'bowler_wicket': 'total_wickets'})
        
        selected_bowler = st.selectbox("Select Bowler", uni_bowler)

        def bowler_analysis(bowler_name):

            all_df = bowler_df[bowler_df['bowler'] == bowler_name]
            all_wk_by_season = all_df.groupby('season')['bowler_wicket'].sum().reset_index()
            all_wk_by_season.rename(columns={'bowler_wicket': 'total_wickets'}, inplace=True)

            fig_all = px.line(all_wk_by_season, x='season', y='total_wickets',
                        title=f'Overall Total Wickets Year Wise for {bowler_name}')
            fig_all.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_all)

            important_stages = ['Final', 'Semi Final', '3rd Place Play-Off',
                      'Qualifier 1', 'Elimination Final', 'Qualifier 2', 'Eliminator']
            imprt_df = all_df[all_df['stage'].isin(important_stages)]
            imprt_wk_by_season = imprt_df.groupby('season')['bowler_wicket'].sum().reset_index()
            imprt_wk_by_season.rename(columns={'bowler_wicket': 'total_wickets'}, inplace=True)
            fig_imprt = px.line(imprt_wk_by_season, x='season', y='total_wickets',
                         title=f'Total Wickets Year Wise for {bowler_name} in Important Matches')
            fig_imprt.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_imprt)

        abhi_blw = df[['runs_bowler', 'match_id', 'bowler']]
        abhi_blw = abhi_blw.groupby(['match_id', 'bowler'])['runs_bowler'].sum().reset_index()
        abhi_blw = abhi_blw.rename(columns={'runs_bowler': 'runs_given'}) 

        bowler_df = pd.merge(bowler_df, abhi_blw, on=['match_id', 'bowler'], how='left')

        abhi_blw1 = df[['match_id', 'bowler', 'valid_ball']]
        abhi_blw1 = abhi_blw1.groupby(['match_id', 'bowler'])['valid_ball'].sum().reset_index()
        abhi_blw1['Total over'] = abhi_blw1['valid_ball'] / 6
        abhi_blw1 = abhi_blw1.drop(columns=['valid_ball'])

        bowler_df = pd.merge(bowler_df, abhi_blw1, on=['match_id', 'bowler'], how='left')

        def bowler(bowler_name):
            new_temp1 = bowler_df[bowler_df['bowler'] == bowler_name]

            tlt_run = new_temp1.groupby('season')['runs_given'].sum().reset_index()
            tlt_run = tlt_run.rename(columns={'runs_given': 'total_runs'})

            tlt_ovr = new_temp1.groupby('season')['Total over'].sum().reset_index()
            tlt_ovr = tlt_ovr.rename(columns={'Total over': 'total_overs'})

            eco_avg_df = tlt_ovr
            eco_avg_df['Total Run Given'] = tlt_run['total_runs']
            eco_avg_df['economy'] = tlt_run['total_runs'] / tlt_ovr['total_overs']

            tlt_wkt = new_temp1.groupby('season')['bowler_wicket'].sum().reset_index()
            tlt_wkt = tlt_wkt.rename(columns={'bowler_wicket': 'total_wickets'})

            eco_avg_df = pd.merge(eco_avg_df, tlt_wkt, on='season', how='left')
            eco_avg_df['economy'] = round(eco_avg_df['economy'], 2)
            eco_avg_df['avrage'] = eco_avg_df['Total Run Given']/eco_avg_df['total_wickets']

            fig = px.bar(eco_avg_df, x='season', y='economy', title=f'Economy Rate per Season for {bowler_name}')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig)
  
            fig = px.bar(eco_avg_df, x='season', y='avrage', title=f'Avrage per Wicket per Season for {bowler_name}')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig)

        bowler_analysis(selected_bowler) 
        bowler(selected_bowler)
    else:
        batter_df = df.groupby(['match_id', 'batter'])['runs_batter'].sum().reset_index()
        batter_df = pd.merge(batter_df, df_ipl[['match_id', 'season']].drop_duplicates(), on='match_id', how='left')
        batter_df = pd.merge(batter_df, df[['match_id', 'stage']].drop_duplicates(), on='match_id', how='left')
        batter_df['season'] = batter_df['season'].astype('Int64')

        uni_batter = batter_df['batter'].unique().tolist()

        Total_run = batter_df.groupby(['season', 'batter'])['runs_batter'].sum().reset_index()
        Total_run = Total_run.rename(columns={'runs_batter': 'total_runs'})

        selected_batter = st.selectbox("Select Batter", uni_batter)

        def batter_analysis(batter_name):
            all_df = batter_df[batter_df['batter'] == batter_name]
            all_run_by_season = all_df.groupby('season')['runs_batter'].sum().reset_index()
            all_run_by_season.rename(columns={'runs_batter': 'total_runs'}, inplace=True)

            fig_all = px.line(all_run_by_season, x='season', y='total_runs',
                        title=f'Overall Total Runs Year Wise for {batter_name}')
            fig_all.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_all)

            important_stages = ['Final', 'Semi Final', '3rd Place Play-Off',
                      'Qualifier 1', 'Elimination Final', 'Qualifier 2', 'Eliminator']
            imprt_df = all_df[all_df['stage'].isin(important_stages)]
            imprt_run_by_season = imprt_df.groupby('season')['runs_batter'].sum().reset_index()
            imprt_run_by_season.rename(columns={'runs_batter': 'total_runs'}, inplace=True)
            fig_imprt = px.line(imprt_run_by_season, x='season', y='total_runs',
                         title=f'Total Runs Year Wise for {batter_name} in Important Matches')
            fig_imprt.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_imprt)
        
        def batter(batter_name):
            batter1 = batter_df[batter_df['batter'] == batter_name]
            abhik = batter1.groupby(['season', 'batter'])['match_id'].count().reset_index()
            abhik = abhik.rename(columns={'match_id': 'matches played'})
            fig = px.bar(abhik, x='season', y='matches played', title=f'Matches Played per Season for {batter_name}')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig)

            batter2 = batter1.groupby(['season', 'batter'])['runs_batter'].mean().reset_index()
            batter2 = batter2.rename(columns={'runs_batter': 'Avrage_Runs'})
            fig = px.bar(batter2, x='season', y='Avrage_Runs', title=f'Avrage Runs per Season for {batter_name}')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig)

        batter_analysis(selected_batter)
        batter(selected_batter)

if option == "City Analysis":
    st.title('City Analysis')
    city_df = df[['city', 'match_id', 'innings', 'stage', 'team_runs']]
    city_df = pd.merge(city_df, df_ipl[['match_id', 'season']].drop_duplicates(), on='match_id', how='left')
    city_df['season'] = city_df['season'].astype('Int64')
    city_df = city_df.groupby(['match_id', 'city','season', 'innings'])['team_runs'].max().reset_index()
    
    city_df_1 = city_df[city_df['innings'] == 1]
    uni_city = city_df_1['city'].unique().tolist()
    city_df_2 = city_df[city_df['innings'] == 2]

    t_count = city_df_1.groupby(['season', 'city'])['match_id'].count().reset_index()
    t_count = t_count.rename(columns={'match_id': 'matches_played'})

    total_match = t_count.groupby('season')['matches_played'].sum().reset_index()
    total_match = total_match.rename(columns={'matches_played': 'total_matches'})

    selected_city = st.selectbox("Select City", uni_city)
    def city_ana(city_name):
        temp = city_df_1[city_df_1['city'] == city_name]
        temp = temp.groupby(['season', 'innings'])['team_runs'].mean().reset_index()
        fig = px.line(temp, x='season', y='team_runs', title=f'Average Team Runs per Season in {city_name} in First Inning')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)

        temp2 = city_df_2[city_df_2['city'] == city_name]
        temp2 = temp2.groupby(['season', 'innings'])['team_runs'].mean().reset_index()
        fig = px.line(temp2, x='season', y='team_runs', title=f'Average Team Runs per Season in {city_name} in second Inning')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)
  
        combo = pd.concat([temp, temp2])

        fig = px.line(combo, x='season', y='team_runs',color='innings',
                title=f'Average Team Runs per Season in {city_name} (First vs Second Inning)',
                labels={'season': 'Season', 'team_runs': 'Average Runs', 'Inning': 'Inning'})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)

        count = t_count[t_count['city'] == city_name]
        fig = px.bar(count, x='season', y='matches_played', title=f'Total Matches Played in {city_name} per Season')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)

        trac1 = go.Bar(x=count['season'], y=count['matches_played'], name='Total Matches by city') # Changed go.bar to go.Bar
        trac2 = go.Bar(x=total_match['season'], y=total_match['total_matches'], name='Total Matches') # Changed go.bar to go.Bar and x-axis to total_match['season']
        data = [trac1, trac2]
        layout = go.Layout(barmode='group', title=f'Total Matches Played and Won in {city_name} per Season')
        fig = go.Figure(data=data, layout=layout)
        st.plotly_chart(fig)

        value = count['matches_played'].sum()
        value2 = total_match['total_matches'].sum()
        values = [value, value2]
        labels = [f'Matches Host by {city_name}', 'Matches Host by Other Cities']
        fig = px.pie(values=values, names=labels, title=f'Total Matches Host by {city_name}')
        st.plotly_chart(fig)
    city_ana(selected_city)