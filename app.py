import streamlit as st

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

df = pd.read_csv('player_advanced_stats.csv')
###print(df)
st.title("Player Efficiency Rating")
st.divider()
# Filter unique positions
position_list = df['Pos'].unique()
selected_position = st.sidebar.selectbox('Select a position:', position_list)

# Filter data based on selected position
filtered_df = df[df['Pos'] == selected_position]


# Display the list of players for the selected position in the sidebar
player_list = filtered_df['Player'].unique()
selected_player = st.sidebar.selectbox('Select a player:', player_list)

# Filter the data for the selected player
player_data = filtered_df[filtered_df['Player'] == selected_player]

# Display the selected player's data
st.header(f"Player: {selected_player} ({selected_position})")
st.write(player_data)

# Calculate average PER and Age for the selected position
average_per_age = filtered_df.groupby('Age')['PER'].mean().reset_index()

# Plotting the bar graph for player's PER vs Age
plt.figure(figsize=(10, 6))
plt.bar(player_data['Age'], player_data['PER'], color='skyblue')
plt.title(f'{selected_player} - Player Efficiency Rating (PER) vs Age')
plt.xlabel('Age')
plt.ylabel('PER')
plt.grid(True)

# Display the plot for player's PER vs Age within Streamlit
st.pyplot()

# Display header for average data
st.header(f"Average Data for {selected_position}")

# Plotting the bar graph for average PER vs Age for the selected position
plt.figure(figsize=(10, 6))
plt.bar(average_per_age['Age'], average_per_age['PER'], color='skyblue')
plt.title(f'Average Player Efficiency Rating (PER) vs Age for {selected_position}')
plt.xlabel('Age')
plt.ylabel('Average PER')
plt.grid(True)


# Display the plot for average PER vs Age within Streamlit
st.pyplot()


st.write("Note: the player efficiency rating sometimes spikes in the later years because typically, only star players continue to play at an older age. The sample size is therefore smaller including only very good players, and this leads to a higher efficiency rating than what otherwise may be expected with an increase in age.")