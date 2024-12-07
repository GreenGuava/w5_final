import warnings
import streamlit as st
import altair as alt
import pandas as pd

# Suppressing warnings so that I don't get errors when opening
warnings.filterwarnings('ignore')

# Load the Titanic dataset from my GitHub while only keeping necessary columns
dataset_url = "https://raw.githubusercontent.com/GreenGuava/w5_final/refs/heads/main/Titanic-Dataset.csv"
df = pd.read_csv(dataset_url, usecols=['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

# Renaming columns to make them more readable
df.rename(columns={
    'Pclass': 'Passenger Class',
    'Sex': 'Gender',
    'SibSp': 'Siblings/Spouses Aboard',
    'Parch': 'Parents/Children Aboard',
    'Embarked': 'Port of Embarkation'
}, inplace=True)

# Clean up the "Port of Embarkation" column to show full names instead of initials
df['Port of Embarkation'].replace({
    'C': 'Cherbourg',
    'Q': 'Queenstown',
    'S': 'Southampton'
}, inplace=True)

# Capitalizing gender values for a more consistent experience, everything else is capitalized
df['Gender'].replace({'male': 'Male', 'female': 'Female'}, inplace=True)

# Grouping age into life stage buckets
age_bins = [0, 12, 19, 45, 65, 130]
age_labels = ['Child', 'Teenage', 'Adult', 'Middle-Aged', 'Senior']
df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, include_lowest=True)

# Making sure the age groups show correctly when ordered in Streamlit
df['Age Group'] = pd.Categorical(df['Age Group'], categories=age_labels, ordered=True)

# Grouping the fares into different bins
fare_bins = [0, 25, 50, 100, 200, 600]
fare_labels = ['0-25', '25-50', '50-100', '100-200', '200+']
df['Fare Range'] = pd.cut(df['Fare'], bins=fare_bins, labels=fare_labels, include_lowest=True)

# Making sure the fare bins show correctly when ordered in Streamlit
df['Fare Range'] = pd.Categorical(df['Fare Range'], categories=fare_labels, ordered=True)

# Title and description for the app
st.title("Exploratory Titanic Survival Rate Heatmap")
st.write("The Titanic was an infamous disaster in 1912 that resulted in the loss of many lives. I built this visualization so that you could compare the different survival rates across a range of data fields that exist on the passengers. Passenger ages and fares paid have been grouped together.")

# Dropdowns for user to select fields
titanic_columns = ['Passenger Class', 'Gender', 'Age Group', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare Range', 'Port of Embarkation']

# Setting up the variables for axis selection
x_field = st.selectbox("Select X Axis:", titanic_columns, index=0)
y_field = st.selectbox("Select Y Axis:", titanic_columns, index=1)

# Make sure that the selected x and y axis are different; user should get a warning if they are the same
if x_field == y_field:
    st.warning("You should select two different values for a meaningful heatmap")
else:
    # Group data and calculate survival rate as percentage
    heatmap_df = df.groupby([x_field, y_field])['Survived'].mean().mul(100).reset_index()
    heatmap_df.rename(columns={'Survived': 'Survival Rate (%)'}, inplace=True)  # Clean up for display

    # Create heatmap with Altair
    heatmap = (
        alt.Chart(heatmap_df)
        .mark_rect()
        .encode(
            x=alt.X(
                f'{x_field}:N', 
                title=x_field, 
                sort=age_labels if x_field == 'Age Group' else fare_labels if x_field == 'Fare Range' else None
            ),
            y=alt.Y(
                f'{y_field}:N', 
                title=y_field, 
                sort=age_labels if y_field == 'Age Group' else fare_labels if y_field == 'Fare Range' else None
            ),
            color=alt.Color(
                'Survival Rate (%):Q',
                title='Survival Rate (%)',
                scale=alt.Scale(scheme='inferno')  # Nice color scheme, stands out well
            ),
            # Tooltip to show more detailed values on hover
            tooltip=[
                alt.Tooltip(f'{x_field}:N', title=x_field),
                alt.Tooltip(f'{y_field}:N', title=y_field),
                alt.Tooltip('Survival Rate (%):Q', title='Survival Rate (%)', format='.1f')
            ]
        )
        .properties(title="Survival Rate Heatmap", height=500)  # Set to 500 to use more vertical space
    )

    # Show the heatmap in Streamlit
    st.altair_chart(heatmap, use_container_width=True)
