from pathlib import Path
import pandas as pd
import plotly.express as px

from pandas_profiling import ProfileReport


dff = pd.read_csv(Path('data/dengue_features.csv'))
dfy = pd.read_csv(Path('data/dengue_labels.csv'))
df = dff.merge(dfy)



if __name__ == '__main__':

    profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
    profile.to_file(Path("eda").joinpath("dengue.html"))



    fig = px.line(df[['week_start_date', 'total_cases', 'city']], color='city', x='week_start_date', y='total_cases')
    fig.update_layout(title="Cases By City")
    fig.write_html(Path("eda").joinpath("CasesByCity.html"))