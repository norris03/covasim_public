import pandas as pd

li_pop = {
  '0-9':  4015,
  '10-19': 4017,
  '20-29': 4723,
  '30-39': 4971,
  '40-49': 5250,
  '50-59': 6118,
  '60-69': 4932,
  '70-79': 3486,
  '80+':   1612,
}

case_data_file_name = 'WHO-COVID-19-global-data'

df = pd.read_csv(f'{case_data_file_name}.csv')

df_li = df[df['Country_code'] == 'LI']

df_li = df_li.head(66)
#217 for total

print(df_li)

df_li = df_li[['Date_reported', 'Cumulative_cases', 'Cumulative_deaths']].rename(
    columns={
        'Date_reported': 'date',
        'Cumulative_cases': 'cum_infections',
        'Cumulative_deaths': 'cum_deaths'
    }
)

df_li.to_csv('data.csv', index=False)