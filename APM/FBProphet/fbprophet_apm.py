import pandas as pd
import fbprophet
import sys
import matplotlib.pyplot as plt

mid = sys.argv[1] # 288
input_period = sys.argv[2] # The last n minutes in the dataset (ex. 6-*24*7)
output_period = sys.argv[3] # The length of prediction (ex. 60 mins)

apm_table = pd.read_csv('./1808-12.csv', names=['server_id', 'ds', 'y'])
time_usage_pair_for_each_server = {}
server_ids = set(apm_table["server_id"])

for server_id in server_ids:
    apm_data = apm_table[apm_table['server_id'] == server_id][['ds', 'y']]
    apm_data = apm_data.sort_values('ds').reset_index(drop=True)
    time_usage_pair_for_each_server[server_id] = apm_data

apm_prophet = fbprophet.Prophet(changepoint_prior_scale=0.05)
target_agent = time_usage_pair_for_each_server[int(mid)][-1*int(input_period):]
apm_prophet.fit(target_agent)
apm_forecast = apm_prophet.make_future_dataframe(periods=int(output_period), freq='min')
apm_forecast = apm_prophet.predict(apm_forecast)
selected_df = apm_forecast[-1*int(output_period):]
selected_df['yhat_lower'][selected_df['yhat_lower'] < 0] = 0
selected_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv("./Data/fbprophet-" + mid + "-" + input_period + "-" + output_period  + ".csv", index=False)
