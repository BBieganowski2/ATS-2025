# %%
import pandas as pd
import glob
import os
import json


# %%
latest_backtest = max(os.listdir('backtests'), key=lambda x: os.path.getmtime(os.path.join('backtests', x)))
content = open(os.path.join('backtests', latest_backtest), 'r').readlines()

# %%
ts_start = [content.index(x) for x in content if 'Trade History' in x][0]
sandbox_log = [content.index(x) for x in content if 'Sandbox logs' in x][0]
al_history = [content.index(x) for x in content if 'Activities log' in x][0]

ts_start, sandbox_log, al_history

# %%
sandbox = content[sandbox_log+1:al_history-3]
sandbox = [x.strip() for x in sandbox]
json_string = "[" + "".join(sandbox).replace("}{", "},{") + "]"
logs = pd.DataFrame(json.loads(json_string))
logs.to_csv('sandbox_logs.csv')

# %%
activity_logs = content[al_history+1:ts_start-4]
df = pd.DataFrame(activity_logs)[0].str.strip().str.split(';', expand=True)
df.columns = df.loc[0]
activity_logs_df = df.iloc[1:]
activity_logs_df
activity_logs_df.to_csv('activity_logs.csv')

# %%
activity_logs_df

# %%
trades = content[ts_start+1:]

# %%
json_str = "".join(trades)

json_str = json_str.replace(",\n  }", "\n  }")  # Remove trailing commas
json_str = json_str.replace(",\n]", "\n]")  # Remove trailing comma 
data = json.loads(json_str)
trades_df = pd.DataFrame(data)
trades_df.to_csv('trades.csv')

# %%
trades_df

# %%



