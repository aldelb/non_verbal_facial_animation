import pandas as pd

def createFinalFile(path_data_out, key, df_list):
    df = []
    for new_df in df_list:
        if(len(df) == 0):
            df = new_df
        else:
            df = pd.concat([df, new_df], ignore_index=True)
       
    if(len(df) > 0):
        df = df.groupby('timestamp').mean().reset_index()
        df.set_index("timestamp", inplace =True)
        save_file = path_data_out + key + ".csv"
        df.to_csv(save_file)