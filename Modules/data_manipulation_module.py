import pandas as pd

#Reading kinematic data.
def Append_dataframe(csv_files, df_list):
    if len(df_list) == 0:
        for filename in csv_files:
            df = pd.read_csv(filename, index_col = None, sep = ';')
            df_list.append(df)

            
#i: trials
#j: Columns of the given trial (0-76).
#NP_df_list[i].iloc[:,j]: One column of the given trial
import antropy as ant
def Approximate_entropy(df_list, return_ApEn):
    ApEn_rows = []

    for i in range(len(df_list)):
        for j in range(len(df_list[i].columns)):
            ApEn_rows.append(ant.app_entropy(df_list[i].iloc[:,j]))
        
        ApEn_series = pd.Series(ApEn_rows, index = return_ApEn.columns)
        return_ApEn = return_ApEn.append(ApEn_series, ignore_index=True)
        ApEn_rows.clear()
        ApEn_series.empty
    
    return return_ApEn


#Attaching GRS values to the data.
def Target_append(df_class, return_ApEn):
    target = df_class["GRS"]
    return_ApEn = return_ApEn.join(target)
    return return_ApEn