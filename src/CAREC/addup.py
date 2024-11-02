import pandas as pd
import ast
import os
import csv


cwd = os.getcwd()

dsname = "beir/scifact"
outputpath = os.path.join(cwd, "data/api", dsname.replace("/", "-"))

dflist=[]

nested_columns = ['ARI', 'CAREC', 'CAREC_M', 'CML2RI', 'FKGL', 'FRE', 'SMOG']

def flatten_json(column):
    return column.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)


for filename in os.listdir(outputpath):
    if filename != 'state.json':
        print(filename)
        df = pd.read_csv(os.path.join(outputpath, filename), delimiter=',', quoting=csv.QUOTE_MINIMAL)
        flattened_data = pd.DataFrame()

        # Process each nested column
        for col in nested_columns:
            flat_col = flatten_json(df[col])
            
            # Normalize and concatenate
            if not flat_col.isnull().all():
                normalized = pd.json_normalize(flat_col)
                normalized.columns = [f'{col}.{c}' for c in normalized.columns]
                flattened_data = pd.concat([flattened_data, normalized], axis=1)

        df_cleaned = df.drop(columns=nested_columns)
        df_final = pd.concat([df_cleaned, flattened_data], axis=1)

        dflist.append(df_final)
        
final = pd.concat(dflist)
final = final.drop(columns=['err'])
# duplicates = df_final.columns[df_final.columns.duplicated()]
# print("Duplicate columns:", duplicates)
final.to_json(path_or_buf=os.path.join(outputpath, 'all_batches.json'), orient='records', lines=True)
final.to_csv(path_or_buf=os.path.join(outputpath, 'all_batches.csv'))