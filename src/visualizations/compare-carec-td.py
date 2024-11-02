import pandas as pd
import os
import json
import csv
import matplotlib.pyplot as plt
import seaborn as sns

cwd = os.getcwd()
datasetname = 'beir/arguana'

carec_path=os.path.join(cwd, 'data/api', datasetname.replace("/", "-"), 'all_batches.json')
td_path=os.path.join(cwd, 'data/spacy/all_tokens', datasetname.replace('/', '-')+'.jsonl')
data_path = os.path.join(cwd, 'data/visualizations/',datasetname, '/td-carec/')
score_mapping={
    'flesch_reading_ease': 'FRE.sc',
    'flesch_kincaid_grade': 'FKGL.sc',
    'smog': 'SMOG.sc',
    'automated_readability_index': 'ARI.sc',
    'sentence_length_mean': 'ARI.comps.0.5 * Average sentence length',
    'token_length_mean':'ARI.comps.4.71 * Average number of characters per word',
    'syllables_per_token_mean':'FRE.comps.-84.6 * Average number of syllables per word',

}

def read_jsonl(path) -> pd.DataFrame:
    df = pd.read_json(path_or_buf=path, lines=True)
    return df
    

if __name__ == '__main__':
    carec_data = read_jsonl(carec_path)
    td_data = read_jsonl(td_path).rename(columns={"docno":"doc_id"})

    merged = carec_data.merge(td_data, on="doc_id").set_index("doc_id").drop(columns=['text'])

    corr_series = pd.Series(index=score_mapping.keys())
    error_df = pd.DataFrame(index=merged.index)
    for score_td, score_carec in score_mapping.items():
        corr = merged[score_td].corr(merged[score_carec])
        corr_series.update({score_td: corr})
        error = merged[score_td] - merged[score_carec]
        error_df[score_td] = error
        
        
    all_correlations=merged.corr()
    plt.figure(figsize=(15, 10))
    sns.set_theme(font_scale=0.5)
    sns.heatmap(all_correlations, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    #plt.savefig(data_path+'corr.pdf')
    plt.show()
    print(corr_series)
    print(error_df)    