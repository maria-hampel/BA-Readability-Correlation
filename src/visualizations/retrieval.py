import pandas as pd 
import os
import matplotlib.pyplot as plt 
import seaborn as sns 

top10_datasets = ['data/retrieval/classic/top10.pkl.gzip', 'data/retrieval/distilbert/top10_distilbert.pkl.gzip', 'data/retrieval/roberta/top10_roberta.pkl.gzip','data/retrieval/jina/top10_jina.pkl.gzip']
allretrieved_datasets = ['data/retrieval/classic/allretrieved.pkl.gzip', 'data/retrieval/distilbert/allretrieved_distilbert.pkl.gzip', 'data/retrieval/roberta/allretrieved_roberta.pkl.gzip', 'data/retrieval/jina/allretrieved_jina.pkl.gzip']

readability_datasets = ['data/spacy/all_tokens/beir-arguana.jsonl', 'data/spacy/all_tokens/beir-nfcorpus.jsonl', 'data/spacy/all_tokens/beir-scifact.jsonl']
carec_datasets = ['data/api/beir-arguana/all_batches.json', 'data/api/beir-nfcorpus/all_batches.json', 'data/api/beir-scifact/all_batches.json']
importantstuff = ['docno', 'qid', 'rank', 'score', 'system']
metrics = ['flesch_reading_ease', 'flesch_kincaid_grade', 'smog', 'automated_readability_index', 'lix', 'rix', 'ARI.sc', 'CAREC.sc', 'CAREC_M.sc','FKGL.sc', 'FRE.sc', 'SMOG.sc']


if __name__ == '__main__':
    top10list = []
    for item in top10_datasets:
        df = pd.read_pickle(item, compression='gzip')
        df.drop(df[df['rank'] == 10].index, inplace=True)
        df['docno'] = df['docno'].astype(str)
        top10list.append(df)
    top10df = pd.concat(top10list)
    print(top10df.columns)
    
    allretrievedlist = []
    for item in allretrieved_datasets:
        df = pd.read_pickle(item , compression='gzip')
        df['docno'] = df['docno'].astype(str)
        allretrievedlist.append(df)
    allretrieveddf = pd.concat(allretrievedlist)
    print(allretrieveddf.columns)
    
    readabilitydatalist =[]
    for item in readability_datasets: 
        df = pd.read_json(item, lines=True)
        df['docno'] = df['docno'].astype(str)  
        readabilitydatalist.append(df)
    readabilitydf = pd.concat(readabilitydatalist)

    
    careclist =[]
    for item in carec_datasets:
        df = pd.read_json(item, lines=True)
        df.rename(columns={'doc_id': 'docno'}, inplace=True)
        df['docno'] = df['docno'].astype(str) 
        careclist.append(df)
    carecdf = pd.concat(careclist)
    
    top10df = top10df.join(readabilitydf.set_index('docno'), on='docno')
    top10df = top10df.join(carecdf.set_index('docno'), on='docno')
    
    print(top10df.head())
    print(top10df.columns)
    print(top10df.shape)
    
    top10df = top10df.loc[:, importantstuff+metrics]
    
    alldocs = readabilitydf.join(carecdf.set_index('docno'), on='docno')
    alldocs = alldocs.loc[:, ['docno']+metrics]
    
    with open('data/retrieval/stats.md', 'a') as file:
        for item in metrics:
            stats = top10df[['system', item]].groupby('system').describe()
            text = stats.to_markdown()
            file.write('# '+item+'\n'+text+'\n')
            
        text = alldocs.describe().to_markdown()
        file.write('# All Documents\n'+text)
    
    
    
    # boxpldf = top10df
    # metric = "flesch_reading_ease" 
    # all_models = sns.boxplot(data=boxpldf, y=metric, hue="system", showfliers=False)

    # sns.move_legend(all_models, "upper left", bbox_to_anchor=(1, 1))
    # plt.show()
    
    
    
    
    