import pandas as pd

kg = pd.read_csv('data/raw/kg.csv', low_memory=False)
dd = kg[kg['relation'] == 'drug_drug'][['x_id', 'y_id']].head(10000)
dd_rev = dd.rename(columns={'x_id': 'y_id', 'y_id': 'x_id'})
overlap = pd.merge(dd, dd_rev)
print(f'drug_drug sample: {len(dd)}, reverse found: {len(overlap)}')

th = kg[kg['relation'].isin(['indication', 'contraindication'])]
print(f'Indication edges: {(th["relation"] == "indication").sum()}')
print(f'Contraindication edges: {(th["relation"] == "contraindication").sum()}')
print(f'Unique diseases in therapeutic: {th["y_id"].nunique()}')
print(f'Unique drugs in therapeutic: {th["x_id"].nunique()}')
print(f'Diseases with indication: {th[th["relation"]=="indication"]["y_id"].nunique()}')
print(f'Diseases with contraindication: {th[th["relation"]=="contraindication"]["y_id"].nunique()}')
