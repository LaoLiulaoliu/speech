import pandas as pd
import os
import json
import re
import warnings
warnings.filterwarnings("ignore", 'This pattern has match groups')

def clean_text(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower()).strip()

path = '/kaggle/input/coleridgeinitiative-show-us-the-data'

stopwords = ['ourselves', 'hers','the', 'between', 'yourself', 'but', 'again','of', 'there', 'about',
             'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some',
             'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is',
             's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below',
             'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more',
             'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to',
             'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and',
             'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what',
             'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has',
             'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom',
             't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 
             'was', 'here', 'than']


df_train = pd.read_csv(path + '/train.csv')
print(df_train.head())
print(df_train.shape)  # (19661, 5)
for col in df_train.columns:
    print(f"{col}: {len(df_train[col].unique())}")

# Id: 14316
# pub_title: 14271
# dataset_title: 45
# dataset_label: 130
# cleaned_label: 130


df_input = pd.DataFrame(columns=['id', 'section_title', 'text', 'data_label'])
for ID in df_train['Id'].values:
    df = pd.read_json(path + '/train/{}.json'.format(ID))

    for data_label in df_train[df_train['Id'] == ID]['dataset_label'].values:
        new_df = df[df['text'].str.contains(data_label)].copy(deep=True)
        new_df['data_label'] = data_label
        new_df['id'] = ID
        new_df.reset_index(inplace=True, drop=True)
        df_input = pd.concat([df_input, new_df], ignore_index=True, sort=False)
        df_input.reset_index(inplace=True, drop=True)




words = df_input['data_label'].values  # numpy.ndarray of String


df_test = pd.read_csv(path + '/sample_submission.csv')
df_test_input = pd.DataFrame(columns=['id', 'section_title', 'text'])
for ID in df_test['Id'].values:
    df = pd.read_json(path + '/test/{}.json'.format(ID))
    
    df['id'] = ID
    df.reset_index(inplace=True, drop=True)
    df_test_input = pd.concat([df_test_input, df], ignore_index=True, sort=False)
    df_test_input.reset_index(inplace=True,drop=True)

df_test_input['length'] = df_test_input.text.str.len()
df_test_input = df_test_input[df_test_input.length > 0]


datasets_titles = [str(x).lower() for x in df_input['data_label'].unique()]
labels = []
for index in df_test['Id']:
    publication_text = df_test_input[df_test_input['id'] == index].text.str.cat(sep='\n').lower()

    label = [clean_text(dataset_title) for dataset_title in datasets_titles if dataset_title in publication_text]
    labels.append('|'.join(label))

submission_df = pd.read_csv(path + '/sample_submission.csv', index_col=0)
submission_df['PredictionString'] = labels
submission_df.to_csv('./submission.csv')
print('submission complete')