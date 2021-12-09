# Comtor Ltda. 
import requests
import pandas as pd
import matplotlib.pyplot as plt
r = requests.get('http://3.132.77.176:1337/api/campaigns', verify=False)
att_file = r.json()['data']

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)

df = pd.DataFrame.from_dict(att_file)
df['content'] = df['attributes'].map(lambda x: x['content'])
df['Assignation'] = df['attributes'].map(lambda x: x['Assignation'])
df['Start'] = df['attributes'].map(lambda x: x['Start'])
df['Expires'] = df['attributes'].map(lambda x: x['Expires'])
df['State'] = df['attributes'].map(lambda x: x['State'])
df['score'] = df['attributes'].map(lambda x: x['score'])

df_f = df.drop(['attributes'], axis=1)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

ids = [str(i) for i in df_f['id'].to_numpy()]
scores = df_f['score'].to_numpy()
ax.bar(ids, scores)
colores = ['red','orange','yellow','green','blue','violet']*len(df.columns)
# ax.get_children()[0].set_color('red')
for i in range(0, len(df_f.columns)):
	ax.get_children()[i].set_color(colores[i])
	
plt.xlabel('ID CAMPAÑA')
plt.ylabel('NLP SCORE')
plt.title('PUNTAJE AI POR CAMPAÑA ')

plt.show()


