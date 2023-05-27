#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df= pd.read_csv("https://homepage.boku.ac.at/leisch/MSA/datasets/mcdonalds.csv")


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.dtypes


# In[9]:


df.nunique()


# In[10]:


df.isnull().sum()


# In[11]:


(df.isnull().sum()/(len(df)))*100


# In[12]:


df['Gender'].value_counts()
df['VisitFrequency'].value_counts()
df['Like'].value_counts()


# In[13]:


labels = ['Female', 'Male']
size = df['Gender'].value_counts()
colors = ['pink', 'green']
explode = [0, 0]
plt.rcParams['figure.figsize'] = (5,5)
plt.pie(size, colors = colors, explode = explode, labels = labels, autopct = '%.2f%%')
plt.title('Gender', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()


# In[14]:


plt.rcParams['figure.figsize'] = (22, 8)
f = sns.countplot(x=df['Age'],palette = 'hsv')
f.bar_label(f.containers[0])
plt.title('Customers Age Distribution')
plt.show()


# In[15]:


df['Like']= df['Like'].replace({'I hate it!-5': '-5','I love it!+5':'+5'})
sns.catplot(x="Like", y="Age",data=df, 
            orient="v", height=5, aspect=2, palette="Set2",kind="swarm")
plt.title('Likeness with repect to Age')
plt.show()


# In[17]:


from sklearn.preprocessing import LabelEncoder
def labelling(x):
    df[x] = LabelEncoder().fit_transform(df[x])
    return df

cat = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast', 'cheap',
       'tasty', 'expensive', 'healthy', 'disgusting']

for i in cat:
    labelling(i)


# In[18]:


df


# In[19]:


plt.rcParams['figure.figsize'] = (12,16)
df.hist()
plt.show()


# In[20]:


df_eleven = df.loc[:,cat]
df_eleven


# In[21]:


x = df.loc[:,cat].values
x


# In[22]:


from sklearn.decomposition import PCA
from sklearn import preprocessing


# In[23]:


pca_data = preprocessing.scale(x)
pca = PCA(n_components=11)
pc = pca.fit_transform(x)
names = ['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11']
pf = pd.DataFrame(data = pc, columns = names)


# In[24]:


pf


# In[25]:


pca.explained_variance_ratio_


# In[26]:


np.cumsum(pca.explained_variance_ratio_)


# In[27]:


loadings = pca.components_
num_pc = pca.n_features_
pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
loadings_df['variable'] = df_eleven.columns.values
loadings_df = loadings_df.set_index('variable')
loadings_df





