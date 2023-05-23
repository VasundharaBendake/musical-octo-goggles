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


# In[28]:


plt.rcParams['figure.figsize'] = (20,15)
ax = sns.heatmap(loadings_df, annot=True, cmap='Spectral')
plt.show()


# In[29]:


get_ipython().system('pip install bioinfokit')


# In[30]:


from bioinfokit.visuz import cluster
cluster.screeplot(obj=[pc_list, pca.explained_variance_ratio_],show=True,dim=(10,5))


# In[31]:


pca_scores = PCA().fit_transform(x)
cluster.biplot(cscore=pca_scores, loadings=loadings, labels=df.columns.values, 
               var1=round(pca.explained_variance_ratio_[0]*100, 2),
    var2=round(pca.explained_variance_ratio_[1]*100, 2),show=True,dim=(10,5))


# In[32]:


get_ipython().system(' pip install yellowbrick')


# In[33]:


from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,12)).fit(df_eleven)
visualizer.show()


# In[34]:


kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0).fit(df_eleven)
df['cluster_num'] = kmeans.labels_ 
print (kmeans.labels_) 
print (kmeans.inertia_)  
print(kmeans.n_iter_) 
print(kmeans.cluster_centers_) 


# In[35]:


from collections import Counter
Counter(kmeans.labels_)


# In[36]:


sns.scatterplot(data=pf, x="pc1", y="pc2", hue=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            marker="X", c="r", s=80, label="centroids")
plt.legend()
plt.show()


# In[37]:


from statsmodels.graphics.mosaicplot import mosaic
from itertools import product
crosstab =pd.crosstab(df['cluster_num'],df['Like'])
crosstab = crosstab[['-5','-4','-3','-2','-1','0','+1','+2','+3','+4','+5']]
crosstab 


# In[38]:


plt.rcParams['figure.figsize'] = (7,5)
mosaic(crosstab.stack())
plt.show()


# In[39]:


crosstab_gender =pd.crosstab(df['cluster_num'],df['Gender'])
crosstab_gender


# In[40]:


plt.rcParams['figure.figsize'] = (7,5)
mosaic(crosstab_gender.stack())
plt.show()


# In[41]:


sns.boxplot(x="cluster_num", y="Age", data=df)


# In[42]:


df['VisitFrequency'] = LabelEncoder().fit_transform(df['VisitFrequency'])
visit = df.groupby('cluster_num')['VisitFrequency'].mean()
visit = visit.to_frame().reset_index()
visit


# In[43]:


df['Like'] = LabelEncoder().fit_transform(df['Like'])
Like = df.groupby('cluster_num')['Like'].mean()
Like = Like.to_frame().reset_index()
Like


# In[44]:


df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
Gender = df.groupby('cluster_num')['Gender'].mean()
Gender = Gender.to_frame().reset_index()
Gender


# In[45]:


segment = Gender.merge(Like, on='cluster_num', how='left').merge(visit, on='cluster_num', how='left')
segment


# In[46]:


plt.figure(figsize = (9,4))
sns.scatterplot(x = "VisitFrequency", y = "Like",data=segment,s=400, color="r")
plt.title("Simple segment evaluation plot for the fast food data set",
          fontsize = 15) 
plt.xlabel("Visit", fontsize = 12) 
plt.ylabel("Like", fontsize = 12) 
plt.show()


# In[ ]:




