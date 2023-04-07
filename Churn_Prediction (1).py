#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
    
from sklearn.preprocessing import OneHotEncoder
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data = pd.read_csv("D:/CodeClause/WA_Fn-UseC_-Telco-Customer-Churn.csv")
data.head()


# In[4]:


data['Churn'].value_counts()


# In[5]:


data.shape


# In[6]:





# In[8]:


data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')


# In[9]:


data.isnull().sum()


# In[10]:


##Univariate Analysis
categorical_var = list(data.dtypes.loc[data.dtypes == 'object'].index)
print(len(categorical_var))
print(categorical_var)


# In[11]:


categorical_var.remove('customerID')


# In[12]:


fig, ax =plt.subplots(6,3,figsize=(12,20))


sns.countplot(data['gender'], ax=ax[0][0])
sns.countplot(data['Partner'], ax=ax[0][1])
sns.countplot(data['Dependents'], ax=ax[0][2])

sns.countplot(data['PhoneService'], ax=ax[1][0])
sns.countplot(data['MultipleLines'], ax=ax[1][1])
sns.countplot(data['InternetService'], ax=ax[1][2])

sns.countplot(data['OnlineSecurity'], ax=ax[2][0])
sns.countplot(data['OnlineBackup'], ax=ax[2][1])
sns.countplot(data['DeviceProtection'], ax=ax[2][2])

sns.countplot(data['TechSupport'], ax=ax[3][0])
sns.countplot(data['StreamingTV'], ax=ax[3][1])
sns.countplot(data['StreamingMovies'], ax=ax[3][2])

sns.countplot(data['Contract'], ax=ax[4][0])
sns.countplot(data['PaperlessBilling'], ax=ax[4][1])
sns.countplot(data['PaymentMethod'], ax=ax[4][2])

sns.countplot(data['Churn'], ax=ax[5][0])

fig.show()


# In[13]:


continuous_var = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges']
data.describe()


# In[14]:


nd = pd.melt(data, value_vars = continuous_var)
n1 = sns.FacetGrid (nd, col='variable', col_wrap=2, sharex=False, sharey = False)
n1 = n1.map(sns.distplot, 'value')
n1


# In[15]:


corr = data[continuous_var].corr()
sns.heatmap(corr)


# In[16]:


print (corr['TotalCharges'].sort_values(ascending=False), '\n')


# In[17]:


sns.jointplot(x=data['TotalCharges'], y=data['tenure'])


# In[18]:


for var in categorical_var:
    if var!='Churn':
        test = data.groupby([var,'Churn'])
        print(test.size(),'\n\n')


# In[19]:


import scipy.stats as stats
from scipy.stats import chi2_contingency

class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        
        self.dfObserved = None
        self.dfExpected = None
        
    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p<alpha:
            result="{0} is IMPORTANT for Prediction".format(colX)
        else:
            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)

        print(result)
        
    def TestIndependence(self,colX,colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        
        self.dfObserved = pd.crosstab(Y,X) 
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        
        self._print_chisquare_result(colX,alpha)

df = data
#Initialize ChiSquare Class
cT = ChiSquare(df)

#Feature Selection
for var in categorical_var:
    cT.TestIndependence(colX=var,colY="Churn" ) 


# In[20]:


##3. Categorical-Continuous Variables
#ANOVA test


# In[21]:


# ANOVA test
import scipy.stats as stats
    
for var in continuous_var:    
    result = stats.f_oneway(data[var][data['Churn'] == 'Yes'], 
                            data[var][data['Churn'] == 'No'])
    print(var)
    print(result)


# In[22]:


#t-test/z-test

from sklearn.feature_selection import SelectKBest
from scipy.stats import ttest_ind

t_stat = []
for var in continuous_var:
    var_no_churn = data[var][data["Churn"] == "No"]
    var_yes_churn = data[var][data["Churn"] == "Yes"]
    t_value = ttest_ind(var_no_churn, var_yes_churn, equal_var=False)
    print(var)
    print(t_value)
    #t_stat.append(t_value)


# In[23]:


data.isnull().sum()


# In[24]:


data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)


# In[25]:


categorical_var


# In[26]:


#first convert all the string columns to categorical form
for var in categorical_var:
    data[var] = data[var].astype('category')


# In[27]:


data[categorical_var] = data[categorical_var].apply(lambda x: x.cat.codes)


# In[28]:


target = data['Churn']
data=data.drop('customerID',axis=1)
all_columns = list(data.columns)
all_columns.remove('Churn')


# In[29]:


import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

X = data[all_columns] # Features
y = data['Churn'] # Target variable

# Feature extraction
model = LogisticRegression()
rfe = RFE(model, 8)
fit = rfe.fit(X, y)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))


# In[30]:


selected_features_rfe = list(fit.support_)


# In[31]:


##Model buidling
#using RFE + logistic_regression


# In[32]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

final_features_rfe = []    
for status, var in zip(selected_features_rfe, all_columns):
    if status == True:
        final_features_rfe.append(var)
        
final_features_rfe


# In[33]:


X_rfe_lr = data[final_features_rfe]
y = data['Churn']

X_train_rfe_lr,X_test_rfe_lr,y_train_rfe_lr,y_test_rfe_lr=train_test_split(X_rfe_lr,y,test_size=0.25,random_state=0)

lr_model = LogisticRegression()

# fit the model with data
lr_model.fit(X_train_rfe_lr,y_train_rfe_lr)
y_pred_rfe_lr=lr_model.predict(X_test_rfe_lr)

acc_rfe_lr = metrics.accuracy_score(y_test_rfe_lr, y_pred_rfe_lr)
print("Accuracy: ",acc_rfe_lr)


# In[34]:


#Logistic regression


# In[35]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
# instantiate the model (using the default parameters)
lr_model_single = LogisticRegression()

# fit the model with data
lr_model_single.fit(X_train,y_train)
y_pred=lr_model_single.predict(X_test)

lr_acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: ",lr_acc)


# In[36]:


# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[37]:


from sklearn.metrics import roc_curve, auc
fpr_1, tpr_1, thresholds = roc_curve(y_test, y_pred_rfe_lr)
fpr_2, tpr_2, thresholds = roc_curve(y_test, y_pred)
roc_auc_1 = auc(fpr_1, tpr_1)
roc_auc_2 = auc(fpr_2, tpr_2)


# In[38]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr_1,tpr_1, color='red',label = 'AUC = %0.2f' % roc_auc_1)
plt.plot(fpr_2,tpr_2, color='green',label = 'AUC = %0.2f' % roc_auc_2)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# In[ ]:




