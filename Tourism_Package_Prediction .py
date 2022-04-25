#!/usr/bin/env python
# coding: utf-8

# # **Problem Statement:**
# 
# You are a Data Scientist for a tourism company named "Visit with us". The Policy Maker of the company wants to enable and establish a viable business model to expand the customer base.
# 
# A viable business model is a central concept that helps you to understand the existing ways of doing the business and how to change the ways for the benefit of the tourism sector.
# 
# One of the ways to expand the customer base is to introduce a new offering of packages.
# 
# Currently, there are 5 types of packages the company is offering - Basic, Standard, Deluxe, Super Deluxe, King. Looking at the data of the last year, we observed that 18% of the customers purchased the packages.
# 
# However, the marketing cost was quite high because customers were contacted at random without looking at the available information.
# 
# The company is now planning to launch a new product i.e. Wellness Tourism Package. Wellness Tourism is defined as Travel that allows the traveler to maintain, enhance or kick-start a healthy lifestyle, and support or increase one's sense of well-being.
# 
# However, this time company wants to harness the available data of existing and potential customers to make the marketing expenditure more efficient.
# 
# You as a Data Scientist at "Visit with us" travel company has to analyze the customers' data and information to provide recommendations to the Policy Maker and Marketing Team and also build a model to predict the potential customer who is going to purchase the newly introduced travel package.

# # Objective:
# 
# To build a model to predict which customer is potentially going to purchase the newly introduced travel package.

# # **Data Description:**
# 
# - CustomerID: Unique customer ID
# - ProdTaken: Product taken flag
# - Age: Age of customer
# - PreferredLoginDevice: Preferred login device of the customer in last month
# - CityTier: City tier
# - DurationOfPitch: Duration of pitch by a sales man to customer
# - Occupation: Occupation of customer
# - Gender: Gender of customer
# - NumberOfPersonVisiting: Total number of person came with customer
# - NumberOfFollowups: Total number of follow up has been done by sales person after sales pitch
# - ProductPitched: Product pitched by sales person
# - PreferredPropertyStar: Preferred hotel property rating by customer
# - MaritalStatus: Marital status of customer
# - NumberOfTrips: Average number of trip in a year by customer
# - Passport: Customer passport flag
# - PitchSatisfactionScore: Sales pitch satisfactory score
# - OwnCar: Customers owns a car flag
# - NumberOfChildrenVisiting: Total number of children with age less than 5 visit with customer
# - Designation: Designation of customer in current organization
# - MonthlyIncome: Gross monthly income of customer

# ### Let's start by importing necessary libraries

# In[1]:


# Library to suppress warnings or deprecation notes 
import warnings
warnings.filterwarnings('ignore')

# Libraries to help with reading and manipulating data
import numpy as np
import pandas as pd

# Libraries to help with data visualization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# Libraries to split data, impute missing values 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Libraries to import decision tree classifier and different ensemble classifiers
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier

# Libtune to tune model, get different metric scores
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV


# ### Load and overview the dataset

# In[2]:


#Loading the dataset - sheet_name parameter is used if there are Basicple tabs in the excel file.
data=pd.read_excel("Tourism.xlsx",sheet_name="Tourism")


# In[52]:


data.head()


# In[64]:


age1 = 100
age2 = 20
age = age2 -age1
print(abs(age))


# In[4]:


data.info()


# - There are total 20 columns and 4,888 observations in the dataset
# - We can see that 8 columns have less than 4,888 non-null values i.e. columns have missing values.

# **Check the percentage of missing values in each column**

# In[48]:


pd.DataFrame(data={'% of Missing Values':round(data.isna().sum()/data.isna().count()*100,2)})


# - `Age` column has 4.62% missing values out of the total observations.
# - `PreferredLoginDevice` column has 0.51% missing values out of the total observations.
# - `DurationOfPitch` column has 5.14% missing values out of the total observations.
# - `NumberOfFollowups` column has 0.92% missing values out of the total observations.
# - `PreferredPropertyStar` column has 0.53% missing values out of the total observations.
# - `NumberOfTrips` column has 2.86% missing values out of the total observations.
# - `NumberOfChildrenVisited` column has 1.35% missing values out of the total observations.
# - `MonthlyIncome` column has 4.77% missing values out of the total observations.
# - We will impute these values after we split the data into train and test sets.

# **Let's check the number of unique values in each column**

# In[6]:


data.nunique()


# - We can drop the column - CustomerID as it is unique for each customer and will not add value to the model.
# - Most of the variables are categorical except - Age, duration of pitch, monthly income  and number of trips of customers.

# In[7]:


#Dropping CustomerID column
data.drop(columns='CustomerID',inplace=True)


# **Summary of the data**

# In[8]:


data.describe().T


# - Mean and median of age column are very close to each other i.e. approx 37 and 36 respectively.
# - Duration of pitch has some outliers at the right end as 75th percentile value is 20 and max value is 127. We need to explore this further.
# - It seems like monthly income has some outliers at both ends. We need to explore this further.
# - Number of trips also has some outliers as 75th percentile value is 4 and max value is 22.
# - We can see that the target variable - ProdTaken is imbalanced as most of the values are 0. 

# **Let's check the count of each unique category in each of the categorical variables.** 

# In[9]:


#Making a list of all catrgorical variables 
cat_col=['PreferredLoginDevice', 'CityTier','Occupation', 'Gender', 'NumberOfPersonVisited',
       'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar',
       'MaritalStatus', 'Passport', 'PitchSatisfactionScore',
       'OwnCar', 'NumberOfChildrenVisited', 'Designation']

#Printing number of count of each unique value in each column
for column in cat_col:
    print(data[column].value_counts())
    print('-'*50)


# - Free lancer category in occupation column has just 2 entries out of 4,888 observations.
# - We can see that Gender has 3 unique values which includes - 'Fe Male' and 'Female'. This must be a data input error, we should replace 'Fe Male' with 'Female'.
# - NumberOfPersonVisited equal to 5 has count equal to 3 only.
# - Majority of the customers are married.
# - Majority of the customers owns a car.

# In[10]:


#Replacing 'Fe Male' with 'Female'
data.Gender=data.Gender.replace('Fe Male', 'Female')


# In[11]:


#Converting the data type of each categorical variable to 'category'
for column in cat_col:
    data[column]=data[column].astype('category')


# In[12]:


data.info()


# # EDA

# ## Univariate Analysis

# In[13]:


# While doing uni-variate analysis of numerical variables we want to study their central tendency 
# and dispersion.
# Let us write a function that will help us create boxplot and histogram for any input numerical 
# variable.
# This function takes the numerical column as the input and returns the boxplots 
# and histograms for the variable.
# Let us see if this help us write faster and cleaner code.
def histogram_boxplot(feature, figsize=(15,15), bins = None):
    """ Boxplot and histogram combined
    feature: 1-d feature array
    figsize: size of fig (default (9,8))
    bins: number of bins (default None / auto)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(nrows = 2, # Number of rows of the subplot grid= 2
                                           sharex = True, # x-axis will be shared among all subplots
                                           gridspec_kw = {"height_ratios": (.25, .75)}, 
                                           figsize = figsize 
                                           ) # creating the 2 subplots
    sns.boxplot(feature, ax=ax_box2, showmeans=True, color='violet') # boxplot will be created and a star will indicate the mean value of the column
    sns.distplot(feature, kde=F, ax=ax_hist2, bins=bins,palette="winter") if bins else sns.distplot(feature, kde=False, ax=ax_hist2) # For histogram
    ax_hist2.axvline(np.mean(feature), color='green', linestyle='--') # Add mean to the histogram
    ax_hist2.axvline(np.median(feature), color='black', linestyle='-') # Add median to the histogram


# ### Observations on Age

# In[14]:


histogram_boxplot(data['Age'])


# - Age distribution looks approximately normally distributed.
# - The boxplot for age column confirms that there are no outliers for this variable
# - Age can be an important variable while targeting customers for tourism package. We will further explore this in bivariate analysis.

# ### Observations on Duration of Pitch

# In[15]:


histogram_boxplot(data['DurationOfPitch'])


# - The distribution for duration of pitch is right skewed.
# - Duration of the pitch for most of the customers is less than 20 minutes.
# - There are some observations which can be considered as outliers as they are very far form the upper whisker in the boxplot. Let's check how many such extreme values are there.

# In[16]:


data[data['DurationOfPitch']>40]


# - We can see that there are just two observations which can be considered as outliers.

# ### Observations on Monthly Income

# In[17]:


histogram_boxplot(data['MonthlyIncome'])


# - The distribution for monthly income shows that most the values lies between 20,000 to 40,000.
# - Income is on of the important factors to consider while approaching a customers with a certain package. We can explore this further in bivariate analysis. 
# - There are some observation on the left and some observation on the right of the boxplot which can be considered as outliers. Let's check how many such extreme values are there. 

# In[18]:


data[(data.MonthlyIncome>40000) | (data.MonthlyIncome<12000)]


# - There are just four such observations which can be considered as outliers.

# ### Observations on Number of Trips

# In[19]:


histogram_boxplot(data['NumberOfTrips'])


# - The distribution for number of trips is right skewed 
# - Boxplot shows that number of trips has some outliers at the right end. Let's check how many such extreme values are there. 

# In[20]:


data.NumberOfTrips.value_counts(normalize=True)


# - We can see that most the customers i.e. 52% have taken 2 or 3 number of trips.
# - As expected, with the increase in the number of trips the percentage of customers is decreasing.
# - The percentage of categories 19 or above is very less. We can consider these values as outliers.
# - We can see that there are just four observations with number of trips 19 or greater

# **Removing these outliers form duration of pitch, monthly income, and number of trips.**

# In[21]:


#Dropping observaions with duration of pitch greater than 40. There are just 2 such observations
data.drop(index=data[data.DurationOfPitch>37].index,inplace=True)

#Dropping observation with monthly income less than 12000 or greater than 40000. There are just 4 such observations
data.drop(index=data[(data.MonthlyIncome>40000) | (data.MonthlyIncome<12000)].index,inplace=True)

#Dropping observations with number of trips greater than 8. There are just 4 such observations
data.drop(index=data[data.NumberOfTrips>10].index,inplace=True)


# **Let's define a function to create barplots for the categorical variables indicating percentage of each category for that variables.**

# In[22]:


def perc_on_bar(feature):
    '''
    plot
    feature: categorical feature
    the function won't work if a column is passed in hue parameter
    '''
    #Creating a countplot for the feature
    sns.set(rc={'figure.figsize':(10,5)})
    ax=sns.countplot(x=feature, data=data)
    
    total = len(feature) # length of the column
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total) # percentage of each class of the category
        x = p.get_x() + p.get_width() / 2 - 0.1 # width of the plot
        y = p.get_y() + p.get_height()           # hieght of the plot
        ax.annotate(percentage, (x, y), size = 14) # annotate the percantage 
        
    plt.show() # show the plot


# ### Observations on Number of Person Visited

# In[23]:


perc_on_bar(data['NumberOfPersonVisited']) 


# - Most customers have 3 persons who visited with them. This can be because most people like to travel with family.
# - As mentioned earlier, there are just 3 observations where number of persons visited with the customer are 5 i.e. 0.1%.

# ### Observations on Occupation

# In[24]:


perc_on_bar(data.Occupation)


# - Majority of customers i.e. 91% are either salaried or owns a small business. 
# - As mentioned earlier, free lancer category have only 2 observations.

# ### Observations on City Tier

# In[25]:


perc_on_bar(data.CityTier)


# - Most of the customers i.e. approx 65% are from tier 1 cities. This can be because of better living standards and exposure as compared to tier 2 and tier 3 cities.
# - Surprisingly, tier 3 cities have much higher count than tier 2 cities. This can be because the company have less marketing in tier 2 cities.

# ### Observations on Gender

# In[26]:


perc_on_bar(data.Gender)


# - Male customers are more than the number of female customers
# - There are approx 60% male customers as compared to 40% female customers
# - This might be because males do the booking/inquiry when traveling with females which implies that males are the direct customers of the company.

# ### Observations on Number of Follow ups

# In[27]:


perc_on_bar(data.NumberOfFollowups)


# - We can see that company usually follows up with 3 or 4 times with their customers
# - We can explore this further and observe which number of follow ups have more customers who buys the product.

# ### Observations on Product Pitched

# In[28]:


perc_on_bar(data.ProductPitched)


# - The company pitches Deluxe or Basic packages to their customers more than the other packages. 
# - This might be because the company makes more profit from Deluxe or Basic packages or these packages are less expensive, so preferred by majority of the customers.

# ### Observations on Preferred Property Star

# In[29]:


perc_on_bar(data.PreferredPropertyStar)


# - Approx 61% customers prefer the three star property.
# - Approx 39% customers prefer 4 or 5 star properties. These can be the high income customers with high income.

# ### Observations on Preferred Login Device

# In[30]:


perc_on_bar(data.PreferredLoginDevice)


# - There are approx 70% customers who reached out to the company first i.e. self-inquiry. 
# - This shows the positive outreach of the company as most of the inquires are initiated from the customer's end.

# ### Observations on Marital Status

# In[31]:


perc_on_bar(data.MaritalStatus)


# - Approx half of the customer base of the company is from the married people. 
# - This might be because company offers more couple friendly or family packages.

# ### Observations on Passport

# In[32]:


perc_on_bar(data.Passport)


# - Most of the customers i.e. approx 71% do not have a passport
# - Company can provide services to help customers with getting new or renewing their passport as most of the customers do not have passport

# ### Observations on Pitch Satisfaction Score

# In[33]:


perc_on_bar(data.PitchSatisfactionScore)


# - Average i.e. 3 is the most common pitch satisfaction score given by customers.
# - We can explore this further and observe which satisfaction score have more customers who actually buy the product.

# ### Observations on Designation

# In[34]:


perc_on_bar(data.Designation)


# - Approx 73% of the customers are at executive or manager level.
# - We can see that the higher the position, the lesser number of observations which makes sense as executives/managers are more common than AVP/VP. 

# ### Observations on Number of Children Visited

# In[35]:


perc_on_bar(data.NumberOfChildrenVisited)


# - Approx 78% customers visit with their children and approx 34% of them have more than 1 child with them.
# - 22% customers visit without children. These may be the single/unmarried customers or recently married.

# ### Observations on Product Taken

# In[38]:


perc_on_bar(data.ProdTaken)


# - This plot shows the distribution of both classes in the target variable is `imbalanced`.
# - We only have approx 19% customers who have purchased the product.

# ## Bivariate Analysis

# In[42]:


sns.pairplot(data=data, hue='ProdTaken')


# - There are overlaps i.e. no clear distinction in the distribution of variables for people who have taken the product and did not take the product.
# - Let's explore this further with the help of other plots.

# **Let's define one more function to plot stacked bar charts**

# In[38]:


### Function to plot stacked bar charts for categorical columns
def stacked_plot(x,flag=True):
    sns.set(palette='nipy_spectral')
    tab1 = pd.crosstab(x,data['ProdTaken'],margins=True)
    if flag==True:
        print(tab1)
        print('-'*120)
    tab = pd.crosstab(x,data['ProdTaken'],normalize='index')
    tab.plot(kind='bar',stacked=True,figsize=(10,5))
    plt.legend(loc='lower left', frameon=False)
    plt.legend(loc="upper left", bbox_to_anchor=(1,1))
    plt.show()


# ### Prod Taken vs Number of Person Visited

# In[39]:


stacked_plot(data.NumberOfPersonVisited)


# - The plot shows that the conversion rate is high when number of persons are more than 1.
# - This might be because company is not providing good solo packages.
# - Conversion rate is zero when number of persons visited is 5. However, there are just 3 such observations so cannot give any conclusive insights.

# ### Prod Taken vs Number of Follow ups

# In[40]:


stacked_plot(data.NumberOfFollowups)


# - We saw earlier that company usually follows up with 3 or 4 times but this plots shows that as number of follow ups increases, conversion rate for customers increases.
# - Salesperson should ensure to follow up with the customers who are interested in buying the product.

# ### Prod Taken vs Occupation

# In[51]:





# In[41]:


stacked_plot(data.Occupation)


# - The conversion rate for large business owners is higher than salaried or small business owners. 
# - This might be because large business owner have high income.
# - Free lancer have 100% conversion rate but there are just 2 such observation, so cannot give any conclusive insights.

# ### Prod Taken vs Marital Status

# In[42]:


stacked_plot(data.MaritalStatus)


# - We have seen that married people are most common customer for the company but this graph shows that the conversion rate is higher for single and unmarried customers as compared to the married customers.
# - Company can target single and unmarried customers more and can modify packages as per these customers.

# ### Prod Taken vs Passport

# In[43]:


stacked_plot(data.Passport)


# - The conversion rate for customers with passport is higher as compared to the customers without passport.
# - The company should customize more international packages to attract more such customers.

# ### Prod Taken vs Product Pitched

# In[44]:


stacked_plot(data.ProductPitched)


# - The conversion rate of customers is higher if the product pitched is Basic. This might be because basic package is less expensive. 
# - We saw earlier that company pitches deluxe package more than the standard package, but standard package shows higher conversion rate than deluxe package. The company can pitch standard package more often.

# ### Prod Taken vs Designation

# In[45]:


stacked_plot(data.Designation)


# - The conversion rate of executives is higher than other designations.
# - Customers at VP and AVP positions have the least conversion rate.

# ### Prod Taken vs Duration of Pitch

# In[46]:


plt.figure(figsize=(15,5))
sns.boxplot(y='DurationOfPitch',x='ProdTaken',data=data)
plt.show()


# - We can clearly see that customers who purchased a package have longer duration of pitch.
# - Company sales person should give more time while pitching a certain package and convey relevant information to the customer  

# ### Prod Taken vs Monthly Income

# In[47]:


plt.figure(figsize=(15,5))
sns.boxplot(y='MonthlyIncome',x='ProdTaken',data=data)
plt.show()


# - The distribution looks right skewed for class 0 as well as class 1 which can be expected. 
# - Customers who purchased a package have lower median income than customers who did not purchase a package. This might be because of our earlier observation that executives are more likely to purchase a package.
# - Let's check this by adding the variable 'Designation' to this plot.

# ### Prod Taken vs Monthly Income vs Designation 

# In[48]:


plt.figure(figsize=(15,5))
sns.boxplot(y='MonthlyIncome',x='Designation',hue='ProdTaken',data=data)
plt.show()


# - As expected, higher the position higher the monthly income of the customer.
# - Not much difference in the income of customers at executive or manager level who did/did not purchase a package. There are many outliers for customers who purchased a package. 
# - Customers at VP or AVP positions who purchases a package have slightly lower median income.

# ### Prod Taken vs Age

# In[49]:


sns.lineplot(x='Age',y='ProdTaken',data=data)


# - This plot shows that younger people are more likely to take the product as compared to middle aged or old people.
# - There is a small peak at the age near 60. These might be people who are retired or about to be retired. 

# ### Grouping data w.r.t to packages to build customer profiles

# In[50]:


data[(data['ProductPitched']=='Basic') & (data['ProdTaken']==1)].describe(include='all').T


# - Average monthly income for customers opting for the basic package is ~20,165.
# - Average age of customers opting for the basic package is ~31
# - Majority of the customers opting for the basic package are at executive designation
# - Majority of the customers opting for the basic package are single

# In[51]:


data[(data['ProductPitched']=='Standard') & (data['ProdTaken']==1)].describe(include='all').T


# - Average monthly income of customers opting for the standard package is ~26,035.
# - Average age of for customers opting for the standard package is ~41
# - Majority of the customers opting for the standard package are at senior manager designation
# - Majority of the customers opting for the standard package are married

# In[52]:


data[(data['ProductPitched']=='Deluxe') & (data['ProdTaken']==1)].describe(include='all').T


# - Average monthly income of customers opting for the deluxe package is ~23,106.
# - Average age of for customers opting for the deluxe package is ~37
# - Majority of the customers opting for the deluxe package are at manager designation
# - Majority of the customers opting for the deluxe package are married

# In[53]:


data[(data['ProductPitched']=='Super Deluxe') & (data['ProdTaken']==1)].describe(include='all').T


# - Average monthly income of customers opting for the super deluxe package is ~29,823.
# - Average age of for customers opting for the super deluxe package is ~43
# - Majority of the customers opting for the super deluxe package are at AVP designation
# - Majority of the customers opting for the super deluxe package are single

# In[54]:


data[(data['ProductPitched']=='King') & (data['ProdTaken']==1)].describe(include='all').T


# - Average monthly income of customers opting for the king package is ~34,672.
# - Average age of for customers opting for the king package is ~49
# - Majority of the customers opting for the king package are at VP designation
# - Majority of the customers opting for the king package are single

# **These profiles can act as a preliminary step to categorize customers for different packages and based on these profiles:**
# - new packages can be customized
# - identify the product to be pitched to the customer

# ### Correlation Heatmap

# In[55]:


sns.set(rc={'figure.figsize':(7,7)})
sns.heatmap(data.corr(),
            annot=True,
            linewidths=.5,
            center=0,
            cbar=False,
            cmap="YlGnBu",
            fmt='0.2f')
plt.show()


# - Number of trips and age have weak positive correlation, which makes sense as age increases number of trips is expected to increase.
# - Age and monthly income are positively correlated.
# - ProdTaken has weak negative correlation with age which agrees with our earlier observation that as age increases probability for purchasing a package decreases.  
# - No other variables have high correlation among them.

# ### Split the dataset into train and test sets

# In[56]:


#Separating target variable and other variables
X=data.drop(columns='ProdTaken')
Y=data['ProdTaken']


#  **As our aim is to predict customers who are more likely to buy the product, we should drop columns `DurationOfPitch','NumberOfFollowups','ProductPitched','PitchSatisfactionScore'` as these columns would not be available at the time of prediction for new data.**

# In[57]:


#Dropping columns
X.drop(columns=['DurationOfPitch','NumberOfFollowups','ProductPitched','PitchSatisfactionScore'],inplace=True)


# In[58]:


#Splitting the data into train and test sets
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.30,random_state=1,stratify=Y)


# **As we saw earlier, our data has missing values. We will impute missing values using median for continuous variables and mode for categorical variables. We will use `SimpleImputer` to do this.**
# 
# **The `SimpleImputer` provides basic strategies for imputing missing values. Missing values can be imputed with a provided constant value, or using the statistics (mean, median or most frequent) of each column in which the missing values are located.**

# In[59]:


si1=SimpleImputer(strategy='median')

median_imputed_col=['Age','MonthlyIncome','NumberOfTrips']

#Fit and transform the train data
X_train[median_imputed_col]=si1.fit_transform(X_train[median_imputed_col])

#Transform the test data i.e. replace missing values with the median calculated using training data
X_test[median_imputed_col]=si1.transform(X_test[median_imputed_col])


# In[60]:


si2=SimpleImputer(strategy='most_frequent')

mode_imputed_col=['PreferredLoginDevice','PreferredPropertyStar','NumberOfChildrenVisited']

#Fit and transform the train data
X_train[mode_imputed_col]=si2.fit_transform(X_train[mode_imputed_col])

#Transform the test data i.e. replace missing values with the mode calculated using training data
X_test[mode_imputed_col]=si2.transform(X_test[mode_imputed_col])


# In[61]:


#Checking that no column has missing values in train or test sets
print(X_train.isna().sum())
print('-'*30)
print(X_test.isna().sum())


# **Let's create dummy variables for string type variables and convert other column types back to float.**

# In[62]:


#converting data types of columns to float
for column in ['NumberOfPersonVisited', 'Passport', 'OwnCar']:
    X_train[column]=X_train[column].astype('float')
    X_test[column]=X_test[column].astype('float')


# In[63]:


#List of columns to create a dummy variables
col_dummy=['PreferredLoginDevice', 'Occupation', 'Gender', 'MaritalStatus', 'Designation', 'CityTier']


# In[64]:


#Encoding categorical varaibles
X_train=pd.get_dummies(X_train, columns=col_dummy, drop_first=True)
X_test=pd.get_dummies(X_test, columns=col_dummy, drop_first=True)


# # Building the model

# ### Model evaluation criterion:
# 
# #### Model can make wrong predictions as:
# 1. Predicting a customer will buy the product and the customer doesn't buy - Loss of resources
# 2. Predicting a customer will not buy the product and the customer buys - Loss of opportunity
# 
# #### Which case is more important? 
# * Predicting that customer will not buy the product but he buys i.e. losing on a potential source of income for the company because that customer will not targeted by the marketing team when he should be targeted.
# 
# #### How to reduce this loss i.e need to reduce False Negatives?
# * Company wants Recall to be maximized, greater the Recall lesser the chances of false negatives.

# **Let's create two functions to calculate different metrics and confusion matrix, so that we don't have to use the same code repeatedly for each model.**

# In[65]:


##  Function to calculate different metric scores of the model - Accuracy, Recall and Precision
def get_metrics_score(model,flag=True):
    '''
    model : classifier to predict values of X

    '''
    # defining an empty list to store train and test results
    score_list=[] 
    
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    train_acc = model.score(X_train,y_train)
    test_acc = model.score(X_test,y_test)
    
    train_recall = metrics.recall_score(y_train,pred_train)
    test_recall = metrics.recall_score(y_test,pred_test)
    
    train_precision = metrics.precision_score(y_train,pred_train)
    test_precision = metrics.precision_score(y_test,pred_test)
    
    score_list.extend((train_acc,test_acc,train_recall,test_recall,train_precision,test_precision))
        
    # If the flag is set to True then only the following print statements will be dispayed. The default value is set to True.
    if flag == True: 
        print("Accuracy on training set : ",model.score(X_train,y_train))
        print("Accuracy on test set : ",model.score(X_test,y_test))
        print("Recall on training set : ",metrics.recall_score(y_train,pred_train))
        print("Recall on test set : ",metrics.recall_score(y_test,pred_test))
        print("Precision on training set : ",metrics.precision_score(y_train,pred_train))
        print("Precision on test set : ",metrics.precision_score(y_test,pred_test))
    
    return score_list # returning the list with train and test scores


# In[66]:


## Function to create confusion matrix
def make_confusion_matrix(model,y_actual,labels=[1, 0]):
    '''
    model : classifier to predict values of X
    y_actual : ground truth  
    
    '''
    y_predict = model.predict(X_test)
    cm=metrics.confusion_matrix( y_actual, y_predict, labels=[0, 1])
    df_cm = pd.DataFrame(cm, index = [i for i in ["Actual - No","Actual - Yes"]],
                  columns = [i for i in ['Predicted - No','Predicted - Yes']])
    group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}" for v1, v2 in
              zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=labels,fmt='')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# ## Decision Tree Classifier

# In[67]:


#Fitting the model
d_tree = DecisionTreeClassifier(random_state=1)
d_tree.fit(X_train,y_train)

#Calculating different metrics
get_metrics_score(d_tree)

#Creating confusion matrix
make_confusion_matrix(d_tree,y_test)


# - The model is overfitting the training data as training recall/precision is much higher than the test recall/precision

# ### Cost Complexity Pruning

# **Let's try pruning the tree and see if the performance improves.**

# In[68]:


path = d_tree.cost_complexity_pruning_path(X_train, y_train)

ccp_alphas, impurities = path.ccp_alphas, path.impurities


# In[69]:


clfs_list = []

for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=1, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs_list.append(clf)

print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(clfs_list[-1].tree_.node_count, ccp_alphas[-1]))


# In[70]:


#Fitting model for each value of alpha and saving the train recall in a list 
recall_train=[]
for clf in clfs_list:
    pred_train=clf.predict(X_train)
    values_train=metrics.recall_score(y_train,pred_train)
    recall_train.append(values_train)


# In[71]:


#Fitting model for each value of alpha and saving the test recall in a list
recall_test=[]
for clf in clfs_list:
    pred_test=clf.predict(X_test)
    values_test=metrics.recall_score(y_test,pred_test)
    recall_test.append(values_test)


# In[72]:


#Plotting the graph for Recall VS alpha 
fig, ax = plt.subplots(figsize=(15,5))
ax.set_xlabel("alpha")
ax.set_ylabel("Recall")
ax.set_title("Recall vs alpha for training and testing sets")
ax.plot(ccp_alphas, recall_train, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, recall_test, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()


# In[73]:


#Creating the model where we get highest test recall
index_best_pruned_model = np.argmax(recall_test)

pruned_dtree_model = clfs_list[index_best_pruned_model]

get_metrics_score(pruned_dtree_model)

make_confusion_matrix(pruned_dtree_model,y_test)


# - We can see from the graph plotted above that maximum test recall is for the model without pruning i.e. alpha=0
# - There is no improvement in the model performance as the best pruned model fitted is the same as we built initially. 
# - Let's try hyperparameter tuning, with class weights to compensate for the imbalanced data, and see if the model performance improves.

# ### Hyperparameter Tuning

# In[74]:


#Choose the type of classifier. 
dtree_estimator = DecisionTreeClassifier(class_weight={0:0.18,1:0.72},random_state=1)

# Grid of parameters to choose from
parameters = {'max_depth': np.arange(2,30), 
              'min_samples_leaf': [1, 2, 5, 7, 10],
              'max_leaf_nodes' : [2, 3, 5, 10,15],
              'min_impurity_decrease': [0.0001,0.001,0.01,0.1]
             }

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.recall_score)

# Run the grid search
grid_obj = GridSearchCV(dtree_estimator, parameters, scoring=scorer,n_jobs=-1)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
dtree_estimator = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
dtree_estimator.fit(X_train, y_train)


# In[75]:


get_metrics_score(dtree_estimator)

make_confusion_matrix(dtree_estimator,y_test)


# - The model is generalizing well and not overfitting the data
# - The recall is still similar on the test data but the precision has decreased significantly.

# ## Random Forest Classifier

# In[76]:


#Fitting the model
rf_estimator = RandomForestClassifier(random_state=1)
rf_estimator.fit(X_train,y_train)

#Calculating different metrics
get_metrics_score(rf_estimator)

#Creating confusion matrix
make_confusion_matrix(rf_estimator,y_test)


# - With default parameters, random forest is performing better than decision tree in terms of precision but has less recall.
# - The model is overfitting the training data.
# - We'll try to reduce overfitting and improve recall by hyperparameter tuning.

# ### Hyperparameter Tuning

# In[77]:


# Choose the type of classifier. 
rf_tuned = RandomForestClassifier(class_weight={0:0.18,1:0.82},random_state=1,oob_score=True,bootstrap=True)

parameters = {  
                'max_depth': list(np.arange(5,30,5)) + [None],
                'max_features': ['sqrt','log2',None],
                'min_samples_leaf': np.arange(1,15,5),
                'min_samples_split': np.arange(2, 20, 5),
                'n_estimators': np.arange(10,110,10)}


# Run the grid search
grid_obj = GridSearchCV(rf_tuned, parameters, scoring='recall',cv=5,n_jobs=-1)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
rf_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
rf_tuned.fit(X_train, y_train)


# In[78]:


#Calculating different metrics
get_metrics_score(rf_tuned)

#Creating confusion matrix
make_confusion_matrix(rf_tuned,y_test)


# - The overfitting has reduced after tuning the model.
# - The recall has improved on the test data but the precision has decreased significantly.

# ## Bagging Classifier

# In[79]:


#Fitting the model
bagging_classifier = BaggingClassifier(random_state=1)
bagging_classifier.fit(X_train,y_train)

#Calculating different metrics
get_metrics_score(bagging_classifier)

#Creating confusion matrix
make_confusion_matrix(bagging_classifier,y_test)


# - With default parameters, bagging classifier is performing good in terms of precision but has less recall.
# - The model is overfitting the training data.
# - We'll try to reduce overfitting and improve recall by hyperparameter tuning.

# ### Hyperparameter Tuning

# In[80]:


# Choose the type of classifier. 
bagging_estimator_tuned = BaggingClassifier(random_state=1)

# Grid of parameters to choose from
parameters = {'max_samples': [0.7,0.8,0.9,1], 
              'max_features': [0.7,0.8,0.9,1],
              'n_estimators' : [10,20,30,40,50],
             }

# Type of scoring used to compare parameter combinations
acc_scorer = metrics.make_scorer(metrics.recall_score)

# Run the grid search
grid_obj = GridSearchCV(bagging_estimator_tuned, parameters, scoring=acc_scorer,cv=5)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
bagging_estimator_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data.
bagging_estimator_tuned.fit(X_train, y_train)


# In[81]:


#Calculating different metrics
get_metrics_score(bagging_estimator_tuned)

#Creating confusion matrix
make_confusion_matrix(bagging_estimator_tuned,y_test)


# - The test recall and test precision has improved but the model is still overfitting the training data.
# - The recall is still very low.

# ## AdaBoost Classifier

# In[82]:


#Fitting the model
ab_classifier = AdaBoostClassifier(random_state=1)
ab_classifier.fit(X_train,y_train)

#Calculating different metrics
get_metrics_score(ab_classifier)

#Creating confusion matrix
make_confusion_matrix(ab_classifier,y_test)


# - The model is not overfitting the data but is giving very low recall on training and test data.

# ### Hyperparameter Tuning

# In[83]:


# Choose the type of classifier. 
abc_tuned = AdaBoostClassifier(random_state=1)

# Grid of parameters to choose from
parameters = {
    #Let's try different max_depth for base_estimator
    "base_estimator":[DecisionTreeClassifier(max_depth=1),DecisionTreeClassifier(max_depth=2),
                      DecisionTreeClassifier(max_depth=3)],
    "n_estimators": np.arange(10,110,10),
    "learning_rate":np.arange(0.1,2,0.1)
}

# Type of scoring used to compare parameter  combinations
acc_scorer = metrics.make_scorer(metrics.recall_score)

# Run the grid search
grid_obj = GridSearchCV(abc_tuned, parameters, scoring=acc_scorer,cv=5)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
abc_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data.
abc_tuned.fit(X_train, y_train)


# In[84]:


#Calculating different metrics
get_metrics_score(abc_tuned)

#Creating confusion matrix
make_confusion_matrix(abc_tuned,y_test)


# - The train as well as test recall have improved significantly but the model is overfitting the training data now.

# ## Gradient Boosting Classifier

# In[85]:


#Fitting the model
gb_classifier = GradientBoostingClassifier(random_state=1)
gb_classifier.fit(X_train,y_train)

#Calculating different metrics
get_metrics_score(gb_classifier)

#Creating confusion matrix
make_confusion_matrix(gb_classifier,y_test)


# - The model is slightly overfitting the training data but is giving very low recall on training and test data.
# - The recall is better as compared to AdaBoost with default parameters but still not great.

# ### Hyperparameter Tuning

# In[86]:


# Choose the type of classifier. 
gbc_tuned = GradientBoostingClassifier(init=AdaBoostClassifier(random_state=1),random_state=1)

# Grid of parameters to choose from
parameters = {
    "n_estimators": [100,150,200,250],
    "subsample":[0.8,0.9,1],
    "max_features":[0.7,0.8,0.9,1]
}

# Type of scoring used to compare parameter combinations
acc_scorer = metrics.make_scorer(metrics.recall_score)

# Run the grid search
grid_obj = GridSearchCV(gbc_tuned, parameters, scoring=acc_scorer,cv=5)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
gbc_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data.
gbc_tuned.fit(X_train, y_train)


# In[87]:


#Calculating different metrics
get_metrics_score(gbc_tuned)

#Creating confusion matrix
make_confusion_matrix(gbc_tuned,y_test)


# - The model performance has improved slightly after hyperparameter tuning but the model is still overfitting the training data.
# - The test precision has decreased slightly and the test recall has increased slightly but still very low. 

# ## XGBoost Classifier

# In[88]:


#Fitting the model
xgb_classifier = XGBClassifier(random_state=1)
xgb_classifier.fit(X_train,y_train)

#Calculating different metrics
get_metrics_score(xgb_classifier)

#Creating confusion matrix
make_confusion_matrix(xgb_classifier,y_test)


# - With default parameters, the model is overfitting the training data.
# - The model is not able to correctly identify potential customers i.e. the test recall is very low.

# ### Hyperparameter Tuning

# In[89]:


# Choose the type of classifier. 
xgb_tuned = XGBClassifier(random_state=1)

# Grid of parameters to choose from
parameters = {
    "n_estimators": np.arange(10,100,20),
    "scale_pos_weight":[0,1,2,5],
    "subsample":[0.5,0.7,0.9,1],
    "learning_rate":[0.01,0.1,0.2,0.05],
    "gamma":[0,1,3],
    "colsample_bytree":[0.5,0.7,0.9,1],
    "colsample_bylevel":[0.5,0.7,0.9,1]
}

# Type of scoring used to compare parameter combinations
acc_scorer = metrics.make_scorer(metrics.recall_score)

# Run the grid search
grid_obj = GridSearchCV(xgb_tuned, parameters,scoring=acc_scorer,cv=5)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
xgb_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data.
xgb_tuned.fit(X_train, y_train)


# In[90]:


#Calculating different metrics
get_metrics_score(xgb_tuned)

#Creating confusion matrix
make_confusion_matrix(xgb_tuned,y_test)


# - The overfitting has reduced after hyperparameter tuning
# - Tuned xgboost model is giving the highest recall yet among all the model we built.
# - Let's try one more model - Stacking classifier.

# ## Stacking Classifier

# - Stacking classifier stacks the output of individual estimators and use a classifier to compute the final prediction
# - Stacking allows to use the strength of each individual estimator by using their output as input of a final estimator

# In[91]:


estimators = [('Random Forest',rf_tuned), ('Gradient Boosting',gbc_tuned), ('Decision Tree',dtree_estimator)]

final_estimator = xgb_tuned

stacking_classifier= StackingClassifier(estimators=estimators,final_estimator=final_estimator)

stacking_classifier.fit(X_train,y_train)


# In[92]:


#Calculating different metrics
get_metrics_score(stacking_classifier)

#Creating confusion matrix
make_confusion_matrix(stacking_classifier,y_test)


# - Stacking classifier has further increased the recall that we got from xgboost model but reduced the precision as well.
# - Model is overfitting the training data.

# ## Comparing all models

# In[93]:


# defining list of models
models = [d_tree, pruned_dtree_model, dtree_estimator,rf_estimator, rf_tuned, bagging_classifier,bagging_estimator_tuned,
          ab_classifier, abc_tuned, gb_classifier, gbc_tuned, xgb_classifier,xgb_tuned, stacking_classifier]

# defining empty lists to add train and test results
acc_train = []
acc_test = []
recall_train = []
recall_test = []
precision_train = []
precision_test = []

# looping through all the models to get the metrics score - Accuracy, Recall and Precision
for model in models:
    
    j = get_metrics_score(model,False)
    acc_train.append(j[0])
    acc_test.append(j[1])
    recall_train.append(j[2])
    recall_test.append(j[3])
    precision_train.append(j[4])
    precision_test.append(j[5])


# In[94]:


comparison_frame = pd.DataFrame({'Model':['Decision Tree','Decision Tree Pruned','Tuned Decision Tree','Random Forest','Tuned Random Forest',
                                          'Bagging Classifier','Bagging Classifier Tuned','AdaBoost Classifier','Tuned AdaBoost Classifier',
                                          'Gradient Boosting Classifier', 'Tuned Gradient Boosting Classifier',
                                          'XGBoost Classifier',  'Tuned XGBoost Classifier', 'Stacking Classifier'], 
                                          'Train_Accuracy': acc_train,'Test_Accuracy': acc_test,
                                          'Train_Recall':recall_train,'Test_Recall':recall_test,
                                          'Train_Precision':precision_train,'Test_Precision':precision_test}) 

#Sorting models in decreasing order of test recall
comparison_frame.sort_values(by='Test_Recall',ascending=False)


# In[95]:


feature_names = X_train.columns
importances = xgb_tuned.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12,12))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='violet', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# - Stacking classifier has no attribute of feature importance, so we have used xgboost model to calculate the feature importance.
# - Passport is the most important feature, followed by designation and city tier, as per the tuned xgboost model.

# # Business Recommendations

# - Our analysis shows that very few customers have passports and they are more likely to purchase the travel package. The company should customize more international packages to attract more such customers.
# - We have customers from tier 1 and tier 3 cities but very few form tier 2 cities. Company should expand its marketing strategies to increase the number of customers from tier 2 cities.
# - We saw in our analysis that people with higher income or at high positions like AVP or VP are less likely to buy the product. The company can offer short term travel packages and customize the package for higher income customers with added luxuries to target such customers.
# - When implementing a marketing strategy, external factors, such as the number of follow ups, time of calling, should also be carefully considered as our analysis shows that the customers who have been followed up more are the one's buying the package.
# - After we identify a potential customer, the company should pitch packages as per the customer's monthly income, for example, do not pitch king packages to a customer with low income and such packages can be pitched more to the higher income customers.
# - We saw in our analysis that young and single people are more likely to buy the offered packages. The company can offer discounts or customize the package to attract more couples, families, and customers above 30 years of age.
