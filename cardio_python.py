"""Description:
 This program classifies a person as having a cardiovascular disease (1) or not (0)
 So the target class "cardio" equals 1, when the patient has cardiovascular desease, and it's 0, 
 when the patient is healthy."""


#Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Store the data into the df variable
df = pd.read_csv('cardio.csv') 
print(df.head(7)) #Print the first 7 rows



#Get the shape of the data (the number of rows & columns)
print(df.shape)

#Count the empty (NaN, NAN, na) values in each column
print(df.isna().sum())

#Another check for any null / missing values
print(df.isnull().values.any())

#View some basic statistical details like percentile, mean, standard deviation etc.
print(df.describe())

#Get a count of the number of patients with (1) and without (0) a cardiovasculer disease 
df['cardio'].value_counts()

#Visualize this count 
sns.countplot(df['cardio'])
plt.show()


#Create a years column
df['years'] = ( df['age'] / 365).round(0)   #Get the years by dividing the age in days by 365
print(df['years'])
df["years"] = pd.to_numeric(df["years"],downcast='integer') # Convert years to an integer
print(df["years"])
#Visualize the data
#colorblind palette for colorblindness
sns.countplot(x='years', hue='cardio', data = df, palette="colorblind", edgecolor=sns.color_palette("dark", n_colors = 1));
plt.show()

#Get the correlation of the columns
print(df.corr())


plt.figure(figsize=(7,7))  #7in by 7in
sns.heatmap(df.corr(), annot=True, fmt='.0%')
plt.show()


#Remove / drop the years column
df = df.drop('years', axis=1)

#splitting the data set into a feature data set also known as the independent data set (X), 
# and a target data set also known as the dependent data set (Y).
X = df.iloc[:, :-1].values # Get all of the columns except the last 
Y = df.iloc[:, -1].values  # Get only the last column


#Split the data again, but this time into 75% training and 25% testing data sets.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 1)


#Feature Scaling
#Scale the data to be values between 0 and 1 inclusive
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)





#Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 1)
forest.fit(X_train, Y_train)

#Test the models accuracy on the training data
model = forest
model.score(X_train, Y_train)

#Test the models accuracy on the test data set
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, model.predict(X_test))
  
TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]
  
print(cm)
print('Model Test Accuracy = "{}!"'.format(  (TP + TN) / (TP + TN + FN + FP)))






































