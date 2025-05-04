import pandas as pd
data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}
#load data into a DataFrame object:
df = pd.DataFrame(data)
print(df) 



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(file_name, header=0)
df = pd.read_csv("people.csv", index_col="First Name")
pd.read_csv("abc.csv", usecols= "abc")

df.to_csv("abc.csv", index=False, columns= ['Name', 'Age'])

# Load the CSV file using pandas with multiple delimiters
df = pd.read_csv('sample.csv',
                 sep='[:, |_]',  # Define the delimiters
                 engine='python')  # Use Python engine for regex separators

# change the date into datetime64 data type
df = pd.read_csv("people.csv", parse_dates=["Date of birth"])
print(df.info())

print(df.info())
print(df.describe())
print(df.dtypes)
print(df.head())
print(df[['Name', 'Age']])  


print(df.columns.tolist())
print(df[['Hours_Studied', 'Exam_Score']].head())



# Apply function
def fun(value):
    if value > 70:
        return "Yes"
    else:
        return "No"
df['Customer Satisfaction'] = df['Spending Score (1-100)'].apply(fun)


# how to assign headers
headers = ['ID', 'Name', 'Score']
df.columns = headers

# counting
xyz = df['columnname_xyz'].value_counts()

# filtering a column
filtered_df = df.loc[lambda x: x['Age'] > 25].sort_values('Age')
df.loc[df['Age'] > 25].sort_values('Age')   # the same as above code

# or can also be 
filtered_df = df[df['Age'] > 25]

# ensures that names are sorted alphabetically without considering case differences.
sorted_df = df.sort_values(by='Name', key=lambda col: col.str.lower())

# sorting data
sorted_df = df.sort_values(by='Age')
sorted_df = df.sort_values(by="Age", na_position="first")

# multiple columns
first = df[["Age", "College", "Salary"]]
print(first.head())

# DataFrame.loc[]: Label-based indexing for selecting data by row/column labels.
# DataFrame.iloc[]: Position-based indexing for selecting data by 
#                   row/column integer positions.
    
print(df.loc[["row1", "row2"], ["column1", "column2", "column3"]])
df.loc[:, ["column1", "column2", "column3"]]    # for all rows, specific column
print(df.loc[df['sample_col1'] == 1])

# Select a single row by position
rowX = df.iloc[[3, 5, 7]]
print(rowX)

print(row_500)
print(df_forecast['Germany DA EUR/MWh'].iloc[405:500].isna().sum())
print(df_forecast[features].iloc[405:500].isna().sum())


# Count NaN in each column:
df.isna().sum()
df['your_column_name'].isna().sum()


# first NaN index on specific column
first_nan_index = df_forecast['rolling_mean_96'].isna().idxmax()
print("First NaN in 'rolling_mean_96' is at row index:", first_nan_index)
print(df_forecast[495:510].isna().any(axis=1))



# Write a code that identifies which columns have missing data.
# Looping: Counts how many Trues and Falses are in that column (i.e., how many missing or not missing values).
# isnull() returns a DataFrame of Boolean values, where True represents missing data (NaN).
# looping each column, to show null/not null in each column
missing_data = df.isnull()
print(missing_data.head())
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")  

# Filtering Data Based on Missing Values
# showing all the data which has NaN on Gender column
df = pd.read_csv("employees.csv")
bool_series = pd.isnull(df["Gender"])
missing_gender_data = df[bool_series]
print(missing_gender_data)


# replace the missing values of weight with the average value of the attribute.
avg_weight = df['Weight_kg'].astype('float').mean(axis=0)
df["Weight_kg"].replace(np.nan, avg_weight, inplace=True)

df.replace(to_replace=np.nan, value=-99)

# astype() function converts the values to the desired data type
# axis=0 indicates that the mean value is to calculated across all column elements in a row.

# replace with most frequent data 
# value_counts will count the freq of each value, the .idxmax will choose the index 
    # where the highest freq value
common_screen_size = df['Screen_Size_cm'].value_counts().idxmax()
df["Screen_Size_cm"].replace(np.nan, common_screen_size, inplace=True)

# replace with the most frequent value
# .mode is to get the most frequent value
most_frequent = df['weight'].mode()[0]
# 2️⃣ Replace missing values with that mode
df['weight'].fillna(most_frequent, inplace=True)



# dropping na NaN value in some columns
df = df.dropna(subset=['highway-mpg', 'price'])
#  drop rows where all values are missing using dropna(how=’all’).
df.dropna(how='all')
# drop NaN column
df.dropna(how='all', axis=1, inplace=True)
# Drop rows with at least one missing value
df.dropna()
# Drop columns with at least one missing value
df.dropna(axis=1)
# Drop rows with any missing value
nd = df.dropna(axis=0, how='any')

# Drop duplicates based on the 'Name' column
result = df.drop_duplicates(subset=['Name'])
# Drop all duplicates, keeps only rows that are entirely unique.
result = df.drop_duplicates(keep=False, inplace=True) 



# printing missing value
print("Missing values in x:", x.isnull().sum())
print("Missing values in y:", y.isnull().sum())



# Convert 'Age' column to float type
df['Age'] = df['Age'].astype(float)
# change data type (to float e.g.)
df[["Weight_kg","Screen_Size_cm"]] = df[["Weight_kg","Screen_Size_cm"]].astype("float")
# Convert 'Join Date' to datetime type
df['Time'] = pd.to_datetime(df['Time'])
# Convert 'Age' to float and 'Salary' to string
df = df.astype({'Age': 'float64', 'Salary': 'str'})

# Converts all uppercase to lowercase
print(df['Names'].str.lower())
print(df['Names'].str.upper())
print(df['Names'].str.strip()) # cleaning spaces before and after string

print(df['Names'].str.len()) #length of the string
print(df['City'].str.get_dummies()) # One-Hot Encoded values like we can see that it returns boolean value 1 if it exists in relative index or 0 if not exists.
print(df['Names'].str.startswith('G'))

# dropping erasing certain row
df_ddrop = df.drop(4).head()


# Data standardization: convert weight from kg to pounds and rename the column
df["Weight_kg"] = df["Weight_kg"]*2.205
df.rename(columns={'Weight_kg':'Weight_pounds'}, inplace=True)
# Data standardization: convert screen size from cm to inch
df["Screen_Size_cm"] = df["Screen_Size_cm"]/2.54
df.rename(columns={'Screen_Size_cm':'Screen_Size_inch'}, inplace=True)
# data normalization using 'simple feature'
df['CPU_frequency'] = df['CPU_frequency']/df['CPU_frequency'].abs().max() 

# min max normalization
for column in df.columns: 
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
  

.shape
.info()
.corr()
.describe()

# binning / categorizing using linspace
bins = np.linspace(min(df["Price"]), max(df["Price"]), 4)
group_names = ['Low', 'Medium', 'High']
df['Price-binned'] = pd.cut(df['Price'], bins, labels=group_names, include_lowest=True )

# plot graph plt using counting method, low, medium, high etc
plt.bar(group_names, df["Price-binned"].value_counts())
plt.xlabel("Price")
plt.ylabel("count")
plt.title("Price bins")




# creating dummy, kind of categorizing 1 thing into two things 
dummy_variable_1 = pd.get_dummies(df["Screen"])
dummy_variable_1.rename(columns={'IPS Panel':'Screen-IPS_panel', 'Full HD':'Screen-Full_HD'}, inplace=True)
df = pd.concat([df, dummy_variable_1], axis=1)  # merge columns

# creating dummy for binary data (eg male and female)
df = pd.DataFrame({'Gender': ['Male', 'Female', 'Male', 'Female']})
df['Gender_binary'] = df['Gender'].map({'Male': 1, 'Female': 0})

print(df)


XYZ = pd.concat([df1, df2], axis=1)
XYZ = pd.concat([df, df1], axis=1, join='inner') #irisan

res3 = pd.concat([df, df1], axis=1, join_axes=[df.index]) # only combine data with same index
res = pd.concat([df, df1], ignore_index=True) # combining all data

res1 = pd.merge(df, df1, on=['key', 'key1']) # merging based on key and key1, both have to be the same on both 



# drop original column "Screen" from "df"
df.drop("Screen", axis = 1, inplace=True)

# Drop columns you don't need
df_filtered = df.drop(columns=['Unwanted_Column1', 'Unwanted_Column2'])

df_numeric = df.select_dtypes(include='number') #include only numerical value



# Creating a Series from a List/dict
ser = pd.Series(data_list)

# using range to create series from 5 to 15
ser = pd.Series(range(5, 15))

# groupby
grouping = df.groupby(['Team', 'Position'])

aggregated_data = df.groupby(['Team', 'Position']).agg(
    total_salary=('Salary', 'sum'),
    avg_salary=('Salary', 'mean'),
    player_count=('Name', 'count'))

filtered_df = df.groupby('Team').filter(lambda x: x['Salary'].mean() >= 1000000)

df.agg(['sum', 'min', 'max'])


# grouping results
df_gptest = df[['drive-wheels','body-style','price']]

df_grouped = df.groupby(['drive-wheels'], as_index=False).agg({'price': 'mean'})
grouped_df = df.groupby('Category')['Sales'].sum()
grouped_df = df.groupby(['Category', 'Region'])['Sales'].sum()

grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')


# correlation
numeric_df = df.select_dtypes(include=['float64', 'int64'])
xxy = numeric_df.corr()
print(xxy)

# creating correlation
# for param in ["CPU_frequency", "Screen_Size_inch","Weight_pounds"]:
#     print(f"Correlation of Price and {param} is ", df[[param,"Price"]].corr())

# Create a regression plot
sns.regplot(x="stroke", y="price", data=df)
# Show the plot
plt.show()


# pivot


# PLOTTING
import matplotlib.pyplot as plt
import seaborn as sns

x = df['quantity']
y = df['price']
plt.scatter(x,y)
plt.plot(x,y)
plt.hist(x,4)
plt.pcolor(df_pivot, cmap='RdBu') #df_pivot is the data name

sns.boxplot(x="Category", y="Price", data=df)

plt.pcolor(grouped_pivot, cmap='RdBu')


# plotting detail scatter
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, df['Predicted_Score'], color='red', label='Regression Line')
plt.title("Hours Studied vs Exam Score")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.legend()
plt.grid(True)
plt.show()



# distribution plot
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
sns.histplot(df['Exam_Score'], kde=True, bins=10, color='skyblue')
plt.title("📊 Distribution of Exam Scores")
plt.xlabel("Exam Score")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# distribution plot example 2 with KDE plot
plt.figure(figsize=(8, 4))
sns.kdeplot(df['Hours_Studied'], fill=True, color="purple")
plt.title("📈 KDE Plot: Hours Studied")
plt.xlabel("Hours Studied")
plt.show()

# example 3 KDE plot histogram for several columns
selected_cols = ['Hours_Studied', 'Exam_Score']

for col in selected_cols:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(df[col], fill=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.show()



# KDE Plot for Each Column
df_filtered = df.drop(columns=['Unwanted_Column1', 'Unwanted_Column2'])

for col in df_filtered.columns:  # Loop through each column in the filtered DataFrame
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=df_filtered, x=col, fill=True)
    plt.title(f"KDE Plot of {col}")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()


# Want All KDEs on One Plot?
plt.figure(figsize=(8, 5))
for col in df_filtered.columns:
    sns.kdeplot(df_filtered[col], label=col, fill=False)
    
plt.title("Combined KDE Plot")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()
    




# regression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample DataFrame with stroke and price
df = pd.DataFrame({
    'stroke': [3.2, 3.5, 3.0, 3.6, 3.1, 3.4, 3.3, 3.7],
    'price': [15000, 18000, 12000, 20000, 14000, 17500, 16000, 21000]
})

# Create a regression plot
sns.regplot(x="stroke", y="price", data=df)

# # Show the plot (No arguments needed)
plt.show()




a = ["geeks", "for", "geeks"]
#Looping through the list using enumerate
# starting the index from 1
for index, x in enumerate(a, start=1):
    print(index, x)

# output:
# 1 geeks
# 2 for
# 3 geeks
































# predicting data using linear regression train test
import numpy as npy
import pandas as pds
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
# Generating Sample Data
npy.random.seed(42)
x = npy.random.rand(100) * 10
y = 3 * x + npy.random.normal(0, 3, 100)  # Linear relation with noise
data = pds.DataFrame({'X': x, 'Y': y})
# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(data[['X']], data['Y'], test_size=0.2, random_state=42)
# Training a Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Plotting KDE for Observed vs. Predicted Values
plt.figure(figsize=(8, 5))
sns.kdeplot(y_test, label='Actual', fill=True, color='blue')
sns.kdeplot(y_pred, label='Predicted', fill=True, color='red')
plt.xlabel('Target Variable')
plt.ylabel('Density')
plt.title('KDE Plot of Actual vs. Predicted Values')
plt.legend()
plt.show()





import pandas as pd

# Load CSV or Excel
df = pd.read_csv('file.csv')
df = pd.read_excel('file.xlsx')

# Quick look
df.head()             # First 5 rows
df.tail()             # Last 5 rows
df.info()             # Summary: column types, non-null counts
df.describe()         # Statistics for numeric columns
df.shape              # (rows, columns)
df.columns            # List of column names
df.dtypes             # Data types


df.isna().sum()                   # Count NaNs per column
df['col'].isna().sum()           # Count NaNs in a specific column
df.notna().sum()                 # Count non-NaN values

df.dropna()                      # Drop all rows with any NaNs
df.dropna(subset=['col1'])      # Drop rows where 'col1' is NaN
df.fillna(0)                     # Replace all NaNs with 0
df['col'].fillna(df['col'].mean())  # Fill NaN with column mean

df.rename(columns={'old': 'new'}, inplace=True)  # Rename columns
df.drop('col', axis=1, inplace=True)             # Drop column
df.drop([0, 1], axis=0)                          # Drop rows by index

df[df['col'] > 10]                   # Rows where col > 10
df[df['col'].isna()]                # Rows where col is NaN
df[(df['a'] > 5) & (df['b'] < 10)]  # Combine conditions

df['new'] = df['col1'] + df['col2']     # Add new column
df['log_col'] = df['col'].apply(np.log) # Apply function
df['category'] = df['val'].map({0: 'low', 1: 'high'})  # Mapping

df['col'].value_counts()         # Frequency count
df.groupby('group_col').mean()   # Mean by group
df.groupby(['a', 'b']).agg(['mean', 'sum'])  # Multi agg
df.pivot_table(index='a', columns='b', values='c', aggfunc='sum')   # pivot table
        table = pd.pivot_table(df, values=['D', 'E'], index=['A', 'C'],
                       aggfunc={'D': "mean", 'E': "mean"})


df.sort_values('col', ascending=False)     # Sort by column
df.reset_index(drop=True, inplace=True)    # Reset index after sorting

df.replace({'old_val': 'new_val'})       # Replace specific values
df['col'].replace([1, 2], [10, 20])       # Replace multiple

df['date'] = pd.to_datetime(df['date'])  # Convert to datetime
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['weekday'] = df['date'].dt.day_name()

pd.concat([df1, df2])                     # Stack vertically
df1.merge(df2, on='key', how='inner')     # Join like SQL

df['col'].apply(lambda x: x ** 2)           # Row-wise
df.apply(np.sum, axis=1)                    # Row-wise sum, summing each row on column A and B 
df.apply(np.mean, axis=0)                   # Column-wise mean

df.to_csv('output.csv', index=False)
df.to_excel('output.xlsx', index=False)


df.duplicated().sum()           # Count duplicate rows
df.drop_duplicates(inplace=True)

df['col'].unique()              # Unique values
df['col'].nunique()             # Number of unique values

df.sample(5)                    # Random sample of 5 rows