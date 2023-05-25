# CarsPurchaseDecision
Dataset:
Cars - Purchase Decision Dataset
Link for dataset:
https://www.kaggle.com/datasets/gabrielsantello/cars-purchase-decision-dataset
**Exploratory Data Analysis**:
In first we need to do Exploration For Data or so called eda,
We use the some library’s in python like { Pandas , sklearn , matplotlib , numpy , seaborn }. In first we need to read data do this with pandas pd.read_csv(‘path’)
And use some functions to do the eda like 
‘ Dataset.shape ‘ use this to get the dimensions for dataset to know what the length for data
‘ Dataset.columns ‘  to know the columns name
Use the dataset.info() function to get the information for dataset like columns names number of columns check if he have none value’s or no and get the count of none values
Use the isnull().sum() to know the summation of None data in columns and return the columns names
‘Dataset.head()‘ to know the first 5 columns
Use the dataset[‘annualSalary’].describe() function to get the to generate descriptive statistics for annualSalar column
Use the dataset[‘Purchased].values_count () function to obtain the frequency count of unique values for Purchased column.
Use the duplicated().sum() function to calculate the number of duplicated values in dataset
Use the dtypes function to know the types of type of each columns 
Use the dataset['AnnualSalary'].max() to know the max value for annual salary 
Finally in ExplorationData we use the 
round(dataset['Purchased'].value_counts()/dataset.shape[0]*100,2)
round(dataset['Gender'].value_counts()/dataset.shape[0]*100,2)
To Calculate percentage distribution of 'Gender'  and Calculate percentage distribution of 'Purchased'
 
When we used the dtypes we find that there is object in the data and we have to do the encoding process
we do this function to do encoding for dataset 
 
We get the preprocessing from sklearn and do loop in dataset columns num  and check if type of column is object we will do encoding to convert the data to 0 or 1 
And we create the function to do scaling for data because we need to convert data between 0 to 1 to know the correlation of dataset
 
This function do scaling for dataset and get the columns name of dataset and create new dataset with scaling data to know the correlation between value’s and get the most value’s and do visualization
And create the correlation function to Get the correlation in dataset and show the correlation in heatmap 
Used the corr() function 
To get the correlation values and import the seaborn as sns to show the correlation in heatmap by the heatmap()
Function and use the maplotlib to show the heatmap
We find the highest correlation is 
1- Age  Purchased 
2- Annal Salary and Purchased  
3- Annal salary and Age
