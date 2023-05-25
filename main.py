import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns

class EDA:
    def __init__(self):
        self.dataset = self.explorationData()
        self.dataEncoding()
        self.dataScaled = self.dataScaling()
        self.drawCorrelation()
        self.Visualization()

    def explorationData(self):
        dataset = pd.read_csv('https://raw.githubusercontent.com/xx36Mostafa/m/main/car-data.csv')
        print('Dimensions ==> ',dataset.shape)
        print(dataset.columns)
        print(dataset.info())
        print(dataset.isnull().sum())
        print(dataset.head()) 
        print(dataset['AnnualSalary'].describe())
        print(dataset['Purchased'].value_counts())
        print('Duplicated rows:',dataset.duplicated().sum())
        print('The Dataset nunique ',dataset.nunique())
        print(f'The Dataset dTypes\n{dataset.dtypes}')
        print(round(dataset['Purchased'].value_counts()/dataset.shape[0]*100,2))
        print(round(dataset['Gender'].value_counts()/dataset.shape[0]*100,2))
        print(dataset['AnnualSalary'].max())
        return dataset 
    
    def dataEncoding(self): 
        dtype = self.dataset.dtypes
        for i in range(self.dataset.shape[1]):
            if dtype[i] == 'object':
                modleEncode = preprocessing.LabelEncoder()
                self.dataset[self.dataset.columns[i]] = modleEncode.fit_transform(self.dataset[self.dataset.columns[i]])

    def dataScaling(self):
        scalerModel = preprocessing.MinMaxScaler()
        ScaledData = scalerModel.fit_transform(self.dataset.values)
        ScaledData = pd.DataFrame(ScaledData,columns=self.dataset.columns)
        return ScaledData

    def drawCorrelation(self):
        r = self.dataScaled.corr()
        print(r)
        sns.heatmap(r, annot=True)
        plt.show()

    def Visualization(self):
        sns.catplot(data=self.dataset ,x='Age',hue='Purchased',kind='count')
        plt.show()
        
        x = self.dataset['AnnualSalary']
        y = self.dataset['Purchased']

        plt.scatter(x, y)
        plt.xlabel('AnnualSalary')
        plt.ylabel('Purchased')

        plt.figure(figsize=(4,4))
        sns.countplot(data=self.dataset , x='Purchased', palette = "YlOrBr_r")
        plt.show()

        sns.heatmap(self.dataset.isnull(),cmap='viridis',cbar=False,yticklabels=False)
        plt.title('missing data')
        plt.show()

        sns.distplot(self.dataset['AnnualSalary'])
        plt.show()

        round(self.dataset["Gender"].value_counts()/self.dataset.shape[0]*100,2).plot.pie(autopct= '%1.1f%%')
        plt.show()
        
        
class Train(EDA):
    def __init__(self):
        super().__init__()
        self.x = self.dataScaled.iloc[:,:-1].values
        self.y = self.dataset.iloc[:,-1].values

    def getData(self,SizeTrain):
        from sklearn.model_selection import train_test_split
        xtrain,xtest,ytrain,ytest = train_test_split(self.x,self.y,test_size=SizeTrain, random_state=42)
        return xtrain,xtest,ytrain,ytest
    
    def trainModel(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import confusion_matrix, recall_score, accuracy_score , precision_score

        def accuracyCurve():
            from sklearn.model_selection import train_test_split
            train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]
            accuracies = []
            for size in train_sizes:
                xtrain, xtest, ytrain, ytest = train_test_split(
                    self.x, self.y, train_size=size, random_state=42)
                RegressionModel = LogisticRegression()
                RegressionModel.fit(xtrain, ytrain)
                ypred = RegressionModel.predict(xtest)
                accu = accuracy_score(ytest, ypred)
                accuracies.append(accu)
            plt.plot(train_sizes, accuracies, marker='o')
            plt.xlabel('Training Set Size')
            plt.ylabel('Accuracy')
            plt.title('Accuracy Curve For Logistic Regression')
            plt.show()
            max_accuracy = max(accuracies)
            max_index = accuracies.index(max_accuracy)
            corresponding_train_size = train_sizes[max_index]
            return corresponding_train_size 
        SizeTrain = accuracyCurve()
 
        xtrain, xtest, ytrain, ytest = self.getData(SizeTrain)
        RegressionModel = LogisticRegression()
        RegressionModel.fit(xtrain, ytrain)

        ypred = RegressionModel.predict(xtest)
        AB = confusion_matrix(ytest, ypred)
        print('Confusion Matrix:\n', AB)
        sns.heatmap(AB,annot=True)
        plt.title("Correlation Heatmap of Confusion Matrix for Logistic Regression")
        plt.show()
        recall = recall_score(ytest, ypred)  
        print('Recall for LogisticRegression is = ', recall)

        p = precision_score(y_true=ytest,y_pred=ypred)
        print("Precision Score For LogisticRegression = ",p)

        accu2 = accuracy_score(ytest, ypred)
        print('Accuracy For LogisticRegression is =', accu2)
        
    def svmModel(self):
        from sklearn.svm import SVC
        from sklearn.metrics import confusion_matrix, recall_score, accuracy_score , precision_score
        def accuracyCurve():
            from sklearn.model_selection import train_test_split
            train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]
            accuracies = []
            for size in train_sizes:
                xtrain, xtest, ytrain, ytest = train_test_split(
                    self.x, self.y, train_size=size, random_state=42)
                model = SVC(kernel='poly',degree=4)
                model.fit(xtrain,ytrain)
                ypred = model.predict(xtest)
                accu2 = accuracy_score(ytest, ypred)
                accuracies.append(accu2)
            plt.plot(train_sizes, accuracies, marker='o')
            plt.xlabel('Training Set Size')
            plt.ylabel('Accuracy')
            plt.title('Accuracy Curve For SVM')
            plt.show()
            max_accuracy = max(accuracies)
            max_index = accuracies.index(max_accuracy)
            corresponding_train_size = train_sizes[max_index]
            return corresponding_train_size 
        
        SizeTrain = accuracyCurve()
        xtrain, xtest, ytrain, ytest = self.getData(SizeTrain)
        model = SVC(kernel='poly',degree=4)

        model.fit(xtrain,ytrain)
        ypred = model.predict(xtest)

        AB = confusion_matrix(ytest, ypred)
        print('Confusion Matrix:\n', AB)
        
        sns.heatmap(AB,annot=True)
        plt.title("Correlation Heatmap of Confusion Matrix for SVM Regression")
        plt.show()
        
        p = precision_score(y_true=ytest,y_pred=ypred)
        print("Precision Score For SVM = ",p)
        
        recall = recall_score(ytest, ypred)  
        print('Recall for log is=', recall)

        accu2 = accuracy_score(ytest, ypred)
        print('Accuracy For SVC is =', accu2)

model = Train() 
model.trainModel()
model.svmModel()