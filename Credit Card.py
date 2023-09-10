"""
This file is for developing model using different algorithms to do Fault Analysis of Credit Card Data
"""

import numpy as np                                        # Numpy to carry out mathematical calculations
import pandas as pd                                       # Pandas to create and manipulate dataframe
#import matplotlib.pyplot as plt                           # Matplotlib to plot the data points onto a graph
#import seaborn as sns                                     # Seaborn to carry out statistical graphical functions
#import sklearn
import warnings
warnings.filterwarnings('ignore')
import scipy.stats as stats

# Loading Dataset

class CCFA:
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(self.path)


    def preprossing(self):
        try:
            print(f'{self.df.head()}')
            # print(self.df.isnull().sum())  # find null values
            # print(f'No of rows and no of col in Dataset: {self.df.shape[0], self.df.shape[1]}')
            # converting cat data to numerical data using Map method

        except Exception as E:
            print(f'Error in main - {E.__str__()}')

    def HNV(self):
        try:
            # Deleting 2 rows null values
            # Checking index of 2 rows having null vales
            print(self.df[self.df['NPA Status'].isnull()].index)
            print(self.df[self.df['age'].isnull()].index)
            # As we can see null values are present at same index of different columns so deleting null values in that index
            self.df = self.df.drop([150000, 150001], axis=0)  # axis=0 because we are removing rows, if col. then axis=1
            print(self.df.isnull().sum())
            # print(f'{self.df.head()}')
            """ As we can see MonthlyIncome and MonthlyIncome.1 column have same no. of null values so we can check 
            if values are also same. If values are same, then we can delete one column as it's duplicate column """
            print(self.df['MonthlyIncome'].head())
            print(self.df['MonthlyIncome.1'].head())
            """We can see that values are same in both the columns but we can't conclude by seeing 5-10 values 
            so to have clear picture we can check for standard deviation,if both are same then it's duplicate 
            so 1 col can be removed if not then no need to removed"""
            print(self.df['MonthlyIncome'].std())
            print(self.df['MonthlyIncome.1'].std())
            # Since, both col. have same std we can remove one column
            self.df = self.df.drop(['MonthlyIncome.1'], axis=1)  # axis=1 as removing col
            print(self.df.isnull().sum())

            # Handling null values for NumberOfDependents col.
            """ NumberOfDependents column is filled with numeric values but in datatype it's showing as object(Cat data)
            So, we need to convert it from object to int using pandas"""
            print(self.df['NumberOfDependents'].head())
            self.df['NumberOfDependents'] = pd.to_numeric(self.df['NumberOfDependents'])
            print(self.df['NumberOfDependents'].head())
            # Handling null Values using different techniques
            mean = self.df['NumberOfDependents'].mean()
            median = self.df['NumberOfDependents'].median()
            mode = self.df['NumberOfDependents'].mode()[0]
            random = self.df['NumberOfDependents'].sample()
            print('Mean of NumberOfDependents =', mean)
            print('Median of NumberOfDependents =', median)
            print('Mode of NumberOfDependents =', mode)
            print('Random Sample of NumberOfDependents =', random)

            # Replace Mean value in null value
            def mean_replacement(df, var, value):
                self.df[var + '_mean_replaced'] = self.df[var].fillna(value)
            mean_replacement(self.df, 'NumberOfDependents', mean)

            # Replace Median value in null value
            def median_replacement(df, var, value):
                self.df[var + '_median_replaced'] = self.df[var].fillna(value)
            median_replacement(self.df, 'NumberOfDependents', median)

            # Replace Mode value in null value
            def mode_replacement(df, var, value):
                self.df[var + '_mode_replaced'] = self.df[var].fillna(value)
            mode_replacement(self.df, 'NumberOfDependents', mode)

            # Replace Random Sample value in null values
            def RSI(df, var, value):
                self.df[var + '_random_replaced'] = self.df[var]
                S = self.df[var].dropna().sample(self.df[var].isnull().sum(), random_state=11)
                S.index = self.df[self.df[var].isnull()].index
                self.df.loc[self.df[var].isnull(), var + '_random_replaced'] = S
            RSI(self.df, 'NumberOfDependents', random)

            print(self.df.columns)

            # Finding std of replaced values and comparing with original values
            print('std of original variable : ', self.df['NumberOfDependents'].std())
            print('std of Mean_replaced variable : ', self.df['NumberOfDependents_mean_replaced'].std())
            print('std of Median_replaced variable : ', self.df['NumberOfDependents_median_replaced'].std())
            print('std of Mode_replaced variable : ', self.df['NumberOfDependents_mode_replaced'].std())
            print('std of Random_replaced variable : ', self.df['NumberOfDependents_random_replaced'].std())
            # differences
            print(1.1150860714871407 - 1.1004039906517658)  # mean
            print(1.1150860714871407 - 1.1070214146370303)  # median
            print(1.1150860714871407 - 1.1070214146370303)  # mode
            print(1.1150860714871407 - 1.114818486432032)   # Random Sample
            # After finding difference we can see how much close values are to original values, random sample is closer
            # So, we can take random sample col. and delete other columns i.e, original, mean, median and mode col.
            # Deleting columns which aren't required
            self.df = self.df.drop(
                ['NumberOfDependents', 'NumberOfDependents_mean_replaced', 'NumberOfDependents_median_replaced',
                 'NumberOfDependents_mode_replaced'], axis=1)
            print(self.df.columns)


            # Similarly, Handling null values for MonthlyIncome col.
            mean1 = self.df['MonthlyIncome'].mean()
            median1 = self.df['MonthlyIncome'].median()
            mode1 = self.df['MonthlyIncome'].mode()[0]
            random1 = self.df['MonthlyIncome'].sample()
            print('Mean of MonthlyIncome =', mean1)
            print('Median of MonthlyIncome =', median1)
            print('Mode of MonthlyIncome =', mode1)
            print('Random of MonthlyIncome =', random1)

            # mean1, median1, mode1, random_sample1
            def mean_replacement1(df, var, value):
                self.df[var + '_mean_replaced'] = self.df[var].fillna(value)
            mean_replacement1(self.df, 'MonthlyIncome', mean1)
            def median_replacement1(df, var, value):
                self.df[var + '_median_replaced'] = self.df[var].fillna(value)
            median_replacement1(self.df, 'MonthlyIncome', median1)
            def mode_replacement1(df, var, value):
                self.df[var + '_mode_replaced'] = self.df[var].fillna(value)
            mode_replacement1(self.df, 'MonthlyIncome', mode1)
            def RSI1(df, var, value):
                self.df[var + '_random_replaced'] = self.df[var]
                S = self.df[var].dropna().sample(self.df[var].isnull().sum(), random_state=11)
                S.index = self.df[self.df[var].isnull()].index
                self.df.loc[self.df[var].isnull(), var + '_random_replaced'] = S
            RSI1(self.df, 'MonthlyIncome', random1)

            print(self.df.columns)
            # Finding std of replaced values and comparing with original values
            print('std of original variable : ', self.df['MonthlyIncome'].std())
            print('std of Mean_replaced variable : ', self.df['MonthlyIncome_mean_replaced'].std())
            print('std of Median_replaced variable : ', self.df['MonthlyIncome_median_replaced'].std())
            print('std of Mode_replaced variable : ', self.df['MonthlyIncome_mode_replaced'].std())
            print('std of MonthlyIncome_random_replaced : ', self.df['MonthlyIncome_random_replaced'].std())
            # differences
            print(14384.674215282135 - 12880.445756228106)  # mean
            print(14384.674215282135 - 12890.395542154734)  # median
            print(14384.674215282135 - 12897.643871988601)  # mode
            print(14384.674215282135 - 15421.181597040963)  # Random Sample
            """After finding difference we can see how much close values are to original values, random sample is closer 
            So, we can take random sample col. and delete other columns i.e, original, mean, median and mode col."""
            # Deleting columns which aren't required
            self.df = self.df.drop(['MonthlyIncome', 'MonthlyIncome_mean_replaced', 'MonthlyIncome_median_replaced',
                          'MonthlyIncome_mode_replaced'], axis=1)
            print(self.df.columns)

        except Exception as E:
            print(f'Error in main - {E.__str__()}')


                                            # Variable Transformation
    def VT(self):
        try:
            self.df_num = self.df.select_dtypes(exclude='object')
            print(self.df_num.columns)

            for i in self.df_num.columns:
                self.df_num[i + '_yoe'], alpha = stats.yeojohnson(self.df_num[i])
            print(self.df_num.columns)

            b = ['NPA Status_yoe', 'RevolvingUtilizationOfUnsecuredLines_yoe', 'age_yoe',
                 'NumberOfTime30-59DaysPastDueNotWorse_yoe', 'DebtRatio_yoe',
                 'NumberOfOpenCreditLinesAndLoans_yoe', 'NumberOfTimes90DaysLate_yoe',
                 'NumberRealEstateLoansOrLines_yoe',
                 'NumberOfTime60-89DaysPastDueNotWorse_yoe',
                 'NumberOfDependents_random_replaced_yoe',
                 'MonthlyIncome_random_replaced_yoe']
            print(b[0], b[3], b[6], b[8])  # deleting features which don't have proper data
            self.df_num = self.df_num.drop(['NPA Status_yoe', 'NumberOfTime30-59DaysPastDueNotWorse_yoe',
                             'NumberOfTimes90DaysLate_yoe', 'NumberOfTime60-89DaysPastDueNotWorse_yoe'], axis=1)
            print(self.df_num.columns)
            print(len(self.df_num.columns))
        except Exception as E:
            print(f'Error in main - {E.__str__()}')


    def HO(self):
        try:
            self.df_num1 = ['RevolvingUtilizationOfUnsecuredLines_yoe', 'age_yoe', 'DebtRatio_yoe',
                       'NumberOfOpenCreditLinesAndLoans_yoe','NumberRealEstateLoansOrLines_yoe',
                 'NumberOfDependents_random_replaced_yoe','MonthlyIncome_random_replaced_yoe']
            self.df_cap = self.df_num1.copy()
            def iqr_capping(df, cols, factor):  # handing the outliers
                for col in cols:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_limit = q1 - (factor * iqr)
                    upper_limit = q3 + (factor * iqr)
                    df[col] = np.where(df[col] > upper_limit, upper_limit,
                                       np.where(df[col] < lower_limit, lower_limit, df[col]))
                    iqr_capping(self.df_cap, self.df_num1, 1.5)
                    print("yuyy")

        except Exception as E:
            print(f'Error in main - {E.__str__()}')




if __name__ == '__main__':  # main mathod (main processor)
    try:
        obj = CCFA('J:\Courses\Vihara tech (Internship)\Projects\Pro2_Credit Card/creditcard.csv')
        # use forward slash (clien's requirement)
        # create instance(object) and do operations
        # for preprossing will use a function
        obj.preprossing()
        obj.HNV()
        obj.VT()
        obj.HO()
    except Exception as E:
        print(f'Error in main - {E.__str__()}')

