
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

customers = pd.read_csv('Ecommerce Customers.csv')

print(customers.head())
print(customers.info())
print(customers.describe())
print(customers.columns)

print(customers.isnull().sum())

sns.pairplot(customers)
plt.show()

sns.histplot(customers['Yearly Amount Spent'])
plt.show()

sns.heatmap(customers.select_dtypes(include='number').corr())
plt.show()

X = customers[['Avg. Session Length', 'Time on App',
               'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=101
)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, y_train)

print(lm.intercept_)

coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

predictions = lm.predict(X_test)

plt.scatter(y_test, predictions)
plt.show()

sns.histplot((y_test - predictions), bins=50)
plt.show()

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
