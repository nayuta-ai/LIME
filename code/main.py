import pandas as pd
import numpy as np
from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

boston = load_boston()
df = pd.DataFrame(boston.data, columns = boston.feature_names)
# print(df)
df['MEDV'] = boston.target
feature_names = np.array(df.drop('MEDV', axis=1).columns)
x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, 0:13], df.iloc[:, 13],
                                                    test_size=0.2, random_state=1)
RF = RandomForestRegressor()
RF.fit(x_train, y_train)

explainer = LimeTabularExplainer(training_data=np.array(x_train),
                                 feature_names=feature_names,
                                 training_labels=np.array(y_train),
                                 discretize_continuous=True,
                                 mode='regression',
                                 verbose=True,
                                )

predict_proba = lambda x: np.array(list(zip(1-RF.predict(x), RF.predict(x))))

exp = explainer.explain_instance(
    x_train.iloc[0], 
    predict_proba, 
    num_features=x_train.columns.shape[0]
)
exp.show_in_notebook(show_all=False)