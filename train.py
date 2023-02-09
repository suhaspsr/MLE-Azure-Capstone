from sklearn.linear_model import LogisticRegression
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from sklearn.utils import resample


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0,
                        help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    parser.add_argument('--solver', type=str, default='lbfgs', help="Algorithm to use in the optimization problem.")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    run.log("Solver:", args.solver)

    
    # Read data
    def get_data():
        columns = ['A'+str(i) for i in range(1,17)]
        df = pd.read_csv('data.csv', header = None, names = columns )
        df.dropna(inplace = True)

        # Removing Null
        for column in df.columns:
            df = df[df[column] != '?']

        # Converting attributes to binary
        df['A16'] = df['A16'].map({'+': 1, '-':0})    
        df['A1']  = df['A1'].map({'b': 1, 'a':0})
        df['A9']  = df['A9'].map({'t': 1, 'f':0})
        df['A10'] = df['A10'].map({'t': 1, 'f':0})
        df['A12'] = df['A12'].map({'t': 0, 'f':1})
        df['A13'] = df['A13'].map({'g': 'mm', 'p':'kk'})

        # Conversting categorical data into onehot encoding
        cat_columns = ['A4', 'A5', 'A6', 'A7', 'A13']
        for column in cat_columns:
            dummies = pd.get_dummies(df[column])
            df[dummies.columns] = dummies
            df.drop(columns = column,inplace = True)
        df = df.astype(float)
        df['A16'] = df['A16'].astype(int)
        return df

    df = get_data()
    y = df['A16']
    x = df.drop(['A16'], axis = 1)

    # Split data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=99)

    # Train Logistic Regression Model
    model = LogisticRegression(C=args.C, max_iter=args.max_iter, solver=args.solver).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))


if __name__ == '__main__':
    main()
