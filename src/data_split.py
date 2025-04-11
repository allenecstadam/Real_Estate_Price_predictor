from sklearn.model_selection import train_test_split

def split_data(df, target_column):
    """
    Splits the dataset into training and testing sets.
    Returns X_train, X_test, y_train, y_test.
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)
    