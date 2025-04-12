
from src.data_processing import load_data
from src.eda import basic_eda
from src.preprocessing import preprocess_data
from src.data_split import split_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model


def main():
    # Load dataset
    df = load_data(r"data/final.csv")
    # Perform basic EDA
    basic_eda(df)

    # Preprocess the data
    df = preprocess_data(df)

    # Split the data into training and testing sets
    
    X_train, X_test, y_train, y_test = split_data(df, "price")  # Ensure column name matches your dataset
    

    model = train_model(X_train, y_train)
    
    # Evaluate the model
    mae, mse = evaluate_model(model, X_test, y_test)
    print(f"Model Evaluation:\nMAE: {mae}\nMSE: {mse}")
    
if __name__ == "__main__":
    main()
