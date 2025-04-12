import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.data_processing import load_data
from src.preprocessing import preprocess_data
from src.data_split import split_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model

st.title("🏡 Real Estate Price Prediction & Visualizations")

# Load dataset from the repo (final.csv)
df = pd.read_csv(r"data\\final.csv")

st.write("### Data Preview")
st.dataframe(df.head())

    # Preprocess the data and train the model
df_processed = preprocess_data(df.copy())

X_train, X_test, y_train, y_test = split_data(df_processed, "price")  # using lowercase "price"
    

model = train_model(X_train, y_train)


mae, mse = evaluate_model(model, X_test, y_test)

st.write("### Model Evaluation")
st.write(f"**MAE:** {mae}")
st.write(f"**MSE:** {mse}")

# Generate dropdown options from the dataset
year_sold = sorted(df["year_sold"].unique())
property_tax = sorted(df["property_tax"].unique())
insurance = sorted(df["insurance"].unique())
beds = sorted(df["beds"].unique())
baths = sorted(df["baths"].unique())
sqft = sorted(df["sqft"].unique())
year_built = sorted(df["year_built"].unique())
lot_size = sorted(df["lot_size"].unique())
property_age = sorted(df["property_age"].unique())

st.write("### Predict a Property Price")
with st.form("predict_form"):
        st.subheader("Enter Property Details")
        # Dropdowns for each numeric parameter
        
        year_sold = st.selectbox("Year Sold", options=year_sold)
        
        
        property_tax = st.selectbox("Property Tax", options=property_tax)
        
       
        insurance = st.selectbox("Insurance", options=insurance)
        
        
        beds = st.selectbox("Beds", options=beds)
        
        
        baths = st.selectbox("Baths", options=baths)
        
       
        sqft = st.selectbox("Sqft", options=sqft)
        
        
        year_built = st.selectbox("Year Built", options=year_built)
        
        
        lot_size = st.selectbox("Lot Size", options=lot_size)
        
        property_age = st.selectbox("Property Age", options=property_age)
        
        # Dropdowns for binary/categorical variables
        basement = st.selectbox("Basement (0 = No, 1 = Yes)", options=[0, 1])
        popular = st.selectbox("Popular (0 = No, 1 = Yes)", options=[0, 1])
        recession = st.selectbox("Recession (0 = No, 1 = Yes)", options=[0, 1])
        property_type = st.selectbox("Property Type", options=["Bunglow", "Condo"])
        
        submitted = st.form_submit_button("Predict Price")
    
if submitted:
        # Create dictionary for user input
        input_dict = {
            "year_sold": year_sold,
            "property_tax": property_tax,
            "insurance": insurance,
            "beds": beds,
            "baths": baths,
            "sqft": sqft,
            "year_built": year_built,
            "lot_size": lot_size,
            "basement": basement,
            "popular": popular,
            "recession": recession,
            "property_age": property_age,
        }
        # Create dummy variables for property type
        if property_type == "Bunglow":
            input_dict["property_type_Bunglow"] = 1
            input_dict["property_type_Condo"] = 0
        else:
            input_dict["property_type_Bunglow"] = 0
            input_dict["property_type_Condo"] = 1

        # Convert dictionary to DataFrame
        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=X_train.columns, fill_value=0)

        # Generate prediction
        prediction = model.predict(input_df)[0]
        st.write("### Predicted Price")
        st.write(f"${prediction:,.2f}")
        
        st.write("### Visualizations with Prediction Marker")
        
        # Visualization 1: Price Distribution with predicted price marker
        st.write("#### Distribution of Price")
        plt.figure()
        plt.hist(df["price"], bins=30, edgecolor='black')
        plt.axvline(x=prediction, color='red', linestyle='--', label="Predicted Price")
        plt.xlabel("Price")
        plt.ylabel("Frequency")
        plt.title("Distribution of Price")
        plt.legend()
        st.pyplot(plt.gcf())
        
        # Visualization 2: Scatter Plot (sqft vs. price) with input point highlighted
        st.write("#### Scatter Plot: Sqft vs Price")
        plt.figure()
        plt.scatter(df["sqft"], df["price"], alpha=0.5, label="Data Points")
        plt.scatter(sqft, prediction, color='red', s=100, label="Your Input")
        plt.xlabel("Sqft")
        plt.ylabel("Price")
        plt.title("Sqft vs Price")
        plt.legend()
        st.pyplot(plt.gcf())
        
        # Visualization 3: Correlation Heatmap
        st.write("#### Correlation Heatmap")
        plt.figure(figsize=(10, 8))
        corr = df.corr()
        im = plt.imshow(corr, cmap='viridis', interpolation='none')
        plt.colorbar(im)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title("Correlation Heatmap")
        st.pyplot(plt.gcf())
