import joblib
import os
import re
import pandas as pd
from appwrite.client import Client
from appwrite.services.databases import Databases
from sklearn.model_selection import train_test_split


def fetch_data_from_appwrite():
    client = (
        Client()
        .set_endpoint("https://cloud.appwrite.io/v1")
        .set_project(os.environ["APPWRITE_FUNCTION_PROJECT_ID"])
        .set_key(os.environ["APPWRITE_API_KEY"])
    )

    databases = Databases(client)

    result = databases.list_documents(
        database_id=os.environ["APPWRITE_DATABASE_ID"],
        collection_id=os.environ["APPWRITE_TRANSACTION_COLLECTION_ID"]
    )

    if result:
        documents = result['documents']
        data = []
        for document in documents:
            payee = document['Payee']
            category = document['category']
            data.append({"payee": payee, "category": category})
        return pd.DataFrame(data)
    else:
        return pd.DataFrame(columns=["payee", "category"])


def preprocess_text(text):
    text = text.lower().strip()
    text = re.sub(r'eindhoven', '', text)
    text = re.sub(r'[^\w\s]|https?://\S+|www\.\S+|https?:/\S+|[^\x00-\x7F]+|\d+', '', text.strip())
    return text


def retrain_model():
    # Load the existing model
    model_path = 'transaction_classifier.pkl'
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    pipeline = joblib.load(model_path)

    # Fetch data from Appwrite
    data = fetch_data_from_appwrite()

    if data.empty:
        print("No data fetched from Appwrite.")
        return

    # Preprocess the payee text
    data['payee'] = data['payee'].apply(preprocess_text)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data['payee'], data['category'], test_size=0.2, random_state=42)

    # Retrain the existing model
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    score = pipeline.score(X_test, y_test)
    print(f"Model accuracy: {score}")

    # Save the updated model to file
    joblib.dump(pipeline, model_path)


if __name__ == "__main__":
    retrain_model()
