from typing import Union, Tuple
from appwrite.client import Client
from appwrite.query import Query
from appwrite.services.databases import Databases
import os
import joblib
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Load the model from file
pipeline = joblib.load('transaction_classifier.pkl')


def predict_category(payee: str):
    """
    Predict the category of a transaction

    :param payee:
    :return:
    """
    # Preprocess the payee
    payee = payee.lower().strip()
    payee = re.sub(r'eindhoven', '', payee)
    payee = re.sub(r'[^\w\s]|https?://\S+|www\.\S+|https?:/\S+|[^\x00-\x7F]+|\d+', '',
                   str(payee).strip())

    # Prediction
    prediction = pipeline.predict([payee])[0]

    return prediction


def update_categories() -> Union[None, Tuple[int, int]]:
    """
    Retrieve documents from the Appwrite database

    :return: Documents from transaction collection
    """
    client = (
        Client()
        .set_endpoint(os.environ["NEXT_PUBLIC_APPWRITE_ENDPOINT"])
        .set_project(os.environ["NEXT_PUBLIC_APPWRITE_PROJECT"])
        .set_key(os.environ["NEXT_APPWRITE_KEY"])
    )

    databases = Databases(client)

    # List documents in the collection
    result = databases.list_documents(
        database_id=os.environ["APPWRITE_DATABASE_ID"],
        collection_id=os.environ["APPWRITE_TRANSACTION_COLLECTION_ID"],
        queries=[
            Query.limit(5000)
        ]
    )

    if result:
        print("Fetched transactions")

        documents = result['documents']
        total = int(result['total'])

        count = 0
        for document in documents:
            payee = document['Payee']
            category = predict_category(payee)
            # Update the category in the database
            databases.update_document(
                database_id=os.environ["APPWRITE_DATABASE_ID"],
                collection_id=os.environ["APPWRITE_TRANSACTION_COLLECTION_ID"],
                document_id=document['$id'],
                data={'category': category}
            )
            count += 1
        return count, total
    else:
        return None


@app.get("/update-categories")
def main():
    """
    Main function to retrieve transactions from the Appwrite database
    """
    # Update the categories
    count, total = update_categories()

    if count:
        return {"message": f"Updated {count} out of {total} transactions"}
    else:
        return {"message": "No transactions found"}


@app.get("/")
def root():
    return {
        "motto": "Build like a team of hundreds_",
        "learn": "https://appwrite.io/docs",
        "connect": "https://appwrite.io/discord",
        "getInspired": "https://builtwith.appwrite.io",
    }


class PayeeRequest(BaseModel):
    Payee: str


@app.post("/predict")
def predict(request: PayeeRequest):
    """
    Endpoint to predict the category of a transaction
    """
    try:
        category = predict_category(request.Payee)
        return {"category": category}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
