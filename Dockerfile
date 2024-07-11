FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY StockTrainModel.ipynb /app/StockTrainModel.ipynb
COPY APP /app/APP
COPY DATA /app/DATA
COPY MODEL /app/MODEL

RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

COPY . /app

EXPOSE 9999

CMD ["python", "APP/StockApplication.py"]
