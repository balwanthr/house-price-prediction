# house-price-prediction
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
dataset = pd.read_excel("HousePricePrediction.xlsx")
 
# Printing first 5 records of the dataset
print(dataset.head(5))
