from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("Train.csv")
df1 = df.copy()
val1 = df1['Item_Identifier'].unique()
for num,var in enumerate(val1):
    num+=1
    df1['Item_Identifier'].replace(var,num,inplace=True)
val2 = df1['Item_Fat_Content'].unique()
for num,var in enumerate(val2):
    num+=1
    df1['Item_Fat_Content'].replace(var,num,inplace=True)
val3 = df1['Item_Type'].unique()
for num,var in enumerate(val3):
    num+=1
    df1['Item_Type'].replace(var,num,inplace=True)
val4 = df1['Outlet_Identifier'].unique()
for num,var in enumerate(val4):
    num+=1
    df1['Outlet_Identifier'].replace(var,num,inplace=True)
val5 = df1['Outlet_Size'].unique()
for num,var in enumerate(val5):
    num+=1
    df1['Outlet_Size'].replace(var,num,inplace=True)
val6 = df1['Outlet_Location_Type'].unique()
for num,var in enumerate(val6):
    num+=1
    df1['Outlet_Location_Type'].replace(var,num,inplace=True)
val7 = df1['Outlet_Type'].unique()
for num,var in enumerate(val7):
    num+=1
    df1['Outlet_Type'].replace(var,num,inplace=True)
    
X = df1.drop(['Item_Outlet_Sales','Item_Visibility'],axis=1)
y = df1['Item_Outlet_Sales']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 456)

sc = StandardScaler()
sc.fit(X_train,y_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

model = joblib.load("Item_outlet_saleprice.pkl")

def Item_outlet_saleprice(model, Item_Identifier, Item_Weight, Item_Fat_Content,
                          Item_Type, Item_MRP, Outlet_Identifier, Outlet_Establishment_Year, Outlet_Size, Outlet_Location_Type, Outlet_Type):
   
    for num,var in enumerate(val1):
        if var == Item_Identifier:
            Item_Identifier = num
    for num,var in enumerate(val2):
        if var == Item_Fat_Content:
            Item_Fat_Content = num
    for num,var in enumerate(val3):
        if var == Item_Type:
            Item_Type = num
    for num,var in enumerate(val4):
        if var == Outlet_Identifier:
            Outlet_Identifier = num
    for num,var in enumerate(val5):
        if var == Outlet_Size:
            Outlet_Size = num
    for num,var in enumerate(val6):
        if var == Outlet_Location_Type:
            Outlet_Location_Type = num
    for num,var in enumerate(val7):
        if var == Outlet_Type:
            Outlet_Type = num
            
    x = np.zeros(len(X.columns))
    
    x[0] = Item_Identifier
    x[1] = Item_Weight
    x[2] = Item_Fat_Content
    x[3] = Item_Type
    x[4] = Item_MRP
    x[5] = Outlet_Identifier
    x[6] = Outlet_Establishment_Year
    x[7] = Outlet_Size
    x[8] = Outlet_Location_Type
    x[9] = Outlet_Type
    
    x = sc.transform([x])[0]
    return model.predict([x])[0]

app =Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    Item_Identifier = request.form["Item_Identifier"]
    Item_Weight = request.form["Item_Weight"]
    Item_Fat_Content = request.form["Item_Fat_Content"]
    Item_Type = request.form["Item_Type"]
    Item_MRP = request.form["Item_MRP"]
    Outlet_Identifier = request.form["Outlet_Identifier"]
    Outlet_Establishment_Year = request.form["Outlet_Establishment_Year"]
    Outlet_Size = request.form["Outlet_Size"]
    Outlet_Location_Type = request.form["Outlet_Location_Type"]
    Outlet_Type = request.form["Outlet_Type"]

    
    predicted_surv1 = Item_outlet_saleprice( model, Item_Identifier, Item_Weight, Item_Fat_Content, Item_Type,
                                            Item_MRP, Outlet_Identifier, Outlet_Establishment_Year, Outlet_Size, Outlet_Location_Type, Outlet_Type)
    predicted_surv = round(predicted_surv1, 2)
    
    return render_template("index.html", prediction_text="the outlet items are {}".format(predicted_surv))

if __name__ == "__main__":
    app.run()


