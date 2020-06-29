import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix,mean_squared_error,mean_absolute_error
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import precision_score,recall_score

def Linear_regression(x_train,x_test,y_train,y_test):
    if st.sidebar.button("Predict on dataset",key = "predict"):
            st.subheader("Linear Regression Results")
            model = LinearRegression()
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)
            mse = mean_squared_error(y_test,y_pred)
            mae = mean_absolute_error(y_test,y_pred)
            st.write("Accuracy:",accuracy.round(2))
            #st.write("Precision:",precision_score(y_test,y_pred).round(2))
            #st.write("Recall:",recall_score(y_test,y_pred).round(2))
            st.write("Mean Squared Error:",mse.round(2))
            st.write("Root Mean Squared Error:",np.sqrt(mse).round(2))
            st.write("Mean Absolute Error:",mae.round(2))
    if st.sidebar.checkbox("Predict on your data",key = "predictC"):
        cylinders = st.sidebar.number_input("cylinders in vechile",1,20,
                                          step = 1,key = 'cylinders')

        displacement = st.sidebar.slider("Displacement of Engine(cc)",100,
                                         1000,key = 'displacement')

        displacement = displacement*0.0610237

        horsepower = st.sidebar.slider("Horse Power(HP)",50,1000,
                                       key = 'horsepower')

        weight = st.sidebar.slider("Weight of Vechile(kg)",1000,10000,
                                   key = 'weight')
        weight = weight * 2.20462

        acceleration = st.sidebar.slider("Acceleration(0 to 60 kmph/sec)",5,30,
                                         key = 'acceleration')

        acceleration = acceleration*0.62137119223733

        model_year = st.sidebar.number_input("Model Year(from 1900 to 2010)",
                                             1900,2020,step = 1,key = 'model_year')

        model_year = model_year%100



        origin = st.sidebar.selectbox("Origin of Car",("American","European",
                                                       "Asian"))
        origin_value1 = 0
        origin_value2 = 0
        origin_value3 = 0


        if origin == 'American':
            origin_value1 = 1
            origin_value2 = 0
            origin_value3 = 0

        elif origin == 'European':
            origin_value1 = 0
            origin_value2 = 1
            origin_value3 = 0

        elif origin == 'Japanese':
            origin_value1 = 0
            origin_value2 = 0
            origin_value3 = 1


        if st.sidebar.button("Predict on custom data",key = 'predict_custom'):
            custom = np.array([[cylinders,displacement,horsepower,weight,
                                acceleration,model_year,origin_value1,
                               origin_value2,origin_value3]])

            model = LinearRegression()
            model.fit(x_train,y_train)

            y_pred = model.predict(custom)
            value  = abs(y_pred.item())*0.43
            st.write("Your vechile's mileage is :",value," kmpl")
            st.write("Your vechile's mileage is :",value/0.43," mpg")
            
            
def Ridge_regression(x_train,x_test,y_train,y_test):
    st.sidebar.subheader("Model Hyperparameteres")
    alpha = st.sidebar.slider("Regularization parameter",1.0,10.0,step = 0.01,
                              key = 'alpha')
    max_iter = st.sidebar.slider("Max number of iterarions",100,500,key = 'max_iter')
    
    if st.sidebar.button("Predict on dataset",key = "predict"):
            st.subheader("Ridge Regression Results")
            model = Ridge(alpha = alpha,max_iter = max_iter,
                                       random_state = 0)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)
            mse = mean_squared_error(y_test,y_pred)
            mae = mean_absolute_error(y_test,y_pred)
            st.write("Accuracy:",accuracy.round(2))
            #st.write("Precision:",precision_score(y_test,y_pred).round(2))
            #st.write("Recall:",recall_score(y_test,y_pred).round(2))
            st.write("Mean Squared Error:",mse.round(2))
            st.write("Root Mean Squared Error:",np.sqrt(mse).round(2))
            st.write("Mean Absolute Error:",mae.round(2))
    if st.sidebar.checkbox("Predict on your data",key = "predictC"):
        cylinders = st.sidebar.number_input("cylinders in vechile",1,20,
                                          step = 1,key = 'cylinders')

        displacement = st.sidebar.slider("Displacement of Engine(cc)",100,
                                         1000,key = 'displacement')

        displacement = displacement*0.0610237

        horsepower = st.sidebar.slider("Horse Power(HP)",50,1000,
                                       key = 'horsepower')

        weight = st.sidebar.slider("Weight of Vechile(kg)",1000,10000,
                                   key = 'weight')
        weight = weight * 2.20462

        acceleration = st.sidebar.slider("Acceleration(0 to 60 kmph/sec)",5,30,
                                         key = 'acceleration')

        acceleration = acceleration*0.62137119223733

        model_year = st.sidebar.number_input("Model Year(from 1900 to 2010)",
                                             1900,2020,step = 1,key = 'model_year')

        model_year = model_year%100



        origin = st.sidebar.selectbox("Origin of Car",("American","European",
                                                       "Asian"))
        origin_value1 = 0
        origin_value2 = 0
        origin_value3 = 0


        if origin == 'American':
            origin_value1 = 1
            origin_value2 = 0
            origin_value3 = 0
            
        elif origin == 'European':
            origin_value1 = 0
            origin_value2 = 1
            origin_value3 = 0

        elif origin == 'Japanese':
            origin_value1 = 0
            origin_value2 = 0
            origin_value3 = 1


        if st.sidebar.button("Predict on custom data",key = 'predict_custom'):
            custom = np.array([[cylinders,displacement,horsepower,weight,
                                acceleration,model_year,origin_value1,
                               origin_value2,origin_value3]])

            model = Ridge(alpha = alpha,max_iter = max_iter,
                                       random_state = 0)
            model.fit(x_train,y_train)

            y_pred = model.predict(custom)
            st.write("Your vechile's mileage is :",abs(y_pred.item())*0.43," kmpl")
            st.write("Your vechile's mileage is :",abs(y_pred.item())," mpg")
            
            
            
            
def Lasso_regression(x_train,x_test,y_train,y_test):
    st.sidebar.subheader("Model Hyperparameteres")
    alpha = st.sidebar.slider("Regularization parameter",1.0,10.0,step = 0.01,
                              key = 'alpha')
    max_iter = st.sidebar.slider("Max number of iterarions",1000,5000,key = 'max_iter')
    
    if st.sidebar.button("Predict on dataset",key = "predict"):
            st.subheader("Lasso Regression Results")
            model = Lasso(alpha = alpha,max_iter = max_iter,
                                       random_state = 0)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)
            mse = mean_squared_error(y_test,y_pred)
            mae = mean_absolute_error(y_test,y_pred)
            st.write("Accuracy:",accuracy.round(2))
            #st.write("Precision:",precision_score(y_test,y_pred).round(2))
            #st.write("Recall:",recall_score(y_test,y_pred).round(2))
            st.write("Mean Squared Error:",mse.round(2))
            st.write("Root Mean Squared Error:",np.sqrt(mse).round(2))
            st.write("Mean Absolute Error:",mae.round(2))
    if st.sidebar.checkbox("Predict on your data",key = "predictC"):
        cylinders = st.sidebar.number_input("cylinders in vechile",1,20,
                                          step = 1,key = 'cylinders')

        displacement = st.sidebar.slider("Displacement of Engine(cc)",100,
                                         1000,key = 'displacement')

        displacement = displacement*0.0610237

        horsepower = st.sidebar.slider("Horse Power(HP)",50,1000,
                                       key = 'horsepower')

        weight = st.sidebar.slider("Weight of Vechile(kg)",1000,10000,
                                   key = 'weight')
        weight = weight * 2.20462

        acceleration = st.sidebar.slider("Acceleration(0 to 60 kmph/sec)",5,30,
                                         key = 'acceleration')

        acceleration = acceleration*0.62137119223733

        model_year = st.sidebar.number_input("Model Year(from 1900 to 2010)",
                                             1900,2020,step = 1,key = 'model_year')

        model_year = model_year%100



        origin = st.sidebar.selectbox("Origin of Car",("American","European",
                                                       "Asian"))
        origin_value1 = 0
        origin_value2 = 0
        origin_value3 = 0


        if origin == 'American':
            origin_value1 = 1
            origin_value2 = 0
            origin_value3 = 0

        elif origin == 'European':
            origin_value1 = 0
            origin_value2 = 1
            origin_value3 = 0

        elif origin == 'Japanese':
            origin_value1 = 0
            origin_value2 = 0
            origin_value3 = 1


        if st.sidebar.button("Predict on custom data",key = 'predict_custom'):
            custom = np.array([[cylinders,displacement,horsepower,weight,
                                acceleration,model_year,origin_value1,
                               origin_value2,origin_value3]])

            model = Lasso(alpha = alpha,max_iter = max_iter,
                                       random_state = 0)
            model.fit(x_train,y_train)

            y_pred = model.predict(custom)
            st.write("Your vechile's mileage is :",abs(y_pred.item())*0.43," kmpl")
            st.write("Your vechile's mileage is :",abs(y_pred.item())," mpg")
            
            
def Random_forest_regression(x_train,x_test,y_train,y_test):
    st.sidebar.subheader("Model Hyperparameteres")
    n_estimators = st.sidebar.slider("number of trees in forest",100,1000,step = 1,
                              key = 'n_estimators')
    min_samples_split = st.sidebar.slider("minimum samples required to split a node",
                                          2,15,step = 1,key = 'min_samples_split')
    
    max_depth = st.sidebar.number_input("maximum depth of tree",2,15,step = 1,
                                        key = 'max_depth')
    
    
    if st.sidebar.button("Predict on dataset",key = "predict"):
            st.subheader("Random Forest Regression Results")
            model = RandomForestRegressor(n_estimators = n_estimators,
                                          min_samples_split = min_samples_split,
                                          max_depth = max_depth,n_jobs = -1,
                                          random_state = 0)
                                          
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)
            mse = mean_squared_error(y_test,y_pred)
            mae = mean_absolute_error(y_test,y_pred)
            st.write("Accuracy:",accuracy.round(2))
            #st.write("Precision:",precision_score(y_test,y_pred).round(2))
            #st.write("Recall:",recall_score(y_test,y_pred).round(2))
            st.write("Mean Squared Error:",mse.round(2))
            st.write("Root Mean Squared Error:",np.sqrt(mse).round(2))
            st.write("Mean Absolute Error:",mae.round(2))
    if st.sidebar.checkbox("Predict on your data",key = "predictC"):
        cylinders = st.sidebar.number_input("cylinders in vechile",1,20,
                                          step = 1,key = 'cylinders')

        displacement = st.sidebar.slider("Displacement of Engine(cc)",100,
                                         1000,key = 'displacement')

        displacement = displacement*0.0610237

        horsepower = st.sidebar.slider("Horse Power(HP)",50,1000,
                                       key = 'horsepower')

        weight = st.sidebar.slider("Weight of Vechile(kg)",1000,10000,
                                   key = 'weight')
        weight = weight * 2.20462

        acceleration = st.sidebar.slider("Acceleration(0 to 60 kmph/sec)",5,30,
                                         key = 'acceleration')

        acceleration = acceleration*0.62137119223733

        model_year = st.sidebar.number_input("Model Year(from 1900 to 2010)",
                                             1900,2020,step = 1,key = 'model_year')

        model_year = model_year%100



        origin = st.sidebar.selectbox("Origin of Car",("American","European",
                                                       "Asian"))
        origin_value1 = 0
        origin_value2 = 0
        origin_value3 = 0


        if origin == 'American':
            origin_value1 = 1
            origin_value2 = 0
            origin_value3 = 0
            
        elif origin == 'European':
            origin_value1 = 0
            origin_value2 = 1
            origin_value3 = 0

        elif origin == 'Japanese':
            origin_value1 = 0
            origin_value2 = 0
            origin_value3 = 1


        if st.sidebar.button("Predict on custom data",key = 'predict_custom'):
            custom = np.array([[cylinders,displacement,horsepower,weight,
                                acceleration,model_year,origin_value1,
                               origin_value2,origin_value3]])

            model = RandomForestRegressor(n_estimators = n_estimators,
                                          min_samples_split = min_samples_split,
                                          max_depth = max_depth,n_jobs = -1,
                                          random_state = 0)
            model.fit(x_train,y_train)

            y_pred = model.predict(custom)
            st.write("Your vechile's mileage is :",abs(y_pred.item())*0.43," kmpl")
            st.write("Your vechile's mileage is :",abs(y_pred.item())," mpg")
    
    

    

def main():
    st.title("Mileage Prediction Web App")
    st.sidebar.title("Mileage Prediction Web App")
    st.markdown("What is the mileage of your car?🚗")
    st.sidebar.markdown("Predict the mileage of your car 🚗")
    
    
    @st.cache(persist = True)
    def load_data():
        data = pd.read_csv("auto-mpg.csv")
        data_calc = pd.read_csv("milesPerGallon.csv")
        del data_calc['Unnamed: 0']
        return data,data_calc
    
    @st.cache(persist = True)
    def split(df_calc):
        y = df_calc.mpg
        x = df_calc.drop(columns = ['mpg'])
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,
                                                         random_state = 0)
        return x_train,x_test,y_train,y_test
    
    
            
            
    
    
    
    df,df_calc = load_data()
    x_train,x_test,y_train,y_test = split(df_calc)
    st.sidebar.subheader("Choose Regression Model")
    regression_model = st.sidebar.selectbox("Model",("Linear Regression",
                                               "Ridge Regression",
                                                "Lasso Regression",
                                               "Random Forest Regression"))
    
    
    
    
    if regression_model == 'Linear Regression':
        Linear_regression(x_train,x_test,y_train,y_test)
        
        
    if regression_model == 'Ridge Regression':
        Ridge_regression(x_train,x_test,y_train,y_test)
        
    if regression_model == 'Lasso Regression':
        Lasso_regression(x_train,x_test,y_train,y_test)
        
    if regression_model == 'Random Forest Regression':
        Random_forest_regression(x_train,x_test,y_train,y_test)

        
        
                
                
             
            
            
            
            

            
            
            
            
            

            
                    
            

                                         
    
    
    if st.sidebar.checkbox("Show raw data",False):
        st.subheader("auto-mpg dataset")
        st.write(df)

   
        
                                         
                                         
                                         
                             
                             
    
    
    
    
    
if __name__ == '__main__':
    main()