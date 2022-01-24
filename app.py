import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Deployment purposes
from custom_transformer import column_transformer
import tensorflow as tf
import numpy as np


# Sidebar widget
st.sidebar.title("Telco Churn Analysis")
st.sidebar.header('Milestone 1 Phase 2')
st.sidebar.markdown("Dashboard Analisis Churn data At Telco Company")
# loading our model
my_keras_model = tf.keras.models.load_model("my_keras_model.h5")


def main():
    page = st.sidebar.selectbox(
        "Select a page", ["Homepage", "Exploration", "Model" ,"Prediction"])

    if page == "Homepage":
        homepage_screen()
    elif page == "Exploration":
        exploration_screen()
    elif page == "Model":
        model_screen()
    elif page == "Prediction":
        model_predict()


@st.cache()
def load_data():
    data = pd.read_csv('clean_data.csv', delimiter=",")
    return data


df = load_data()


def homepage_screen():
    st.image("analysis.jpg",use_column_width=True)
    st.title('CHURN PREDICTION')
    #st.header("Dataset Information")
    st.write("""  
        **About Dataset**  

        Each row represents a customer, each column contains customer’s attributes described on the column Metadata.

        The data set includes information about:

        Customers who left within the last month – the column is called Churn
        Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
        Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
        Demographic info about customers – gender, age range, and if they have partners and dependents
        
        The dataset constains of 7043 instances and 21columns. More information of the dataset can be accessed [here](https://www.kaggle.com/blastchar/telco-customer-churn).
        
    """)
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv', delimiter=",")
    
        # Load data
    if st.checkbox("Show Dataset"): #checkbox untuk menampilkan dataset
            number=st.number_input("Number of Rows to view",5,15)
            st.dataframe(df.head(number))
            st.success("Data loaded successfully") 
            
            
            data_dim= st.radio("Shape of the dataset:", ("Number of Rows","Number of Columns")) #radio button widget
            if st.button("Show Dimension"):
                if data_dim== 'Number of Rows':
                    st.write(df.shape[0])
                elif data_dim== 'Number of Columns':
                    st.write(df.shape[1])
                else:
                    st.write(df.shape)

            Info =['Display the dataset summary','Check for missing values in the dataset']
            options=st.selectbox("Pilihan - pilihan terhadap dataset",Info)
            
            if options=='Check for missing values in the dataset': 
                 st.write(df.isnull().sum(axis=0)) #cekk null values
                 if st.button("Drop Null Values"):
                     df=df.dropna() #drop null values
                     st.success("Null values droped successfully")
            
            if options=='Display the dataset summary':
                 st.write(df.describe().T)    



def exploration_screen():
    st.image("EDA.png",use_column_width=True)
    st.title("Data Exploration")
    st.write(""" 
        This page contains general exploratory data analysis in order to get basic insight of the dataset information and get the feeling about what this dataset is about.
    """)

    st.write("""
        #### 📌  Features Correlation Value toward Target Column
        
    """)
    # Display correlation towards target column
    fig, axs = plt.subplots(figsize=(10, 4))
    corr_df = pd.read_csv('corr_data.csv', delimiter=",")
    corr = corr_df.corr()['Churn'].reset_index()
    sns.barplot(data=corr, x='index', y='Churn', ax=axs)
    plt.xticks(rotation=70)
    st.write(fig)

    st.write("""
        #### 📌 Target Label Frequency  
        
    """)

    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
    df['Churn'].value_counts().plot(kind='bar', ax=axs[0])
    df['Churn'].value_counts().plot.pie(
        autopct='%1.1f%%', startangle=90, ax=axs[1], colors=['green', 'teal'])
    st.write(fig)
    
    st.write("""
        #### 📌  MonthlyCharges and Tenure
        
    """)
    fig,axs = plt.subplots()
    # fig = plt.figure(figsize=(5,5))
    sns.scatterplot(data=df, x='MonthlyCharges', y='TotalCharges', hue='Churn', ax=axs)
    st.write(fig)

    st.write("""
        #### 📌  TotalCharges and Tenure
        
    """)
    fig, axs = plt.subplots()
    sns.scatterplot(data=df, y='TotalCharges', x='Tenure', hue='Churn', ax=axs)
    st.write(fig)

    st.write("""
        #### 📌  PaymentMethod.
        
    """)
    fig, axs = plt.subplots()
    sns.countplot(data=df, x='PaymentMethod', hue='Churn',ax=axs)
    plt.xticks(rotation=40)
    plt.show()
    st.write(fig)

    st.write("""
        #### 📌  Contract
        
    """)
    fig, axs = plt.subplots()
    sns.countplot(data=df, x='Contract', hue='Churn', ax=axs)
    st.write(fig)
    
    st.write("""
        Customer with two years contract
        
    """)
    data = df[df['Contract'] == 'Two year']

    fig, axs = plt.subplots()
    sns.kdeplot(data=data, x="TotalCharges", hue='Churn',ax=axs);
    st.write(fig)
    
    
    st.write("""
        #### 📌  Internet Services
        
    """)
    fig, axs = plt.subplots()
    sns.countplot(data=df, x='InternetService', hue='Churn', ax=axs);
    st.write(fig)


def model_screen():
    st.image("ANN.jpg",use_column_width=True)
    st.title("Model")
    st.write(""" 
             ANN Model Evaluation
             """)
    model_selected = st.selectbox("Select Evaluation Type: ", ['History Tracking Sequential','History Tracking Functional'])
   
    if model_selected ==  'History Tracking Sequential':
        st.image("seq.png",use_column_width=True)
        st.write(f"""
                👉 Validation's loss value: 0.4713919460773468
                👉 Validation's accuracy value: 0.7547974586486816
                """)
        validation(np.array([[784,  96],[249, 278]]))

        
    if model_selected ==  'History Tracking Functional':
        st.image("seq.png",use_column_width=True)
        st.write(f"""
                👉 Validation's loss value: 0.47096991539001465
                👉 Validation's accuracy value: 0.7725657224655151
                """)
        validation(np.array([[835,  122],[198, 252]]))    

def validation(matrix):
    fig, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, cmap="YlGnBu", fmt='g', ax=ax)
    ax.set_xlabel('PREDICTED')
    ax.set_ylabel('ACTUAL')
    ax.set_title('Confusion Matrix')
    st.write(fig)

def model_predict():
    st.image("churn.png",use_column_width=True)
    st.title("Prediction")
    st.write("### Predict the posibility of customer to churn here !")
    
    SeniorCitizen = st.radio("Is the customer a senior citizen", ['Yes', 'No'])
    Tenure = st.slider(
        "How long have been joined the Telco product", 0, 70)
    MultipleLines =st.radio("Have multiple lines of phone service?", ['Yes', 'No'])
    InternetService =  st.selectbox(
        "Kind of internet service?", ['Fiber optic','DSL','No'])
    OnlineSecurity = st.selectbox(
        "Have online security?", ['No','Yes'])
    TechSupport =st.selectbox(
        "Have techsupport?", ['No','Yes'])
    Contract = st.radio(
        "Kind of customer contract?", ['One year','Month-to-month','Two year'])
    PaperlessBilling = st.radio(
        "Have paperless billing?", ['Yes', 'No'])
    PaymentMethod = st.selectbox(
        "What payment method does the customer have?", ['Electronic check' , 'Mailed check', 'Bank transfer (automatic)',
    'Credit card (automatic)'])
    MonthlyCharges =st.number_input(label="Monthly charges", min_value=0,max_value=120, step=1,)
    TotalCharges =st.number_input(label="Total charges", min_value=0,max_value=8000, step=1,)
    submit_button = st.button("Predict")

    if SeniorCitizen == 'Yes':
        SeniorCitizen = 1
    else:
        SeniorCitizen = 0

    data = {
        'SeniorCitizen': [SeniorCitizen], 'Tenure': [Tenure], 'MultipleLines': [MultipleLines], 'InternetService': [InternetService], 'OnlineSecurity': [OnlineSecurity],'TechSupport': [TechSupport], 'Contract': [Contract], 'PaperlessBilling': [PaperlessBilling], 'PaymentMethod': [PaymentMethod], 'MonthlyCharges': [MonthlyCharges], 'TotalCharges': [TotalCharges]
    }
    transformer = column_transformer()
    X = df.drop('Churn', axis=1)
    fitting = transformer.fit_transform(X)
    new_data = pd.DataFrame(data=data)
    new_data =transformer.transform(new_data)
    
    if submit_button:
        result = my_keras_model.predict(new_data)
        updated_res = result.flatten().astype(float)
      
        if updated_res[0] > 0.50:
            updated_res = f"😱 Probability value: {np.ceil(updated_res[0]*100)} %. This customer might churn."
        else:
            updated_res = f"🤩 Probability value:  {np.ceil(updated_res[0]*100)} %. This customer might not churn."
        st.success(
            '{}'.format(updated_res))


main()
