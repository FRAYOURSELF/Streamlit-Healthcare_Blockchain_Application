import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import sqlite3
import pandas
from matplotlib.figure import Figure

import pyspark_func as pyspark_func
import scikit_func as scikit_func

matplotlib.use("agg")
from matplotlib.backends.backend_agg import RendererAgg

_lock = RendererAgg.lock

st.set_page_config(
    page_title='Streamlit with Healthcare Data',
    layout="wide",
    initial_sidebar_state="expanded",
)


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    elif clf_name == 'Decision Tree':
        params['criterion'] = st.sidebar.radio("criterion", ('gini', 'entropy'))
        params['max_features'] = st.sidebar.selectbox("max_features", (None, 'auto', 'sqrt', 'log2'))
        params['max_depth'] = st.sidebar.slider('max_depth', 1, 32)
        params['min_samples_split'] = st.sidebar.slider('min_samples_split', 0.1, 1.0)
    return params


def pyspark_buildmodel(pyspark_classifier_name):
    spark = pyspark_func.get_spark_session()
    trainingData, testData = pyspark_func.prepare_dataset(spark, data)
    return pyspark_func.training(spark, pyspark_classifier_name, trainingData, testData)


def pyspark_operation(pyspark_col):
    st.sidebar.subheader('PySpark')
    pyspark_classifier_name = st.sidebar.selectbox(
        'Select classifier',
        pyspark_func.get_sidebar_classifier(), key='pyspark'
    )
    pyspark_col.write(f'Classifier = {pyspark_classifier_name}')
    accuracy = pyspark_buildmodel(pyspark_classifier_name)
    pyspark_col.write(f'Accuracy = {accuracy}')


def create_sidelayout(scipy_col, pyspark_col):
    st.sidebar.title('Machine Learning Options')
    st.sidebar.subheader('Scikit-Learn')
    scikit_classifier_name = st.sidebar.selectbox(
        'Select classifier',
        scikit_func.get_sidebar_classifier(), key='scikit'
    )
    scipy_col.write(f'Classifier = {scikit_classifier_name}')
    params = add_parameter_ui(scikit_classifier_name)
    accuracy = scikit_func.trigger_classifier(scikit_classifier_name, params, X_train, X_test, y_train, y_test)
    scipy_col.write(f'Accuracy =  {accuracy}')

    if pyspark_enabled == 'Yes':
        pyspark_operation(pyspark_col)

def get_table(table_name:str, file_location:str):
    table_name=pd.read_csv(file_location)
    return table_name


def create_subcol():
    """
    Create 2 Center page Columns for Scikit-Learn and Pyspark
    :return:
    """
    scipy_col, pyspark_col = st.columns(2)
    if pyspark_enabled == 'Yes':
        pyspark_col.header('''
        __PySpark Learn ML__  
        ![pyspark](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Apache_Spark_logo.svg/120px-Apache_Spark_logo.svg.png)
        ''')
    return scipy_col, pyspark_col


def plot():
    """
    Plotting some basic chart to demonstrate the data
    TODO: Add caching to avoid loading the plots for every event
    :return:
    """
    col_plot1, col_plot2 = st.columns(2)
    temp_df = data
    with col_plot1, _lock:
        st.subheader('Age over Number of people with CVD exceed')
        fig = Figure()
        ax = fig.subplots()
        temp_df['years'] = (temp_df['age'] / 365).round().astype('int')
        sns.countplot(x='years', hue='cardio', data=temp_df, palette="Set2", ax=ax)

        ax.set_xlabel('Year')
        ax.set_ylabel('Count')
        st.pyplot(fig)

    with col_plot2, _lock:
        st.subheader('People Exposed to CVD more')
        fig = Figure()
        ax = fig.subplots()
        df_categorical = temp_df.loc[:, ['cholesterol', 'gluc', 'smoke', 'alco', 'active']]
        sns.countplot(x="variable", hue="value", data=pd.melt(df_categorical), ax=ax)
        ax.set_xlabel('Variable')
        ax.set_ylabel('Count')
        st.pyplot(fig)


data = scikit_func.load_data()
X_train, X_test, y_train, y_test = scikit_func.prepare_dataset(data)
st.sidebar.markdown('''
Selecting _Yes_ will trigger Spark Session automatically!  
''', unsafe_allow_html=True)
pyspark_enabled = st.sidebar.radio("PySpark_Enabled", ('No', 'Yes'))


def main():
    st.markdown("""<h1><u>SecureYou</u></h1>""", unsafe_allow_html=True)
    st.markdown("""<h2 style="color:Tomato;">We keep your medical history safe and private for your convenience</h2>""", unsafe_allow_html=True)
    # st.title(
    #     '''Streamlit ![](https://assets.website-files.com/5dc3b47ddc6c0c2a1af74ad0/5e0a328bedb754beb8a973f9_logomark_website.png) Healthcare ML Data App''')

    st.subheader("Enter details")
    option=st.selectbox('Enter details as authority', ('', 'Register a medical History', 'Get information related to a patient'))
    if option=="Register a medical History":
        col11, col21=st.columns(2)
        with col11:
            Unique_id=st.text_input("Enter Unique ID", "")
            professional_name=st.text_input("Enter Professional Name", "");
            Diagnosis=st.text_input("Enter Diagnosis", "")
            button_press=st.button(label="Enter")
        with col21:
            patient_name=st.text_input("Enter Patient name", "")
            sex=st.text_input("Enter Patient's Sex", "")
            Fee_paid=st.number_input("Enter fee paid")
        file='file.csv'
        df=get_table('df', file)
        df=pd.read_csv(file)
        df['ref'][-1]=df['ref'].max()+1
        add_ref=df['ref'].max()+1
        if button_press:
            new_data={'ref':add_ref, 'unique_id':Unique_id, 'Patient_name':patient_name, 'Professional_name':professional_name, 'Sex':sex, 'Diagnosis':Diagnosis, 'fee':Fee_paid}
            df=df.append(new_data, ignore_index=True)
            df.to_csv(file, index=False)
    if option=="Get information related to a patient":
        unique_id=st.text_input("Enter unique Id whose details are to be load")
        button_press=st.button(label="Enter")
        Patient='Patient.csv'
        df1=get_table('df1', Patient)
        df1=pd.read_csv(Patient)
        df1['ref'][-1]=df1['ref'].max()+1
        add_ref=df1['ref'].max()+1
        if button_press:
            new_data={'ref':add_ref, 'unique_id':unique_id}
            df1=df1.append(new_data, ignore_index=True)
            df1.to_csv(Patient, index=False)
            file1='file.csv'
            df2=get_table('df2', file1)
            df2=pd.read_csv(file1)
            df2.loc[df2["unique_id"]==unique_id]
    st.markdown("""<h3 style="color:Yellow;">Cancer Stories</h3>""", unsafe_allow_html=True)
    col34, col35, col36=st.columns(3)
    col34.image("Cancer.png")
    col35.image("canc.png")
    col36.image("cancerr.png")
    st.markdown("""<h4 style="color:Yellow;">Add your story</h4>""", unsafe_allow_html=True)
    user_input=st.text_area("", "Enter Your Story")
    # if user_input!="Enter Your Story":
    #     st.write(user_input)
    st.dataframe(data=data.head(20), height=200)
    scipy_col, pyspark_col = create_subcol()
    create_sidelayout(scipy_col, pyspark_col)
    plot()
    st.subheader(
        'Streamlit Healthcare By '
        '[Ishika Bhola + Sahil Agarwal](https://www.linkedin.com/in/ishika-bhola/)')

if __name__ == '__main__':
    main()
