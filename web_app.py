import pandas as pd
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import product
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest 
from collections import Counter
from sklearn.feature_selection import mutual_info_regression, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from streamlit_option_menu import option_menu
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import make_column_selector as selector
import pickle


df = pd.read_csv("housing_dataset.csv")
df.drop(columns =["Id"])
numerical_feature_df= pd.read_csv("Selected_numerical_features.csv")
categorical_feature_df = pd.read_csv("Selected_categorical_features.csv")
categorical_feature_df= categorical_feature_df[["Feature","Frequence"]]

st.set_page_config(
    page_title="machine_learning",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)


## Data Cleaning
def remove_outliers_iqr(df_arg, k):
    df_iqr = df_arg.copy()
    #columns = [col for col in df_arg.columns]
    for col in df_arg.columns:
        if df_arg[col].dtypes!="object":
            q25, q75 = q25, q75 = df_iqr[col].quantile(.25), df_iqr[col].quantile(.75)
            ecart_iqr = q75- q25
            cut_off = ecart_iqr*k
            lower = q25-cut_off
            upper = q75+cut_off
            df_iqr[col] =np.where(((df_iqr[col]< lower)|(df_iqr[col]> upper)),df_arg[col].median(),df_iqr[col])
    return df_iqr 

def plot_view(x_col,k,numerical_df= df.select_dtypes(include=["int64"]),target = "SalePrice"):
    df_res = remove_outliers_iqr(numerical_df.copy(),k)
    fig, ax= plt.subplots()
    numerical_df.plot.scatter(x=x_col, y = target,ax = ax, color ="red")
    df_res.plot.scatter(x=x_col, y = target, ax = ax, color = "green")
    return fig

def plot_validation(plot_type,x_col,k,numerical_df= df.select_dtypes(include=["int64"]),target = "SalePrice"):
    df_res = remove_outliers_iqr(numerical_df.copy(),k)
    if plot_type =="Scatter":
        fig, ax= plt.subplots()
        numerical_df.plot.scatter(x=x_col, y = target,ax = ax, color ="red")
        df_res.plot.scatter(x=x_col, y = target, ax = ax, color = "green")
    if plot_type=="Box":
        fig,ax = plt.subplots(figsize  =(20,20))
        sns.boxplot(df_res,x=x_col, y=target)
    if plot_type =="Line":
        fig,ax = plt.subplots(figsize =(20,20))
        sns.regplot(x = x_col,y=target, data=df_res, scatter=True,fit_reg=True)
    return fig
    
def plot_distribution(x_col,k,numerical_df= df.select_dtypes(include=["int64"])):
    df_res = remove_outliers_iqr(numerical_df.copy(),k)
    fig,ax = plt.subplots(figsize =(20,20))
    sns.distplot(df_res[x_col])
    return fig

        
    
## feature selection technique
@st.cache_data
def corr_feature_selector(threshold, target = "SalePrice",corr =df.select_dtypes(include=["int64"]).corr()):
    target_corr= corr[corr[target].abs()>=threshold][target]
    return target_corr, target_corr.axes[0].to_list()

@st.cache_data
def rf_features_selector(top_n,df_arg=df.select_dtypes(include=["int64"]), target_name ="SalePrice"):
    seed = np.random.seed(10)
    #df_w = df_arg.copy()
    features = [col for col in df_arg.columns if col!=target_name]
    X = df_arg.copy()[features]
    y = df_arg[target_name].values
    model = RandomForestRegressor(random_state = seed)
    model.fit(X,y)
    #get feaures importance
    importance = model.feature_importances_
    indices = np.argsort(importance)
    feat_importances = pd.Series(importance, index=X.columns)
    rf_df =feat_importances.nlargest(top_n)
    plt.xlabel('importance')
    rf_features = pd.DataFrame(feat_importances.nlargest(top_n)).axes[0].tolist()
    return rf_df, rf_features

@st.cache_data
def lassoReg_feat_selector(df_arg=df.select_dtypes(include=["int64"]),target_name ="SalePrice"):
    np.random.seed(10)
    features = [col for col in df_arg.columns if col!=target_name]
    X = df_arg.copy()[features]
    y = df_arg[target_name].values
    estimator = LassoCV(cv=5)
    sfm = SelectFromModel(estimator, prefit=False, norm_order=1, max_features=None)
    sfm.fit(X,y)
    feature_idx = sfm.get_support()
    Lasso_features = X.columns[feature_idx].tolist()
    return Lasso_features
@st.cache_data
def rfe_feature_selector(df_arg=df.select_dtypes(include=["int64"]),target_name ="SalePrice"):
    np.random.seed(10)
    features = [col for col in df_arg.columns if col!=target_name]
    X = df_arg.copy()[features]
    y = df_arg[target_name].values
    rfe= RFE(estimator=LinearRegression(),n_features_to_select=15)
    rfe.fit(X,y)
    rfe_support = rfe.get_support()
    rfe_feat = X.loc[:,rfe_support].columns.to_list()
    return rfe_feat
@st.cache_data
def mif_feature_selector(df_arg=df.select_dtypes(include=["int64"]),target_name ="SalePrice"):
    np.random.seed(10)
    features = [col for col in df_arg.columns if col!=target_name]
    X = df_arg.copy()[features]
    y = df_arg[target_name].values
    mif= SelectKBest(score_func=mutual_info_regression,k=15)
    mif.fit(X, y)
    mif_support = mif.get_support()
    mif_feat = X.loc[:,mif_support].columns.to_list()
    return mif_feat
@st.cache_data    
def feature_selector(num_threshold, categ_threshold,num_feat_df=numerical_feature_df, categ_feat_df=categorical_feature_df):
    numerical_feature_list = numerical_feature_df[numerical_feature_df["Frequence"]>num_threshold].Feature.values
    categorical_feature_list = categorical_feature_df[categorical_feature_df["Frequence"]>categ_threshold].Feature.values
    final_feature = list(numerical_feature_list)+list(categorical_feature_list )
    return final_feature

##########################################################################################
# Modelisation
##########################################################################################

@st.cache_data 
def preprocess_data(df_arg,features, train_size=0.8):
    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
    numerical_preprocessor = MinMaxScaler()
    x_inputs = df_arg.copy().drop(columns=["SalePrice"])[features]
    numerical_columns_selector = selector(dtype_exclude=object)
    categorical_columns_selector = selector(dtype_include=object)
    numerical_columns = numerical_columns_selector(x_inputs)
    categorical_columns = categorical_columns_selector(x_inputs)
    preprocessor = ColumnTransformer(transformers=[
    ('one-hot-encoder', categorical_preprocessor,categorical_columns),
    ('min_max_scaler', numerical_preprocessor, numerical_columns)]) 
    X_processed= preprocessor.fit_transform(x_inputs)
    #y_processed = (df_arg.copy()["SalePrice"]-df_arg.copy()["SalePrice"].min())/((df_arg.copy()["SalePrice"].max()-df_arg.copy()["SalePrice"].min()))
    y_processed = df_arg.copy()["SalePrice"]
    trainX, testX, trainY, testY= train_test_split(X_processed, y_processed, train_size=train_size)
    #trainX.shape, testY.shape
    return trainX, testX, trainY,testY,X_processed,y_processed
@st.cache_data
def predict_house_price(user_input_data, _predictor):
    categorical_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value =1)
    numerical_preprocessor = MinMaxScaler()
    numerical_columns_selector = selector(dtype_exclude=object)
    categorical_columns_selector = selector(dtype_include=object)
    numerical_columns = numerical_columns_selector(user_input_data)
    categorical_columns = categorical_columns_selector(user_input_data)
    preprocessor = ColumnTransformer(transformers=[
    ('ordinalEncoder', categorical_preprocessor,categorical_columns),
    ('min_max_scaler', numerical_preprocessor, numerical_columns)])
    X_processed= preprocessor.fit_transform(user_input_data)
    #print(X_processed.shape)
    result = _predictor.predict(X_processed)
    return result
    


@st.cache_data 
def get_metrics(y_true, y_pred,precision=5):
    metrics_dict  = {"MSE": 0,"MAE":0, "RMSE":0,"MAPE":0}
    mse = np.mean(np.power(y_true - y_pred,2))
    metrics_dict["MSE"] = np.round(mse,precision)

    mae = np.mean(np.abs(y_true - y_pred))
    metrics_dict["MAE"] = np.round(mae,precision)
    rmse = np.sqrt(np.mean(np.power(y_true - y_pred,2)))
    metrics_dict["RMSE"] = np.round(rmse,precision)
    mape = 100*np.abs(((y_true - y_pred)/np.abs(y_true))).sum() /len(y_true)
    metrics_dict["MAPE"] = np.round(mape,precision)
    return metrics_dict
@st.cache_data 
def fit_models(_model,df_arg,feature):
    train_test_metrics = {}
    cross_val_dict={}
    trainX, testX, trainY, testY, X, y= preprocess_data(df_arg,feature)
    model_fit = _model.fit(trainX,trainY)
    y_pred =model_fit.predict(testX)
    train_pred = model_fit.predict(trainX)
    train_test_metrics["Train"]=get_metrics(trainY,train_pred)
    train_test_metrics["Test"]=get_metrics(testY,y_pred)
    cv_score =cross_val_score(_model,X,y, cv=10)
    cross_val_dict["mean"] =np.mean(cv_score)
    cross_val_dict["std"] = np.std(cv_score)
    return y_pred,train_pred,pd.DataFrame.from_dict(train_test_metrics),cross_val_dict
@st.cache_data   
def display_model_res(_model, df_arg, feature):
    y_pred, train_pred, metrics_data, cross_val_score= fit_models(_model,df_arg,feature)
    cv_col, metrics_col=st.columns(2)
    cv_col.metric("number of feature",value =f"{len(feature)}")
    #cv_col.subheader("Cross validation")
    cv_col.metric("Coss validation score", value = f'{round(cross_val_score["mean"],5)} +/-{round(cross_val_score["std"],5)}')
    metrics_col.subheader("Evaluation metrics")
    metrics_col.dataframe(metrics_data)

st.title("House Price Prediction Tool")

selected_page = option_menu(
    "",
    options=["Home", "EDA", "Feature Engineering", "Modelisation"],
    menu_icon="cast",
    icons=["house-fill", "database-add", "graph-up", "calculator-fill"],
    orientation="horizontal",
)

if selected_page=="Home":
    top_n = st.sidebar.slider("nrows",5,100,5,5)
    current_model = st.sidebar.selectbox("Select model to use",options=["RandomForestRegressor","GradientBoostingRegressor","HistGradientBoostingRegressor","ExtraTreesRegressor","SVR"])
    saved_model = {"RandomForestRegressor":open("models/randomForest.pkl","rb"), "GradientBoostingRegressor":open("models/gradientboosting.pkl","rb"),"HistGradientBoostingRegressor":open("models/histgradboosting.pkl","rb"),"ExtraTreesRegressor":open("models/extratreesreg.pkl","rb"),"SVR":open("models/svr.pkl","rb")}
    st.sidebar.subheader("Select your criteria")
    predictor = pickle.load(saved_model[current_model])
    user_inp = pd.DataFrame({"'KitchenQual":[st.sidebar.selectbox("KitchenQual",np.unique(df["KitchenQual"]))],"ExterQual":[st.sidebar.selectbox("ExterQual",np.unique(df["KitchenQual"]))],"TotalBsmtSF":[st.sidebar.slider("TotalBsmtSF",float(df["TotalBsmtSF"].min()),float(df["TotalBsmtSF"].max()))],
        "GrLivArea":[st.sidebar.slider("GrLivArea",float(df["GrLivArea"].min()),float(df["GrLivArea"].max()))]})
    user_inp_col, price_col = st.columns(2)
    user_inp_col.subheader("features value")
    user_inp_col.dataframe(user_inp)
    house_price = predict_house_price(user_inp,predictor)
    price_col.subheader("Prediction result")
    price_col.metric(":house_buildings:",f"{np.mean([round(x,2) for x in house_price])} 💲")
    #price_col.metric("Real Price",f"{np.mean([round(x,2) for x in real_price])} $")
    st.subheader("Overview on data")
    st.dataframe(df.head(top_n),use_container_width=True)


if selected_page=="EDA":
    with st.sidebar:
        k= st.slider("Select k to detect outliers",0.0,5.0,0.5)
        eda_plot = st.selectbox("Plot type",options = ["Box","Scatter","Line"])
        x_col = st.selectbox("X",options =df.select_dtypes(include =["int64"]).drop(columns=["Id"], axis=1).columns)
    distrib_col, val_col = st.columns(2)
    val_col.subheader("Validation result")
    val_col.pyplot(plot_validation(eda_plot,x_col,k))
    distrib_col.subheader("Distribution")
    distrib_col.pyplot(plot_distribution(x_col,k))

if selected_page=="Feature Engineering":
    with st.sidebar:
        data_or_plot = st.radio("View as",options=["Data","Plots"])
        feature_selection_technique = st.selectbox("Feature selector",options = ["Correlation","RandomForestReg","Recrusive","Mitual Information Reg","LassoReg"])
        top_n_rf = st.slider("Top n Random Forest Features",1,20,3,1)
        corr_threshold = st.slider("Correlation threshold",0.0,1.0,0.1)
        num_feat_freq = st.slider("Numerical feature selection frequence",1,6,1)
        st.subheader("Select final features")
        num_th=  st.slider("numerical feature",2,5,2,1)
        categ_th=  st.slider("categorical feature",2,6,2,1)
    # Appel des fonction de selection
    target_corr, corr_feat_list =corr_feature_selector(corr_threshold)
    rf_feat_importance, rf_feat_list = rf_features_selector(top_n_rf)
    lasso_feat_list = lassoReg_feat_selector()
    rfe_feat_list = rfe_feature_selector()
    mif_feat_list = mif_feature_selector()

    combined_feat_list  =rfe_feat_list+lasso_feat_list+rfe_feat_list+corr_feat_list+mif_feat_list
    feat_freq= Counter(combined_feat_list)
    feat_freq_df = pd.DataFrame({"Feature":feat_freq.keys(),"Frequence":feat_freq.values()})
    #feat_freq_df.sort_values("Frequence", ascending =False)
    
    feat_freq_df = feat_freq_df[feat_freq_df.Feature!="SalePrice"]
    final_feat_freq_df = feat_freq_df[feat_freq_df.Frequence>=num_feat_freq]
    feat_selection_data = {"Correlation":corr_feat_list,"RandomForestReg":rf_feat_list,"Recrusive":rfe_feat_list,"Mitual Information Reg":mif_feat_list,"LassoReg":lasso_feat_list}
    if data_or_plot=="Data":
        feat_sel_col, feat_df_col = st.columns(2)
        feat_sel_col.subheader(f"Seleced Features with {feature_selection_technique}")
        feat_selector_result = feat_sel_col.dataframe(feat_selection_data[feature_selection_technique], use_container_width=True)
        feat_df_col.subheader("Final numerical selected features")
        feat_df_col.dataframe(final_feat_freq_df.sort_values("Frequence", ascending=False), use_container_width=True)
        feat_df_col.subheader("Final mixed features data")
        feat_df_col.dataframe(pd.concat([numerical_feature_df[numerical_feature_df.Frequence>=num_th],categorical_feature_df[categorical_feature_df.Frequence>=categ_th]]),use_container_width=True)
        feature = feature_selector(num_th,categ_th)
        feat_sel_col.subheader("Final mixed features")
        feat_sel_col.dataframe(feature, use_container_width=True)
    else:
       corr= df.select_dtypes(include=["int64"]).corr()
       corr_col, rf_plot_col = st.columns(2)

       corr_col.subheader("Correlation matrix")
       fig, ax = plt.subplots()
       plt.figure(figsize= (20,10))
       plt.title('Correlation')
       sns.heatmap(corr[corr>corr_threshold],vmin = -1, cmap = "coolwarm",vmax = 1, annot = True, ax = ax) 
       corr_col.pyplot(fig)

       rf_plot_col.subheader("RF selected features")
       fig, ax1= plt.subplots()
       rf_feat_importance.plot.barh(ax=ax1)
       rf_plot_col.pyplot(fig)
       
       fig, ax2= plt.subplots()
       corr_col.subheader("Numerical")
       final_feat_freq_df.sort_values("Frequence", ascending=False).plot.barh(x = "Feature",ax =ax2)
       corr_col.pyplot(fig)

       fig, ax3= plt.subplots()
       rf_plot_col.subheader("Both")
       pd.concat([numerical_feature_df[numerical_feature_df.Frequence>num_th],categorical_feature_df[categorical_feature_df.Frequence>categ_th]]).sort_values("Frequence", ascending=False).plot.barh(x = "Feature",ax =ax3)
       rf_plot_col.pyplot(fig)


if selected_page=="Modelisation":
    with st.sidebar:
        feat_type = st.radio("Feature type", options =["Numerical","Both"])
        st.subheader("Select final features")
        model_type= st.selectbox("Select a model",options=["RadomForestReg","GradientBoostReg","HistGradienBoot","ExtraTreesReg","SVR"])
        models_dict=  {"RadomForestReg":RandomForestRegressor(),"GradientBoostReg":GradientBoostingRegressor(),"HistGradienBoot":HistGradientBoostingRegressor(),"ExtraTreesReg":ExtraTreesRegressor(), "SVR":svm.SVR()}
        if feat_type=="Numerical":
            num_th=  st.slider("numerical feature",2,5,2,1)
            feature = numerical_feature_df[numerical_feature_df.Frequence>num_th].Feature.values
        if feat_type=="Both":
            both_th=  st.slider("Mixed features",2,6,2,1)
            feature= pd.concat([numerical_feature_df[numerical_feature_df.Frequence>both_th],categorical_feature_df[categorical_feature_df.Frequence>both_th]]).Feature.values

    df_work=remove_outliers_iqr(df,3)
    st.header("Evaluation of "+model_type)
    display_model_res(models_dict[model_type],df_work,feature)






