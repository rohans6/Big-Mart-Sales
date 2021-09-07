# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
sns.set_style("darkgrid")
plt.style.use("fivethirtyeight")
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
import streamlit as st
from PIL import Image
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import mode
from sklearn.model_selection import KFold
from pickle import load
# App will have 4 pages:- 1.Introduction,2.EDA,3.Feature Engineering 4. Model Building 5. Make Prediction

# Main Layout of App
st.title("Web App to predict Big-Mart Sales")
image=Image.open("image.jpg")

st.image(image,use_column_width=True)

st.subheader("The aim is to build a predictive model and predict the sales of each product at a particular outlet")
# Loading files
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
num_cols=[col for col in train.columns if train[col].dtype!=object ]
cat_cols=[col for col in train.columns if train[col].dtype==object]


def Feature_Engineering(data):
    dataframe=data.copy()
    dataframe['New_Item_Type']=dataframe['Item_Identifier'].apply(lambda x: x[:2])
    dataframe['New_Item_Type']=dataframe['New_Item_Type'].map({"FD":"Food","NC":"Non Consumable","DR":"Drinks"})
    dataframe['Years_Established']=2021-dataframe['Outlet_Establishment_Year']
    dataframe.loc[dataframe['New_Item_Type']=="Non Consumable","Item_Fat_Content"]="Non-Edible"
    return dataframe

def preprocessing():

	train=pd.read_csv("train.csv")
	test=pd.read_csv("test.csv")
	train['file']='train'
	test['file']='test'
	all_data=pd.concat([train,test],ignore_index=True)
	# Missing Data Imputation
	item_avg_weight=all_data.pivot_table(values="Item_Weight",index="Item_Identifier")
	all_data.loc[all_data['Item_Weight'].isnull(),"Item_Weight"]=all_data.loc[all_data['Item_Weight'].isnull(),"Item_Identifier"].apply(lambda x: item_avg_weight.loc[item_avg_weight.index==x,"Item_Weight"][0])
	a=all_data.pivot_table(values="Outlet_Size",index= 'Outlet_Type',aggfunc=lambda x: mode(x).mode[0] )
	all_data.loc[all_data['Outlet_Size'].isnull(),"Outlet_Size"]=all_data.loc[all_data['Outlet_Size'].isnull(),"Outlet_Type"].apply(lambda x: a.loc[a.index==x,"Outlet_Size"][0])
	# Data Cleaning

	item_avg_visibility=all_data.pivot_table(values="Item_Visibility",index="Item_Identifier")
	all_data.loc[all_data['Item_Visibility']==0,"Item_Visibility"]=all_data.loc[all_data['Item_Visibility']==0,"Item_Identifier"].apply(lambda x: item_avg_visibility.loc[item_avg_visibility.index==x,"Item_Visibility"][0])
	# Preparing data for Modelling
	train=all_data[all_data['file']=='train']
	test=all_data[all_data['file']=='test']
	train=Feature_Engineering(train)
	test=Feature_Engineering(test)
	y=train['Item_Outlet_Sales'].copy()
	train.drop(['Item_Outlet_Sales','file'],axis=1,inplace=True)
	test.drop(['Item_Outlet_Sales','file'],axis=1,inplace=True)
	nc=[col for col in train.columns if train[col].dtype!=object]
	cc=[col for col in train.columns if train[col].dtype==object]
	preprocessing_pipeline=ColumnTransformer([("Numerical",RobustScaler(),nc),("Categorical",OneHotEncoder(handle_unknown="ignore"),cc)])
	return train,test,preprocessing_pipeline,y
def Train_Models(name,n_est,max_depth,lr,train,test,preprocessing_pipeline,y):
	

	val_predictions=np.zeros((len(test),1))
	train_val_predictions=np.zeros((len(train),1))
	kfold=KFold(n_splits=5,shuffle=True,random_state=42)

	for train_index,val_index in kfold.split(train,y):

		train_data=preprocessing_pipeline.fit_transform(train.loc[train_index])
		val_data=preprocessing_pipeline.transform(train.loc[val_index])
		test_data=preprocessing_pipeline.transform(test)
		y_train=y[train_index]
		if name=='RF':

			model=RandomForestRegressor(n_estimators=n_est,max_depth=max_depth,min_samples_split=3,min_impurity_decrease=5)
		else:

			model=LGBMRegressor(n_estimators=n_est,max_depth=max_depth,learning_rate=lr,subsample=0.8)
    
		model.fit(train_data,y_train)
    
		p=model.predict(val_data)
		train_val_predictions[val_index]+=np.reshape(p,(len(p),1))
		p=model.predict(test_data)
		val_predictions+=np.reshape(p,(len(p),1))/5
	return train_val_predictions,val_predictions
	
		


def main():
	activites=['Introduction','EDA','Feature Engineering','Model Building','Prediction']
	menu=st.sidebar.selectbox("Select Project Phase",activites)
	train=pd.read_csv("train.csv")
	test=pd.read_csv("test.csv")
	num_cols=[col for col in train.columns if train[col].dtype!=object ]
	cat_cols=[col for col in train.columns if train[col].dtype==object and col!='Item_Identifier']
	

	if menu=='Introduction':
		st.subheader("What is BigMart")
		st.markdown("Big Mart Retail is a grocery supermarket brand, and it is widely known for its home delivery services of food and grocery. It enables the customer to walk away from the drudgery of grocery shopping and shop in a relaxed way of browsing from their home or office.")
		st.text("="*100)
		st.markdown("The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined.")
		st.markdown("We have train (8523) and test (5681) data set, train data set has both input and output variable(s).")
		st.text("="*100)
		st.subheader("Top 10 Training Records:-")
		st.dataframe(train.head(10))
		st.text("="*100)
		st.subheader("Features:-")
		st.text("Item Identifier: Unique Product Id")
		st.text("Item_Weight: Weight of Product")
		st.text("Item_Fat_Content: Fat Content of Product -Low/Regular/")
		st.text("Item_Visibility: The % of total display area of all products in a store allocated to the particular product")
		st.text("item_Type: Category of Product")
		st.text("Item_MRP: Maximum Retail Price of the Product")
		st.text("Outlet_Identifier: Store ID")
		st.text("Outlet_Establishment_Year: The Year in which store is established")
		st.text("Outlet_Size: Areawise distribution of Stores- Low/Medium/High")
		st.text("Outlet_Location_Type: Type of city in which outlet is located")
		st.text("Outlet_Type: Type of outlet - Grocery store or supermarket")
		st.text("Item_Outlet_Sales: Sale price of product - The dependant variable to be predicted")
		st.text("="*100)
		st.subheader("Some Assumptions :-")
		st.text("1. Item_Weight will follow a Gaussian Distribution because the grocery  products will not have very high weights. Also we Item_mrp will follow a right-skew distribution because some grocery products will be costly.")
		st.text("2. Many Stores will have same products and pricing of products should remain same throughout all stores on which it is available.")
		st.text("3. Item_Visibiity of same products across different stores will differ")
		st.text("4. Generally now in 2021 people are more consicous about their diet so we except low quality fat products to be sold more then  high quality fat and also prices of low quality products will be more then high fat products.")
		st.text("5. Sales of a product will depend on Outlet Location like stores which are on outskirts of a city will have low product sales as compared to a Store which is in centre of city")
		st.text("6. The Combinations of Product and Shop must not repeat.")
		st.text("="*100)






		st.subheader("Evaluation Metric for Test Dataset :-")
		st.markdown("RMSE: Root Mean Squared Error")
		st.text("="*100)
		st.subheader("Project Structure :-")
		st.markdown("1. EDA ")
		st.markdown("2. Feature Engineering ")
		st.markdown("3. Predictive Modelling")
		st.markdown("4. Making Predictions")
	if menu=='EDA':
		st.header("EDA")
		if st.checkbox("Numerical Columns Summary"):
			st.write(train.describe())
		if st.checkbox("Display Missing Values :"):
			st.write(train.isnull().sum())
		st.text("="*100)
		st.write("No of Unique product Ids =",1559)
		st.write("No of total Outlets =",10)
		st.write("Total records (train and test) =",15590)
		st.write("Hence the assumption that there will be Unique product-outlet pairs is correct")
		st.text("="*100)
		# Analysing Individual Numerical Columns:-

		st.subheader("Analysing Individual Numerical Columns :-")
		column=st.selectbox("Select Column",num_cols)
		st.write("Skew =",train[column].skew())
		st.write("Variance =",train[column].var())
		st.write("Missing Values =",train[column].isnull().sum())
		st.write("Mean =",train[column].mean())
		st.write("Median =",train[column].median())
		st.write("Min =",train[column].min())
		st.write("Max =",train[column].max())
		st.write("Histogram")
		fig,ax=plt.subplots()
		sns.histplot(train[column],ax=ax)
		st.pyplot(fig)
		st.write("QQ-plot")
		fig,ax=plt.subplots()
		qqplot(train[column].dropna(),line='s',ax=ax)
		st.pyplot(fig)
		st.write("Box-Plots")
		fig,ax=plt.subplots()
		sns.boxplot(train[column],ax=ax)
		st.pyplot(fig)
		st.text("="*100)

		# Analysing Individual Categorical Columns

		st.subheader("Analysing Individual Categorical Columns :-")
		column=st.selectbox("Select Column",cat_cols)
		st.write("Value Counts:-",train[column].value_counts())
		st.write("Missing Values =",train[column].isnull().sum())
		st.write("Count-Plots")
		fig,ax=plt.subplots(figsize=(20,15))
		sns.countplot(x=column,data=train,ax=ax)
		ax.set_xlabel(column,fontdict={"fontsize":40})
		ax.set_ylabel(column,fontdict={"fontsize":40})
		ax.set_xticklabels(ax.get_xticklabels(),fontdict={"fontsize":25})
		st.pyplot(fig)
		st.text("="*100)

		# BiVariate Analysis
		st.header("Bi-Variate Analysis")

		st.subheader("HeatMap")
		fig,ax=plt.subplots()
		sns.heatmap(train.corr(),annot=True,cmap='viridis',ax=ax)
		st.pyplot(fig)
		st.text("="*100)

		# Hexagonal Binning plots
		st.subheader("Hexagonal Binning Plots")
		selected_columns=st.multiselect("Select 2 Column",num_cols)
		if len(selected_columns)==2:
			fig=sns.jointplot(train[selected_columns[0]],train[selected_columns[1]],kind='hex',color='#ff6666')
			st.pyplot(fig)
		else:
			st.error("Select 2 Columns")
		st.text("="*100)

		
		
		

		# Violin Plots
		st.subheader("Violin Plots")
		cat_col=st.selectbox("Select Categorical Col",cat_cols)
		num_col=st.selectbox("Select Numerical Col",num_cols)
		fig,ax=plt.subplots()
		sns.violinplot(cat_col,num_col,ax=ax,data=train)
		st.pyplot(fig)
		st.text("="*100)


		# Crosstabs of Categorical Columns
		if st.checkbox("Categorical Columns :"):
			selected_columns=st.multiselect("Select 2 Columns to display Summary: ",cat_cols)
			if len(selected_columns)==2:
				summary=pd.crosstab(train[selected_columns[0]],train[selected_columns[1]])
				st.write(summary)
			else:
				st.error("Select 2 Columns")


	if menu=='Feature Engineering':
		st.header("Feature Engineering")
		st.subheader("Missing Values Imputation")
		st.markdown("In EDA we have seen there are NAN values in Item_Weight,Outlet_Size.")
		st.markdown("The Item_Weight will be different for different items,so we calculate avg item weight for all items and impute with corresponding item weight for that item for e.g if we recieved a data point with item_weight as NAN and item_identifier=='FD001' then we will impute it with mean item_weight of all points in train dataset having item_indentifier=='FD001'.")
		st.markdown("Similarly as Item_Weight, Outlet_size will be different for different Outlet_Type so same as Item_weight but here we will use mode")
		st.subheader("Data Cleaning")
		st.markdown("As observed from histogram of Item_Visibiity alot many values are zero which does not make sense")
		st.markdown("We will calculate avg item Item_Visibiity for all items, so for all such points having item_visibility zero we will see what is avg Item_Visibiity for that item and replace that zero with these mean_item_visibility")
		st.markdown("Item_Fat_Content has 5 Unique values where broadly 3 categories can be merged into rest 2")
		st.subheader("Feature Extraction")
		st.markdown("Item_Identifer's first 2 intials stand for following FD-Food,NC- Non Consumable, DR: Drinks. So we can create new feature extracting these")
		st.markdown("We can only extract no of years established for a store by 2021-year of establishment.")
		st.subheader("Preparing data for Modelling")
		st.markdown("We will use a RobustScaler to scale numerical columns and OneHotEncoder for Categorical columns")
		

	if menu=='Model Building':
		# Data Preprocessing 
		train,test,preprocessing_pipeline,y=preprocessing()
		# LGBMRegressor
		st.header("LightGBM-Regressor")
		st.markdown("Choose Parameters")
		n_est=st.slider("Select Number of Estimators",50,1000,10)
		lr=st.slider("Select Learning Rate ",0.01,0.9,0.01)
		max_depth=st.slider("Select Max depth",3,15,1)
		# Training LGBMRegressor

		lgm_val_predictions,lgm_predictions=Train_Models("LGM",n_est,max_depth,lr,train,test,preprocessing_pipeline,y)
		rmse=np.sqrt(mean_squared_error(y,lgm_val_predictions))
		r2=r2_score(y,lgm_val_predictions)
		st.write("RMSE of LGBMRegressor =",rmse)
		st.write("R2 Score of LGBMRegressor =",r2)

		#RandomForestRegressor
		st.header("Random RandomForestRegressor")
		st.markdown("Choose Parameters")
		n_est=st.slider("Select No of Estimators",50,1000,10)
		max_depth=st.slider("Select Max Depth",3,15,1)

		rf_val_predictions,rf_predictions=Train_Models("RF",n_est,max_depth,lr,train,test,preprocessing_pipeline,y)
		rmse=np.sqrt(mean_squared_error(y,rf_val_predictions))
		r2=r2_score(y,rf_val_predictions)
		st.write("RMSE of LGBMRegressor =",rmse)
		st.write("R2 Score of LGBMRegressor =",r2)

		# Ensemble 
		st.subheader("Ensemble of LGBMRegressor and RandomForestRegressor")
		weights=np.arange(0.01,1,0.01)

		min_error=10000
		for w in weights:


			p=w*rf_val_predictions+(1-w)*lgm_val_predictions
			e=np.sqrt(mean_squared_error(y,p))
			if e<min_error:



				min_error=e
				best_weight=w
		st.write("Best Weight =",best_weight)
		p=best_weight*rf_val_predictions+(1-best_weight)*lgm_val_predictions
		rmse=np.sqrt(mean_squared_error(y,p))
		r2=r2_score(y,p)
		st.write("RMSE of EnsembleRegressor =",rmse)
		st.write("R2 Score of EnsembleRegressor =",r2)
		training=True
		st.write(training)
	if menu=='Prediction':


		
		

		st.subheader("Prediction")
		train=pd.read_csv("train.csv")
		test=pd.read_csv("test.csv")
		all_data=pd.concat([train,test],ignore_index=True)
		df=pd.DataFrame()
		item_identifier=st.text_input("Enter Item Identifier ")
		item_weight=st.number_input("Enter Item Weight in grams ")
		item_fat_content=st.selectbox("Select Item_Fat_Content ",list(all_data['Item_Fat_Content'].unique())+['Non-Edible'])
		item_visibility=st.number_input("Enter Item Visibility")
		item_type=st.selectbox("Enter Item Type ",list(all_data['Item_Type'].unique()))
		item_mrp=st.number_input("Enter MRP Price of Item")
		outlet_size=st.selectbox("Enter Outlet Size",list(all_data['Outlet_Size'].unique()))
		outlet_identifier=st.selectbox("Enter Outlet Identifier",list(all_data['Outlet_Identifier'].unique()))
		outlet_est_year=st.selectbox("Enter Outlet Establishment Year",list(all_data['Outlet_Establishment_Year'].unique()))
		outlet_location=st.selectbox("Enter Outlet_Location_Type",list(all_data['Outlet_Location_Type'].unique()))
		year_established=2021-outlet_est_year
		outlet_type=st.selectbox("Enter Outlet_Type",list(all_data['Outlet_Type'].unique()))
		df=pd.DataFrame(columns=['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
       'Item_Type', 'Item_MRP', 'Outlet_Identifier',
       'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type',
       'Outlet_Type'])
		df.loc[0]=[item_identifier,item_weight,item_fat_content,item_visibility,item_type,item_mrp,outlet_identifier,outlet_est_year,outlet_size,outlet_location,outlet_type]
		
		df['New_Item_Type']=df['Item_Identifier'].apply(lambda x: x[:2])
		df['New_Item_Type']=df['New_Item_Type'].map({"FD":"Food","NC":"Non Consumable","DR":"Drinks"})
		df['Years_Established']=year_established
		st.write(item_identifier)
		st.dataframe(df)
		gbm_prediction=0
		model_path="D://Data Science//Machine Learning//Projects//Big Mart Sales Prediction//Saved Models//"
		preprocessing_path="D://Data Science//Machine Learning//Projects//Big Mart Sales Prediction//Saved Preprocessing Pipeline//"
		for i in range(5):
			file=model_path+'LGBM_fold_'+str(i)+".pkl"
			with open(file,'rb') as f:
				model=load(f)
			file=preprocessing_path+'LGBM_Preprocessing'+str(i)+".pkl"
			with open(file,'rb') as f:
				preprocessing_pipe=load(f)
			x_test=preprocessing_pipe.transform(df)
			gbm_prediction+=model.predict(x_test)
		gbm_prediction=gbm_prediction/5
		rf_prediction=0
		for i in range(5):
			file=model_path+'RF_fold_'+str(i)+".pkl"
			with open(file,'rb') as f:
				model=load(f)
			file=preprocessing_path+'RF_Preprocessing'+str(i)+".pkl"
			with open(file,'rb') as f:
				preprocessing_pipe=load(f)
			x_test=preprocessing_pipe.transform(df)
			rf_prediction+=model.predict(x_test)
		rf_prediction=rf_prediction/5
		prediction=0.29*rf_prediction+0.71*gbm_prediction
		if item_weight==0 :
			st.error("Weight of Item Cannot be Zero. Please Enter a Valid Item Weight")
		elif item_visibility==0:
			st.error("Visibility of Item at a Store cannot be zero. Please Enter a Valid Item Visibility")
		elif item_mrp==0:
			st.error("Price of Item cannot be zero. Please Enter a Valid Item MRP Price")
		elif item_identifier=="":
			st.error("Please Enter a Item Identifier")
		else:


			st.write("Average Sale for specified product at specified outlet =",np.round(prediction[0],2))
		













		



		

		

		




				




			

			

			


	   

if __name__ == '__main__':
	main()

