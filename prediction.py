import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
# from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRegressor
import xgboost

from abc import ABC, abstractmethod
import copy
import io


class DataLoader:
	def __init__(self, file_path):
		"""
		Initializes the environment for loading and processing the CSV file.

		Parameters:
		- file_path (str): The path to the CSV file to load.
		"""
		self.file_path = file_path
		self.df = None

		print(f"Environment initialized. File to load: {file_path}")

	def load_csv(self):
		"""
		Loads a CSV file and stores it as a DataFrame.

		Returns:
		- DataFrame: The loaded DataFrame.
		"""
		try:
			self.df = pd.read_csv(self.file_path)
			print(f"File loaded successfully from {self.file_path}")
			return self.df
		except FileNotFoundError:
			print(f"Error: The file at {self.file_path} was not found.")
			return None
		except pd.errors.EmptyDataError:
			print(f"Error: The file at {self.file_path} is empty.")
			return None
		except Exception as e:
			print(f"Error: {e}")
			return None

class ConfigManager:
	_instance = None  # Store the single instance

	def __new__(cls):
		if cls._instance is None:
			cls._instance = super().__new__(cls)
			# Initialize configuration only once
			cls._instance.settings = {
				"dataset_path": "data/train.csv",
				"testset_path": "data/test.csv", # unlabeled
				"learning_rate": 0.01,
				"epochs": 50
			}

			# set pd settings here
			pd.set_option('display.max_columns', None)

		return cls._instance  # Return the same instance


class DataTransformer(ABC):
	"""Abstract base class for data transformation strategies."""

	@abstractmethod
	def transform(self, df):
		pass

class OHTransformerBuilder:
	qual_map = {
			'Ex': 5,  # Excellent
			'Gd': 4,  # Good
			'TA': 3,  # Typical/Average
			'Fa': 2,  # Fair
			'Po': 1   # Poor
		}

	def __init__(self, data) -> None:
		self.data = data
		self.most_freq_imputer = SimpleImputer(strategy='most_frequent')  # Fill missing values with the most frequent value
		self.zero_imputer = SimpleImputer(strategy='constant', fill_value=0)  # Use 0 for missing values
		self.mean_imputer = SimpleImputer(strategy="mean")


	def process_KitchenQuality(self):
		self.data['KitchenQual'] = self.data['KitchenQual'].map(self.qual_map)
		self.data['KitchenQual'] = self.most_freq_imputer.fit_transform(self.data[['KitchenQual']])
		return self

	def precss_ExterQuality(self):
		self.data['ExterQual'] = self.data['ExterQual'].map(self.qual_map)
		self.data['ExterCond'] = self.data['ExterCond'].map(self.qual_map)

		self.data['ExterQual'] = self.most_freq_imputer.fit_transform(self.data[['ExterQual']])
		self.data['ExterCond'] = self.most_freq_imputer.fit_transform(self.data[['ExterCond']])
		return self

	def process_BsmtFirePlaceGarage(self):
		self.data['BsmtCond'] = self.data['BsmtCond'].map(self.qual_map)
		self.data['FireplaceQu'] = self.data['FireplaceQu'].map(self.qual_map)
		self.data['GarageQual'] = self.data['GarageQual'].map(self.qual_map)

		self.data['BsmtCond'] = self.zero_imputer.fit_transform(self.data[['BsmtCond']])
		self.data['FireplaceQu'] = self.zero_imputer.fit_transform(self.data[['FireplaceQu']])
		self.data['GarageQual'] = self.zero_imputer.fit_transform(self.data[['GarageQual']])
		return self

	def process_age(self):
		# process date type col
		self.data['AgeAtSale'] = self.data['YrSold'] - self.data['YearBuilt']
		self.data['AgeAtSale'] -= (self.data['MoSold'] > 6)

		self.data['AgeRemodAdd'] = self.data['YrSold'] - self.data['YearRemodAdd']
		self.data['AgeRemodAdd'] -= (self.data['MoSold'] > 6)

		self.data['AgeGarage'] = self.data['YrSold'] - self.data['GarageYrBlt']
		self.data['AgeRemodAdd'] -= (self.data['MoSold'] > 6)

		self.data = self.data.drop(columns=['YearBuilt', 'YrSold', 'MoSold', 'YearRemodAdd', 'GarageYrBlt'])
		return self

	def process_living_area(self):
		self.data['TotalLivArea'] = self.data['GrLivArea'] + self.data['TotalBsmtSF']
		self.data = self.data.drop(columns=['GrLivArea', 'TotalBsmtSF'])
		return self


	def drop_low_importance(self):
		# print(self.data.describe(include="all"))
		return self

	def process_missing_numeric(self):
		# Impute numeric cols
		numeric_cols_with_missing = [col for col in self.data.select_dtypes(include=[np.number]).columns if self.data[col].isnull().any()]
		self.data[numeric_cols_with_missing] = self.mean_imputer.fit_transform(self.data[numeric_cols_with_missing])
		return self

	def process_garage(self):
		garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
		imputer = SimpleImputer(strategy="constant", fill_value="NoGarage")
		self.data[garage_cols] = imputer.fit_transform(self.data[garage_cols])
		return self

	def process_eletrical(self):
		self.data[['Electrical']] = self.most_freq_imputer.fit_transform(self.data[['Electrical']])
		return self

	def fill_missing(self):
		imputer = SimpleImputer(strategy="constant", fill_value="None")
		cols_with_missing = [col for col in self.data.columns if self.data[col].isnull().any()]
		self.data[cols_with_missing] = imputer.fit_transform(self.data[cols_with_missing])
		return self

	def encode_categoricals(self):
		# Identify categorical columns
		object_cols = self.data.select_dtypes(include=['object']).columns.tolist()
		# print("Categorical variables:", object_cols)

		# One-Hot Encoding with readable column names
		OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
		OH_encoded = OH_encoder.fit_transform(self.data[object_cols])

		# Convert to DataFrame with proper column names
		OH_col_names = OH_encoder.get_feature_names_out(object_cols)  # Get readable column names
		OH_cols_train = pd.DataFrame(OH_encoded, columns=OH_col_names, index=self.data.index)  # Maintain index

		# Drop original categorical columns and concatenate the one-hot encoded DataFrame
		self.data = self.data.drop(object_cols, axis=1)
		self.data = pd.concat([self.data, OH_cols_train], axis=1)

		# Ensure all column names are strings
		self.data.columns = self.data.columns.astype(str)
		return self

	def get_data(self):
		return self.data

class HousePriceTransformerDirector(DataTransformer):
	def transform(self, df):
		self.builder = OHTransformerBuilder(df)
		self.builder.process_KitchenQuality().precss_ExterQuality().process_BsmtFirePlaceGarage().process_age() \
			.process_missing_numeric().process_garage().process_eletrical().fill_missing() \
			.encode_categoricals()

		return self.builder.get_data()

class HousePriceModel(ABC):
	"""Abstract base class for house price prediction models."""

	def __init__(self, config):
		self.model = None  # Placeholder for the actual model
		self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None
		self.selected_features = None
		self.least_important_features = None
		self.train_path =  config.settings["dataset_path"]
		self.test_final_path = config.settings["testset_path"]

		# config transformer
		# self.transformer = HousePriceOHTransformer()
		self.transformer = HousePriceTransformerDirector()

	def get_name(self):
		return "HousePriceModel Abstract"

	def load_data(self, split=False, test_size=0.2, random_state=42):
		train_data_loader = DataLoader(self.train_path)
		df_train = train_data_loader.load_csv()
		# print(df_train)
		df_train = self.transform_data(df_train)

		x = df_train.drop(columns=['SalePrice'])
		y = df_train['SalePrice']

		if split:
			self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
				x, y, test_size=test_size, random_state=random_state
			)
		else:
			self.x_train = x
			self.y_train = y

		if self.test_final_path is not None:
			test_data_loader = DataLoader(self.test_final_path)
			self.x_final_test = test_data_loader.load_csv()
			self.x_final_test = self.transform_data(self.x_final_test)

			# Identify and handle missing columns
			missing_cols = set(self.x_train.columns) - set(self.x_final_test.columns)
			for col in missing_cols:
				self.x_final_test[col] = 0  # You may replace 0 with np.nan if needed

			# Ensure column order matches training data
			self.x_final_test = self.x_final_test[self.x_train.columns]

	def transform_data(self, X):
		return self.transformer.transform(X)


	@abstractmethod
	def train(self):
		"""Make predictions on the test set."""
		# predictions = self.pipeline.predict(self.x_final_test) if self.x_final_test is not None else None
		# return predictions
		# pass


	def clone(self, new_class):
		"""Clone the current instance into a new class while preserving data."""
		cloned_instance = copy.deepcopy(self)  # Create a deep copy
		cloned_instance.__class__ = new_class  # Change the class type
		return cloned_instance

	def evaluate_model(self, calculate_mae=False, output_to_stdout=False):
		"""Compute and return (r2_train, evaluation metrics as a string), or print to stdout if configured."""
		if self.model is None:
			raise RuntimeError("model is None")

		output = io.StringIO()
		output_target = sys.stdout if output_to_stdout else output  # Choose output destination

		print(f"\nmodel: {self.get_name()}\n=============================================", file=output_target)
		print(self.model, file=output_target)

		if self.selected_features is not None:
			print("apply with selected features")
			self.x_test = self.x_test[self.selected_features]


		if self.least_important_features is not None:
			print("drop less important features")
			self.x_test.drop(columns=self.least_important_features, errors='ignore')
		print(f"x_test shape:{self.x_test.shape}", file=output_target)
		train_predictions = self.model.predict(self.x_train)
		test_predictions = self.model.predict(self.x_test) if self.x_test is not None else None

		# Compute errors
		mse_train = mean_squared_error(self.y_train, train_predictions)
		r2_train = r2_score(self.y_train, train_predictions)

		print(f"Mean of y_train: {self.y_train.mean():.4f}", file=output_target)
		print(f"Range of y_train: {self.y_train.min()} - {self.y_train.max()}", file=output_target)
		print(f"Train MSE: {mse_train:.4f}, RMSE: {np.sqrt(mse_train):.2f}, R2: {r2_train:.4f}", file=output_target)

		if calculate_mae:
			scores = -1 * cross_val_score(self.model, self.x_train, self.y_train, cv=5, scoring='neg_mean_absolute_error')
			print("MAE scores:\n", scores, file=output_target)
			print("Average MAE score (across experiments):", scores.mean(), file=output_target)

		r2_test = None
		if self.y_test is not None:
			mse_test = mean_squared_error(self.y_test, test_predictions)
			r2_test = r2_score(self.y_test, test_predictions)
			print(f"Mean of y_test: {self.y_test.mean():.4f}", file=output_target)
			print(f"Range of y_test: {self.y_test.min()} - {self.y_test.max()}", file=output_target)
			print(f"Test MSE: {mse_test:.4f}, RMSE: {np.sqrt(mse_test):.2f}, R2: {r2_test:.4f}", file=output_target)
		else:
			print("y_test is None. Skipping test evaluation.", file=output_target)

		# return tuple(r2_score, report)
		return (r2_test if r2_test is not None else r2_train, None if output_to_stdout else output.getvalue())

	def predict_final_test(self, save = True, output_path="predictions.csv"):
		"""Make predictions on the final test set and save them to a CSV file.

		Args:
			output_path (str): File path to save predictions.
		"""
		if self.x_final_test is None or self.model is None:
			print("No test data or model available. Skipping prediction.")
			return  # No-op if test data is missing

		if self.selected_features is not None:
			print("apply with selected features")
			self.x_final_test = self.x_final_test[self.selected_features]


		if self.least_important_features is not None:
			print("drop less important features")
			self.x_final_test.drop(columns=self.least_important_features, errors='ignore')


		# Make predictions
		predictions = self.model.predict(self.x_final_test)

		if save:
			# Save predictions
			submission = pd.DataFrame({"Id": self.x_final_test['Id'], "SalePrice": predictions})
			submission.to_csv(output_path, index=False)
			print(f"Predictions saved to {output_path}")

		return predictions


	def plot_regression(self):
		"""Plot actual vs predicted values for regression analysis."""
		if self.y_test is not None:
			test_predictions = self.model.predict(self.x_test)
			plt.figure(figsize=(8, 6))
			sns.scatterplot(x=self.y_test, y=test_predictions)
			plt.xlabel("Actual SalePrice")
			plt.ylabel("Predicted SalePrice")
			plt.title("Actual vs Predicted Sale Prices")
			plt.show()
		else:
			print("y_test is None. Skipping regression plot.")

	def plot_correlation(self):
		"""Plot a heatmap for feature correlations."""
		correlation_matrix = self.x_train.corr()
		plt.figure(figsize=(10, 8))
		sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
		plt.title("Feature Correlation Matrix")
		plt.show()

	def plot_top10_correlation(self):
		"""Plot a heatmap for top 10 feature correlations."""
		correlation = self.x_train.corrwith(self.y_train).abs().sort_values(ascending=False)
		print(correlation)
		print(type(correlation))

		top_10_correlation = correlation.head(10)
		print(top_10_correlation)

		top_10_features = top_10_correlation.index.tolist()  # List of feature names
		top_10_values = top_10_correlation.tolist()  # List of correlation values

		print("Top 10 Features:", top_10_features)
		print("Top 10 Correlation Values:", top_10_values)
		for feature, value in top_10_correlation.items():
			print(f"{feature}: {value:.4f}")

		# top 10 related
		# Compute correlation with target variable (y_train)
		correlation = self.x_train.corrwith(self.y_train).abs()

		# Get top 10 correlated features with the target
		top_10_correlation = correlation.sort_values(ascending=False).head(10)

		# Select the top 10 features' correlations with other features
		top_10_features = top_10_correlation.index.tolist()

		# Compute the correlation matrix for just the top 10 features
		top_10_corr_matrix = self.x_train[top_10_features].corr()

		# Plot the heatmap for the top 10 correlation matrix
		plt.figure(figsize=(8, 6))
		sns.heatmap(top_10_corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0, linewidths=0.5)
		# Add title and labels
		plt.title("Top 10 Correlated Features Heatmap")
		plt.show()

	def plot_important_features(self):
		pass

class HousePriceDummy(HousePriceModel):
	def get_name(self):
		return "dummy"

	def train(self):
		pass

class HousePriceLinear(HousePriceModel):
	def get_name(self):
		return "LinearRegression"

	def train(self):
		self.model = Pipeline([
			('scaler', StandardScaler()),  # Scale features
			('linear', LinearRegression())
		])
		"""Fit the model to training data."""
		self.model.fit(self.x_train, self.y_train)
		# print(self.model)

class HousePriceRandomForest(HousePriceModel):
	def get_name(self):
		return "RandomForestRegressor"

	def train(self):
		self.model = Pipeline([
			('scaler', StandardScaler()),  # Scale features
			('forest', RandomForestRegressor(n_estimators=100, random_state=42))  # Random Forest model
		])
		"""Fit the model to training data."""
		self.model.fit(self.x_train, self.y_train)


	def plot_important_features(self):
		# Get feature names from x_train
		feature_names = self.x_train.columns  # Extract column names

		# Extract feature importance from the RandomForest model inside the pipeline
		feature_importances = self.model.named_steps['forest'].feature_importances_


		# Create a DataFrame to show feature importance
		importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})
		importance_df = importance_df.sort_values(by="Importance", ascending=False)
		# self.least_important_features = importance_df.tail(10)['Feature'].tolist()
		print("Feature Importance Ranking(Forest):")
		print(importance_df)

		top_20 = importance_df.head(20)

		plt.figure(figsize=(10, 5))
		plt.barh(top_20["Feature"], top_20["Importance"], color="blue")
		plt.xlabel("Feature Importance Score")
		plt.ylabel("Feature")
		plt.title("Top 20 Feature Importance (Random Forest)")
		plt.gca().invert_yaxis()  # Show most important feature on top
		plt.show()



class HousePriceXGB(HousePriceModel):
	def get_name(self):
		return "XGBRegressor"

	def train(self):
		my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
		print(f"xgboost version: {xgboost.__version__}")
		"""Fit the model to training data."""
		self.model = my_model.fit(self.x_train, self.y_train)

	def plot_important_features(self):
		pass

		# self.model = my_model.fit(self.x_train, self.y_train,
		#      eval_set=[(self.x_test, self.y_test)],
		#      verbose=False)
		# print(self.model)

class HousePriceAdaBoost(HousePriceModel):
	def get_name(self):
		return "AdaBoostRegressor"

	def train(self):
		base_model = DecisionTreeRegressor(max_depth=3)
		self.model = AdaBoostRegressor(base_model, n_estimators=50, random_state=42)
		# Train and evaluate
		self.model.fit(self.x_train, self.y_train)

		"""Fit the model to training data."""
		# print(self.model)


class HousePriceStackingRgressor(HousePriceModel):
	def get_name(self):
		return "StackingRegressor"

	def train(self):
		# Initialize base models and meta-model
		base_learners = [
			('ridge', Ridge(alpha=1.0)),
			('dt', DecisionTreeRegressor(random_state=42)),
			('svr', SVR())
		]
		meta_model = Ridge()

		# Initialize Stacking Regressor
		self.model = StackingRegressor(estimators=base_learners, final_estimator=meta_model)
		self.model.fit(self.x_train, self.y_train)

class HousePriceLasso(HousePriceModel):
	def get_name(self):
		return "Lasso"

	def train(self):

		self.model = Lasso(alpha=0.1)
		self.model.fit(self.x_train, self.y_train)

	def plot_important_features(self):
		# Get feature names from x_train
		feature_names = self.x_train.columns

		# Get absolute values of coefficients and sort by importance
		coef_abs = np.abs(self.model.coef_)
		top_20_indices = np.argsort(coef_abs)[-20:]  # Get indices of top 20 features

		# Select top 20 features and their coefficients
		top_features = feature_names[top_20_indices]
		top_coefs = self.model.coef_[top_20_indices]

		# Plot
		plt.figure(figsize=(10, 5))
		plt.barh(top_features, top_coefs, color="blue")
		plt.xlabel("Coefficient Value")
		plt.ylabel("Feature")
		plt.title("Top 20 LASSO Feature Importance")
		plt.gca().invert_yaxis()  # Invert to show most important at the top
		plt.show()



class HousePriceLassoXGB(HousePriceLasso):  # Inherit from HousePriceLasso to reuse feature selection
	def get_name(self):
		return "LassoXGBoost"

	def train(self):
		super().train()  # Reuse LASSO feature selection from parent class
		self.selected_features = self.x_train.columns[self.model.coef_ != 0]

		print(f"Filter out features(Lasso coef == 0): {self.x_train.columns[self.model.coef_ == 0]}")
		# print(f"Coef {self.model.coef_}")

		self.x_train = self.x_train[self.selected_features]  # Use selected features
		self.model = XGBRegressor(n_estimators=1000, learning_rate=0.05).fit(self.x_train, self.y_train)



class HousePriceForestLasso(HousePriceRandomForest):
	def get_name(self):
		return "ForestLasso"

	def train(self, filter_out_size=50):
		super().train()  # Reuse LASSO feature selection from parent class

				# Get feature names from x_train
		feature_names = self.x_train.columns  # Extract column names

		# Extract feature importance from the RandomForest model inside the pipeline
		feature_importances = self.model.named_steps['forest'].feature_importances_


		# Create a DataFrame to show feature importance
		importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})
		importance_df = importance_df.sort_values(by="Importance", ascending=False)

		# drop the less importance
		self.least_important_features = importance_df.tail(filter_out_size)['Feature'].tolist()

		# Drop the least important features
		self.x_train.drop(columns=self.least_important_features, errors='ignore', inplace=True)

		print("Feature Importance Ranking(Forest):")
		print(importance_df)

		self.model = Lasso(alpha=0.1)
		self.model.fit(self.x_train, self.y_train)

		# print(f"original: {self.x_train.shape}")


class HousePriceForestLassoXGB(HousePriceForestLasso):  # Inherit from HousePriceLasso to reuse feature selection
	def get_name(self):
		return "ForestLassoXGB"

	def train(self, filter_out_size=45):
		super().train(filter_out_size)  # Reuse LASSO feature selection from parent class
		self.selected_features = self.x_train.columns[self.model.coef_ != 0]

		print(f"Filter out features(Lasso coef == 0): {self.x_train.columns[self.model.coef_ == 0]}")

		self.x_train = self.x_train[self.selected_features]  # Use selected features
		self.model = XGBRegressor(n_estimators=1000, learning_rate=0.05).fit(self.x_train, self.y_train)


class HousePriceModelFactory:
	"""Factory to create different house price prediction models."""
	predictor_dummy = None

	@staticmethod
	def load_config(config):
		HousePriceModelFactory.predictor_dummy = HousePriceDummy(config)
		HousePriceModelFactory.predictor_dummy.load_data(split=True)

	@staticmethod
	def create_model(model_type):
		if HousePriceModelFactory.predictor_dummy is None:
			raise RuntimeError("Config must be loaded first using load_config().")

		model_map = {
			"LinearRegression": HousePriceLinear,
			"RandomForestRegressor": HousePriceRandomForest,
			"XGBRegressor": HousePriceXGB,
			"AdaBoostRegressor": HousePriceAdaBoost,
			"StackingRegressor": HousePriceStackingRgressor,
			"Lasso": HousePriceLasso,
			"LassoXGBoost": HousePriceLassoXGB,
			"ForestLassoXGB":HousePriceForestLassoXGB
		}

		if model_type not in model_map:
			raise ValueError(f"Unknown model type: {model_type}")

		return HousePriceModelFactory.predictor_dummy.clone(model_map[model_type])

def main():
	config = ConfigManager()
	print(config.settings)

	# # Initialize the factory with config
	# HousePriceModelFactory.load_config(config)

	# # List of models to train and evaluate
	# model_types = [
	# 	"LinearRegression",
	# 	"RandomForestRegressor",
	# 	"XGBRegressor",
	# 	"AdaBoostRegressor",
	# 	"StackingRegressor",
	# 	"Lasso",
	# 	"LassoXGBoost",
	# 	"ForestLassoXGB"
	# ]

	# # Create, train, and evaluate each model using the factory
	# models_report = []
	# for model_type in model_types:
	# 	model = HousePriceModelFactory.create_model(model_type)
	# 	model.train()
	# 	r2_train, report = model.evaluate_model()  # Returns (r2_train, report string)
	# 	models_report.append((r2_train, report))  # Store as tuple

	# models_report.sort(key=lambda x: x[0])
	# for _, report in models_report:
	# 	print(report)



	# construct dummy to load data
	predictor_dummy = HousePriceDummy(config)
	predictor_dummy.load_data(split = True)

	predictor = predictor_dummy.clone(HousePriceLassoXGB)
	predictor.train()
	predictor.predict_final_test()

	#####  plot important features
	# predictor_forest = predictor_dummy.clone(HousePriceRandomForest)
	# predictor_forest.train()
	# predictor_forest.plot_important_features()

	# predictor_lasso = predictor_dummy.clone(HousePriceLasso)
	# predictor_lasso.train()
	# predictor_lasso.plot_important_features()
	##############################



	# score, report =	predictor_forestlasso.evaluate_model()


	### test how many random forest less important features to drop
	# highest_score = 0
	# highest_paramter = 0
	# for i in range(1, 200):
	# 	predictor_forestlasso = predictor_dummy.clone(HousePriceForestLassoXGB)
	# 	predictor_forestlasso.train(filter_out_size=i)
	# 	score, report =	predictor_forestlasso.evaluate_model()
	# 	if score > highest_score:
	# 		highest_score = score
	# 		highest_paramter = i
	# 	print(report)
	# print(f"highest_score:{highest_score} highest_paramter:{highest_paramter}")


	# # predictor_lasso = predictor_dummy.clone(HousePriceLasso)
	# # predictor_lasso.train()
	# # predictor_lasso.evaluate_model()




	# _, report =	predictor_lassoXGB.evaluate_model()
	# print(report)
	# predictor_linear = predictor_dummy.clone(HousePriceLinear)
	# predictor_linear.train()
	# predictor_linear.evaluate_model()

	# predictor_forest.evaluate_model()

	# predictorxgb = predictor_dummy.clone(HousePriceXGB)
	# predictorxgb.train()
	# predictorxgb.evaluate_model()
	# predictorxgb.plot_regression()

	# predictoradaboost = predictor_dummy.clone(HousePriceAdaBoost)
	# predictoradaboost.train()
	# predictoradaboost.evaluate_model()

	# predictorstacking = predictor_dummy.clone(HousePriceStackingRgressor)
	# predictorstacking.train()
	# predictorstacking.evaluate_model()

if __name__ == "__main__":
	main()

