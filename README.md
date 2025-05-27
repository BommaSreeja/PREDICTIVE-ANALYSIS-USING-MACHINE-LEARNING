# PREDICTIVE-ANALYSIS-USING-MACHINE-LEARNING

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: BOMMA SREEJA

*INTERN ID*: CT04DL352

*DOMAIN*: DATA ANALYSIS

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTHOSH

his Python program is a comprehensive machine learning pipeline for building a crop recommendation system based on environmental and soil attributes. It leverages popular data science libraries including Pandas, NumPy, Matplotlib, Seaborn, and machine learning components from Scikit-learn. The code performs end-to-end tasks from data loading and exploration to feature selection, model training, and evaluation, offering a practical solution for predicting suitable crops based on environmental data.
The process begins by importing essential libraries. Pandas and NumPy handle data manipulation, while Matplotlib and Seaborn are used for visualizing data insights, such as correlation matrices and model evaluation plots. Scikit-learn modules are used for data preprocessing, feature selection, splitting the dataset, and training and evaluating machine learning models.
The dataset, stored in CSV format, is loaded using Pandas. It includes agricultural features such as nitrogen (N), phosphorus (P), potassium (K), temperature, humidity, pH, and rainfall, along with the crop label. An initial data exploration step checks for missing values and overall structure using df.info() and df.isnull().sum(). Any missing entries are dropped for simplicity, although more advanced imputation methods could be used in production systems.
Next, the code performs feature selection through two techniques. First, a correlation matrix is plotted using Seaborn’s heatmap to understand linear relationships among numerical variables. This visualization helps identify highly correlated features which might affect model performance due to multicollinearity. Following this, a Random Forest Classifier is trained to determine feature importance. Random Forest is an ensemble method known for its ability to handle both linear and non-linear relationships and for ranking feature importance effectively. Features with importance scores greater than 0.05 are selected for the final model, helping reduce noise and improve accuracy.
After selecting the most relevant features, the dataset is split into training and testing sets using an 80-20 ratio. To ensure optimal model performance, particularly for models sensitive to feature scales, the input features are standardized using StandardScaler. This step normalizes the feature values, ensuring they all have the same scale, which is crucial for algorithms like Logistic Regression.
The chosen model for classification is Logistic Regression, a simple yet effective linear model suitable for multiclass classification problems. The model is trained on the scaled training data and tested on the test set. The performance is evaluated using accuracy score, confusion matrix, and a classification report that includes precision, recall, and F1-score for each crop class.
The final output includes a confusion matrix heatmap and a detailed classification report, which offer a clear picture of how well the model is distinguishing between different crop labels. This machine learning pipeline is highly applicable in the field of precision agriculture, enabling farmers, agronomists, and AgriTech companies to make data-driven crop decisions based on environmental inputs.
Designed to run on any standard Python environment with necessary libraries installed, this tool is ideal for agricultural research, educational projects, and real-world farming applications where crop selection is critical for maximizing yield and sustainability.

*OUTPUT*:

![Image](https://github.com/user-attachments/assets/82868d42-6414-4ccf-b332-ed0a383d5548)

![Image](https://github.com/user-attachments/assets/005d4e37-45c2-4b24-b80b-41f674f469b2)

