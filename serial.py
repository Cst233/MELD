import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from datasets import Dataset
from scipy.io.arff import loadarff
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder

# Helper imports must be preserved for dynamic eval() calls
from helper.external_datasets_variables import *

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Constants
DATA_DIR = Path('datasets')
DATASET_NAMES = [
    'heart', 'diabetes', 'blood', 'calhousing', 'car', 
    'student', 'employee', 'jungle', 'health', 'bank', 'income'
]

FILE_MAP = {
    'heart': 'heart.csv',
    'creditg': 'dataset_31_credit-g.arff',
    'diabetes': 'diabetes.csv',
    'blood': 'php0iVrYT.arff',
    'somerville': 'SomervilleHappinessSurvey2015.csv',
    'bank': 'phpkIxskf.arff',
    'caesarian': 'caesarian.arff',
    'calhousing': 'houses.arff',
    'car': 'car.data',
    'student': 'student_lifestyle_dataset.csv',
    'health': 'Maternal Health Risk Data Set.csv',
    'haberman': 'dataset_43_haberman.arff',
    'employee': 'Employee.csv',
    'jungle': 'jungle_chess_2pcs_raw_endgame_complete.arff',
    'water': 'water_potability.csv',
    'breast_cancer': 'breast-cancer.data',
    'income': ''  # income has multiple files, handled specially
}

def parse_args():
    parser = argparse.ArgumentParser(description="Serialize datasets for MELD experiments.")
    parser.add_argument(
        "--mode", 
        type=str, 
        default='gtl', 
        choices=['meld_bin', 'meld_nobin', 'gtl', 'tabula'],
        help="Processing mode, determines binning strategy and output folder."
    )
    return parser.parse_args()

def decode_byte_columns(df):
    """Decodes byte-string columns in a DataFrame to regular strings."""
    for col, dtype in df.dtypes.items():
        if dtype == object:
            df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
    return df

def load_and_preprocess_dataframe(dataset_name, file_path):
    """Loads raw data and performs dataset-specific preprocessing."""
    
    if dataset_name == "creditg":
        # https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
        df = pd.DataFrame(loadarff(file_path)[0])
        df = decode_byte_columns(df)
        df.rename(columns={'class': 'label'}, inplace=True)
        df['label'] = df['label'] == 'good'

    elif dataset_name == "blood":
        col_map = {'V1': 'recency', 'V2': 'frequency', 'V3': 'monetary', 'V4': 'time', 'Class': 'label'}
        df = pd.DataFrame(loadarff(file_path)[0])
        df = decode_byte_columns(df)
        df.rename(columns=col_map, inplace=True)
        df['label'] = df['label'] == '2'

    elif dataset_name == "bank":
        feature_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day',
                         'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
        col_map = {'V' + str(i + 1): v for i, v in enumerate(feature_names)}
        df = pd.DataFrame(loadarff(file_path)[0])
        df = decode_byte_columns(df)
        df.rename(columns=col_map, inplace=True)
        df.rename(columns={'Class': 'label'}, inplace=True)
        df['label'] = df['label'] == '2'

    elif dataset_name == "calhousing":
        df = pd.DataFrame(loadarff(file_path)[0])
        df = decode_byte_columns(df)
        df.rename(columns={'median_house_value': 'label'}, inplace=True)
        # Convert to binary task: median split
        median_price = df['label'].median()
        df['label'] = df['label'] > median_price

    elif dataset_name == "car":
        cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety_dict', 'label']
        df = pd.read_csv(file_path, names=cols)
        label_dict = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
        df['label'] = df['label'].replace(label_dict)

    elif dataset_name == "heart":
        df = pd.read_csv(file_path)
        df = df.rename(columns={'HeartDisease': 'label'})
        # Handle zero values as missing for Cholesterol
        df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)
        
        # Impute missing values
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        encoder = OrdinalEncoder()
        df[cat_cols] = encoder.fit_transform(df[cat_cols])
        
        imputer = IterativeImputer(random_state=0)
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        
        df_imputed[cat_cols] = encoder.inverse_transform(df_imputed[cat_cols])
        int_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'label']
        for feat in int_cols:
            df_imputed[feat] = df_imputed[feat].astype(int)
        df = df_imputed

    elif dataset_name == "diabetes":
        # https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
        df = pd.read_csv(file_path)
        df = df.rename(columns={'Outcome': 'label'})
        
        # Treat 0 as NaN for relevant columns
        cols_to_clean = [col for col in df.columns if col not in ['Pregnancies', 'label']]
        df[cols_to_clean] = df[cols_to_clean].replace(0, np.nan)
        
        imputer = IterativeImputer(random_state=0)
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        
        int_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'label', 'Age']
        for feat in int_cols:
            df[feat] = df[feat].astype(int)

    elif dataset_name == "employee":
        df = pd.read_csv(file_path)
        df = df.rename(columns={'LeaveOrNot': 'label'})

    elif dataset_name == "somerville":
        df = pd.read_csv(file_path)
        df = df.rename(columns={'D': 'label'})

    elif dataset_name == "caesarian":
        df = pd.DataFrame(loadarff(file_path)[0])
        df = decode_byte_columns(df)
        df['Age'] = df['Age'].astype(int)
        df.rename(columns={'Caesarian': 'label'}, inplace=True)
        df['label'] = df['label'] == '1'

    elif dataset_name == "student":
        df = pd.read_csv(file_path)
        df.drop(columns=['Student_ID'], inplace=True)
        df.rename(columns={'Stress_Level': 'label'}, inplace=True)
        label_dict = {'High': 0, 'Moderate': 1, 'Low': 2}
        df['label'] = df['label'].replace(label_dict)

    elif dataset_name == "haberman":
        df = pd.DataFrame(loadarff(file_path)[0])
        df = decode_byte_columns(df)
        df['Patients_year_of_operation'] = df['Patients_year_of_operation'].astype(int)
        df.rename(columns={'Survival_status': 'label'}, inplace=True)
        df['label'] = df['label'] == '1' # 1 = survived >= 5 years

    elif dataset_name == "jungle":
        df = pd.DataFrame(loadarff(file_path)[0])
        df = decode_byte_columns(df)
        df.rename(columns={'class': 'label'}, inplace=True)
        df['label'] = df['label'] == 'w'  # White wins

    elif dataset_name == "health":
        df = pd.read_csv(file_path)
        df = df.rename(columns={'RiskLevel': 'label'})
        label_dict = {'high risk': 0, 'mid risk': 1, 'low risk': 2}
        df['label'] = df['label'].replace(label_dict)

    elif dataset_name == "water": 
        df = pd.read_csv(file_path)
        df = df.rename(columns={'Potability': 'label'})
        imputer = IterativeImputer(random_state=0)
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        df['label'] = df['label'].astype(int)

    elif dataset_name == "breast_cancer": 
        cols = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']
        df = pd.read_csv(file_path, names=cols)
        df = df.rename(columns={'Class': 'label'})
        df['label'] = df['label'] == 'recurrence-events'
        
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        encoder = OrdinalEncoder()
        df[cat_cols] = encoder.fit_transform(df[cat_cols])
        
        imputer = IterativeImputer(random_state=0)
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        
        df_imputed[cat_cols] = encoder.inverse_transform(df_imputed[cat_cols])
        df_imputed['deg-malig'] = df_imputed['deg-malig'].astype(int)
        df_imputed['label'] = df_imputed['label'].astype(int)
        df = df_imputed

    elif dataset_name == "income":
        cols = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
                'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
                'native_country', 'label']

        def strip_string_columns(d):
            d[d.select_dtypes(['object']).columns] = d.select_dtypes(['object']).apply(lambda x: x.str.strip())

        # Load Train
        df_train = pd.read_csv(DATA_DIR / dataset_name / 'adult.data', names=cols, na_values=['?', ' ?'])
        df_train = df_train.drop(columns=['fnlwgt', 'education_num'])
        strip_string_columns(df_train)
        df_train['label'] = df_train['label'] == '>50K'

        # Load Test
        df_test = pd.read_csv(DATA_DIR / dataset_name / 'adult.test', names=cols, na_values=['?', ' ?'])
        df_test = df_test.drop(columns=['fnlwgt', 'education_num'])
        strip_string_columns(df_test)
        df_test['label'] = df_test['label'] == '>50K.' # Note the trailing dot in raw file

        # Concatenate for global quantization/encoding
        df = pd.concat([df_train, df_test], ignore_index=True)

        cat_cols = df.select_dtypes(include='object').columns.tolist()
        encoder = OrdinalEncoder()
        df[cat_cols] = encoder.fit_transform(df[cat_cols])
        
        imputer = IterativeImputer(random_state=0)
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        
        df_imputed[cat_cols] = encoder.inverse_transform(df_imputed[cat_cols])
        df_imputed['label'] = df_imputed['label'].astype(int)
        df = df_imputed

    return df

def main():
    args = parse_args()
    
    # Determine if continuous variables should be binned based on mode
    enable_binning = False if args.mode in ['meld_nobin', 'gtl', 'tabula'] else True

    for dataset_name in DATASET_NAMES:
        print(f"Processing dataset: {dataset_name}...")
        
        file_path = DATA_DIR / dataset_name / FILE_MAP[dataset_name]
        df = load_and_preprocess_dataframe(dataset_name, file_path)

        categorical_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        # Retrieve dataset-specific configuration from external variables
        # Note: 'eval' is used to resolve variable names dynamically imported from helper
        except_numerical_list = eval(f"except_numerical_value_{dataset_name}")
        except_numerical_list.append('label')

        if enable_binning:
            for feature in numeric_cols:
                if feature in except_numerical_list:
                    continue
                
                # Special handling for 'bank' dataset pdays
                if dataset_name == 'bank' and feature == 'pdays':
                    df.loc[df[feature] == -1, feature] = np.nan

                # Calculate non-equal width bins based on quantiles
                q1 = df[feature].quantile(0.20)
                q2 = df[feature].quantile(0.40)
                q3 = df[feature].quantile(0.60)
                q4 = df[feature].quantile(0.80)
                
                # Fallback to equal-width if quantiles are not strictly increasing
                if not (q1 < q2 < q3 < q4):
                    min_val, max_val = df[feature].min(), df[feature].max()
                    step = (max_val - min_val) / 5
                    q1 = min_val + step
                    q2 = min_val + 2 * step
                    q3 = min_val + 3 * step
                    q4 = min_val + 4 * step

                bins = [-float('inf'), q1, q2, q3, q4, float('inf')]
                labels = ['very low', 'low', 'medium', 'high', 'very high']
                
                df[feature] = pd.cut(df[feature], bins=bins, labels=labels, include_lowest=True)

                if dataset_name == 'bank' and feature == 'pdays':
                    df[feature] = df[feature].astype(object)
                    df.loc[df[feature].isna(), feature] = "client was not previously contacted"

        # Load configuration for serialization
        dataset_feature_tuples = eval(f"{dataset_name}_feature_names")
        dataset_pre_processing = eval(f"template_config_{dataset_name}['pre']")
        task_description = eval(f"{dataset_name}_task_description")

        def serialize_row(row):
            """Converts a DataFrame row into a serialized string feature representation."""
            features = []
            for i, col in enumerate(row.index):
                value = row[col]
                
                # Convert float integers to int for cleaner strings
                if isinstance(value, float) and value.is_integer():
                    value = int(value)
                
                # Apply dataset specific value formatting (e.g. F -> female)
                if col in dataset_pre_processing:
                    value = dataset_pre_processing[col](value)
                
                feature_desc = dataset_feature_tuples[i][1]
                
                # Serialization logic for MELD modes
                if args.mode in ['meld_nobin', 'meld_bin']:
                    features.append(f"{feature_desc} is {value}")
            
            return "{" + ", ".join(features) + "}"

        # Create serialized DataFrame
        # 'note' column contains the JSON-like serialized string
        processed_data = pd.DataFrame({
            'note': df.drop(['label'], axis=1).apply(serialize_row, axis=1),
            'label': df['label'].astype(int),
            'task_description': [task_description] * len(df),
            'features': ['; '.join([desc for _, desc in dataset_feature_tuples])] * len(df)
        })

        if not processed_data.empty:
            print(f"Sample serialization: {processed_data.iloc[1, 0]}")

        # Convert to HuggingFace Dataset
        hf_dataset = Dataset.from_pandas(processed_data)

        if enable_binning:
            output_dir_path = f'datasets_json_serialized/{dataset_name}'
        else:
            output_dir_path = f'datasets_json_serialized_{args.mode}/{dataset_name}'

        hf_dataset.save_to_disk(output_dir_path)
        print(f"Saved to {output_dir_path}\n")

if __name__ == "__main__":
    main()