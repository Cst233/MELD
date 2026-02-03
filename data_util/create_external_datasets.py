import logging
import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder

# Preserving wildcard import for dynamic eval() calls relying on external variables
from helper.external_datasets_variables import *

logger = logging.getLogger(__name__)

def load_train_validation_test(dataset_name, data_dir, bin_count=5, quantify=True):
    """
    Loads, preprocesses, and splits the dataset into train, validation, and test sets.
    
    Args:
        dataset_name: Name of the dataset.
        data_dir: Path to the directory containing raw data files.
        bin_count: Number of bins for quantizing continuous features.
        quantify: Whether to apply quantization (binning).
    
    Returns:
        Dictionary containing 'train', 'validation', and 'test' pandas DataFrames.
    """

    def split_data(data):
        """
        Splits data: 50% Train, 10% Validation, 40% Test.
        shuffle=False is crucial to maintain alignment with other processing scripts.
        """
        data_train, data_rest = train_test_split(data, test_size=0.50, shuffle=False)
        data_valid, data_test = train_test_split(data_rest, test_size=0.80, shuffle=False)
        return data_train, data_valid, data_test

    def decode_byte_columns(data):
        """Decodes byte-string columns to UTF-8 strings."""
        for col, dtype in data.dtypes.items():
            if dtype == object:
                data[col] = data[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
        return data

    dataset = None

    # --- Dataset Loading Logic ---

    if dataset_name == "creditg":
        dataset = pd.DataFrame(loadarff(data_dir / 'dataset_31_credit-g.arff')[0])
        dataset = decode_byte_columns(dataset)
        dataset.rename(columns={'class': 'label'}, inplace=True)
        dataset['label'] = dataset['label'] == 'good'

    elif dataset_name == "blood":
        columns = {'V1': 'recency', 'V2': 'frequency', 'V3': 'monetary', 'V4': 'time', 'Class': 'label'}
        dataset = pd.DataFrame(loadarff(data_dir / 'php0iVrYT.arff')[0])
        dataset = decode_byte_columns(dataset)
        dataset.rename(columns=columns, inplace=True)
        dataset['label'] = dataset['label'] == '2'

    elif dataset_name == "bank":
        columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day',
                   'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
        col_map = {'V' + str(i + 1): v for i, v in enumerate(columns)}
        dataset = pd.DataFrame(loadarff(data_dir / 'phpkIxskf.arff')[0])
        dataset = decode_byte_columns(dataset)
        dataset.rename(columns=col_map, inplace=True)
        dataset.rename(columns={'Class': 'label'}, inplace=True)
        dataset['label'] = dataset['label'] == '2'

    elif dataset_name == "calhousing":
        dataset = pd.DataFrame(loadarff(data_dir / 'houses.arff')[0])
        dataset = decode_byte_columns(dataset)
        dataset.rename(columns={'median_house_value': 'label'}, inplace=True)
        # Create binary task by splitting at median
        median_price = dataset['label'].median()
        dataset['label'] = dataset['label'] > median_price

    elif dataset_name == "car":
        columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety_dict', 'label']
        dataset = pd.read_csv(data_dir / 'car.data', names=columns)
        label_dict = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
        dataset['label'] = dataset['label'].replace(label_dict)

    elif dataset_name == "heart":
        dataset = pd.read_csv(data_dir / 'heart.csv')
        dataset = dataset.rename(columns={'HeartDisease': 'label'})
        # Treat 0 cholesterol as missing
        dataset['Cholesterol'] = dataset['Cholesterol'].replace(0, np.nan)
        
        # Encoding & Imputation
        cat_cols = dataset.select_dtypes(include='object').columns.tolist()
        encoder = OrdinalEncoder()
        dataset[cat_cols] = encoder.fit_transform(dataset[cat_cols])
        
        imputer = IterativeImputer(random_state=0)
        df_imputed = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)
        df_imputed[cat_cols] = encoder.inverse_transform(df_imputed[cat_cols])
        df_imputed['FastingBS'] = df_imputed['FastingBS'].astype(int)
        df_imputed['label'] = df_imputed['label'].astype(int)
        dataset = df_imputed

    elif dataset_name == "diabetes":
        dataset = pd.read_csv(data_dir / 'diabetes.csv')
        dataset = dataset.rename(columns={'Outcome': 'label'})
        
        # Treat 0 as NaN for relevant columns
        cols_to_replace = [col for col in dataset.columns if col not in ['Pregnancies', 'label']]
        dataset[cols_to_replace] = dataset[cols_to_replace].replace(0, np.nan)
        
        imputer = IterativeImputer(random_state=0)
        dataset = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)
        dataset['label'] = dataset['label'].astype(int)
    
    elif dataset_name == "somerville":
        dataset = pd.read_csv(data_dir / 'SomervilleHappinessSurvey2015.csv')
        dataset = dataset.rename(columns={'D': 'label'})
    
    elif dataset_name == "caesarian":
        dataset = pd.DataFrame(loadarff(data_dir / 'caesarian.arff')[0])
        dataset = decode_byte_columns(dataset)
        dataset.rename(columns={'Caesarian': 'label'}, inplace=True)
        dataset['label'] = dataset['label'] == '1'
        dataset['Age'] = dataset['Age'].astype(int)
    
    elif dataset_name == "student":
        dataset = pd.read_csv(data_dir / 'student_lifestyle_dataset.csv')
        dataset.drop(columns=['Student_ID'], inplace=True)
        dataset = dataset.rename(columns={'Stress_Level': 'label'})
        label_dict = {'High': 0, 'Moderate': 1, 'Low': 2}
        dataset['label'] = dataset['label'].replace(label_dict)

    elif dataset_name == "employee":
        dataset = pd.read_csv(data_dir / 'Employee.csv')
        dataset = dataset.rename(columns={'LeaveOrNot': 'label'})

    elif dataset_name == "haberman":
        dataset = pd.DataFrame(loadarff(data_dir / 'dataset_43_haberman.arff')[0])
        dataset = decode_byte_columns(dataset)
        dataset.rename(columns={'Survival_status': 'label'}, inplace=True)
        dataset['label'] = dataset['label'] == '1' # 1 for survived 5 years or longer
        dataset['Patients_year_of_operation'] = dataset['Patients_year_of_operation'].astype(int)
    
    elif dataset_name == "jungle":
        dataset = pd.DataFrame(loadarff(data_dir / 'jungle_chess_2pcs_raw_endgame_complete.arff')[0])
        dataset = decode_byte_columns(dataset)
        dataset.rename(columns={'class': 'label'}, inplace=True)
        dataset['label'] = dataset['label'] == 'w'  # White wins

    elif dataset_name == "health":
        dataset = pd.read_csv(data_dir / 'Maternal Health Risk Data Set.csv')
        dataset.rename(columns={'RiskLevel': 'label'}, inplace=True)
        label_dict = {'high risk': 0, 'mid risk': 1, 'low risk': 2}
        dataset['label'] = dataset['label'].replace(label_dict)
    
    elif dataset_name == "water": 
        dataset = pd.read_csv(data_dir / "water_potability.csv")
        dataset = dataset.rename(columns={'Potability': 'label'})
        imputer = IterativeImputer(random_state=0)
        dataset = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)

    elif dataset_name == "breast_cancer": 
        columns = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']
        dataset = pd.read_csv(data_dir / "breast-cancer.data", names=columns)
        dataset = dataset.rename(columns={'Class': 'label'})
        dataset['label'] = dataset['label'] == 'recurrence-events'
        
        cat_cols = dataset.select_dtypes(include='object').columns.tolist()
        encoder = OrdinalEncoder()
        dataset[cat_cols] = encoder.fit_transform(dataset[cat_cols])
        
        imputer = IterativeImputer(random_state=0)
        df_imputed = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)
        df_imputed[cat_cols] = encoder.inverse_transform(df_imputed[cat_cols])
        df_imputed['deg-malig'] = df_imputed['deg-malig'].astype(int)
        df_imputed['label'] = df_imputed['label'].astype(int)
        dataset = df_imputed

    elif dataset_name == "income":
        columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
                   'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
                   'native_country', 'label']

        def strip_string_columns(df):
            df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.str.strip())

        dataset_train = pd.read_csv(data_dir / 'adult.data', names=columns, na_values=['?', ' ?'])
        dataset_train = dataset_train.drop(columns=['fnlwgt', 'education_num'])
        strip_string_columns(dataset_train)
        dataset_train['label'] = dataset_train['label'] == '>50K'

        dataset_test = pd.read_csv(data_dir / 'adult.test', names=columns, na_values=['?', ' ?'])
        dataset_test = dataset_test.drop(columns=['fnlwgt', 'education_num'])
        strip_string_columns(dataset_test)
        dataset_test['label'] = dataset_test['label'] == '>50K.'

        # Concatenate for global quantification context
        dataset = pd.concat([dataset_train, dataset_test], ignore_index=True)

        cat_cols = dataset.select_dtypes(include='object').columns.tolist()
        encoder = OrdinalEncoder()
        dataset[cat_cols] = encoder.fit_transform(dataset[cat_cols])
        
        imputer = IterativeImputer(random_state=0)
        df_imputed = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)
        df_imputed[cat_cols] = encoder.inverse_transform(df_imputed[cat_cols])
        df_imputed['label'] = df_imputed['label'].astype(int)
        dataset = df_imputed

    else:
        raise ValueError("Dataset not found")

    # --- Verification ---
    # Expected feature count (including label)
    dataset_specs = {
        'income': 13, 'car': 7, 'heart': 12, 'diabetes': 9, 'creditg': 21,
        'blood': 5, 'bank': 17, 'calhousing': 9, 'somerville': 7,
        'caesarian': 6, 'student': 7, 'employee': 9, 'haberman': 4,
        'jungle': 7, 'health': 7, 'water': 10, 'breast_cancer': 10
    }
    assert dataset_name in dataset_specs.keys() and len(dataset.columns) == dataset_specs[dataset_name]

    # --- Processing & Splitting ---
    if quantify:    
        dataset = quantify_df(dataset, dataset_name, bin_count)
    
    dataset_train, dataset_valid, dataset_test = split_data(dataset)

    return {"train": dataset_train, "validation": dataset_valid, "test": dataset_test}

def quantify_df(df, dataset_name, bin_count):
    """
    Discretizes continuous features into bins.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Dynamically fetch exclusion list (e.g., boolean or ID columns)
    except_numerical_value = eval(f"except_numerical_value_{dataset_name}")
    except_numerical_value.append('label')

    # Helper function to apply quantile binning or fallback to equal-width
    def calculate_bins(series, num_bins):
        quantiles = [series.quantile(i / num_bins) for i in range(1, num_bins)]
        
        # Check if quantiles are strictly increasing (needed for valid bins)
        strictly_increasing = all(x < y for x, y in zip(quantiles, quantiles[1:]))
        
        if not strictly_increasing:
            # Fallback to equal width
            min_val, max_val = series.min(), series.max()
            step = (max_val - min_val) / num_bins
            bins = [-float('inf')] + [min_val + i * step for i in range(1, num_bins)] + [float('inf')]
        else:
            bins = [-float('inf')] + quantiles + [float('inf')]
            
        return bins

    if bin_count == 5:
        labels = ['very low', 'low', 'medium', 'high', 'very high']
    elif bin_count == 4:
        labels = ['very low', 'low', 'high', 'very high']
    else:
        # Fallback or raise error if necessary, currently preserving logic implies only these specific bins are handled fully
        return df

    for feature in numeric_cols:
        if feature in except_numerical_value:
            # If specified in config, apply pre-processing map (e.g. converting ints to strings for one-hot encoding later)
            if feature != 'label':
                 pre_process_map = eval(f"template_config_{dataset_name}")['pre'].get(feature)
                 if pre_process_map:
                     df[feature] = df[feature].map(pre_process_map)
            continue

        if dataset_name == 'bank' and feature == 'pdays':
             df.loc[df[feature] == -1, feature] = np.nan

        # Calculate Bins
        bins = calculate_bins(df[feature], bin_count)
        
        # Apply Cut
        df[feature] = pd.cut(df[feature], bins=bins, labels=labels, include_lowest=True)

        # Dataset-specific cleanup
        if dataset_name == 'bank' and feature == 'pdays':
            df[feature] = df[feature].astype(object)
            df.loc[df[feature].isna(), feature] = "client was not previously contacted"
            
    return df