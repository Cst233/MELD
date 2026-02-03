import logging
import numpy as np
import pandas as pd
from datasets import DatasetDict, concatenate_datasets
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

logger = logging.getLogger(__name__)

def prepare_data(dataset_name, model_name, dataset, enc=None, scale=True):
    """
    Preprocesses the dataset by encoding categorical variables and scaling numerical ones.
    
    Args:
        dataset_name: Name of the dataset (unused in logic, kept for interface compatibility).
        model_name: Name of the model (affects encoding logic slightly).
        dataset: HuggingFace DatasetDict containing 'train', 'validation', 'test'.
        enc: Encoding method for categorical variables ('ordinal', 'one-hot', or None).
        scale: Boolean, whether to apply Z-score normalization to numerical features.
        
    Returns:
        Processed DatasetDict with pandas dataframes or compatible structure.
    """

    def remove_constants(data):
        """Removes columns with a single unique value."""
        return data[[c for c in data if data[c].nunique() > 1]]

    # Identify feature columns
    numeric_columns = [c for c in dataset['train'].select_dtypes(include=np.number).columns.tolist() if c not in ['idx', 'label']]
    cat_columns = [c for c in dataset['train'].columns.tolist() if (c not in (numeric_columns + ['idx', 'label']))]

    # Initialize OrdinalEncoder if needed
    if enc == 'ordinal':
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        ordinal_encoder.fit(dataset['train'][cat_columns])

    def encode_categorical(data):
        """Applies the selected encoding strategy to categorical columns."""
        if enc is not None and model_name != 'output_datasets':
            if enc == 'one-hot':
                # Convert columns to string to ensure consistent naming
                data.columns = data.columns.astype(str)
                data = pd.get_dummies(data, dummy_na=True) 
                data.columns = data.columns.astype(object) 
                data = remove_constants(data)
            elif enc == 'ordinal' and data.shape[0] > 0:
                data[cat_columns] = ordinal_encoder.transform(data[cat_columns])
        return data

    # Process Training Data
    dataset['train'] = encode_categorical(dataset['train'])

    if scale and len(numeric_columns) > 0:
        # Z-normalization of numerical columns based on training statistics
        scaler = StandardScaler()
        scaler.fit(dataset['train'][numeric_columns])
        dataset['train'][numeric_columns] = scaler.transform(dataset['train'][numeric_columns])

    def process_eval_split(split_name):
        """Processes validation/test splits using training transformations."""
        data = dataset[split_name]
        data = encode_categorical(data)
        
        if scale and len(numeric_columns) > 0 and data.shape[0] > 0:
            data[numeric_columns] = scaler.transform(data[numeric_columns])

        # Align columns: Add missing columns (from one-hot encoding) as False/0
        for column in [c for c in dataset['train'].columns if c not in data.columns]:
            data[column] = False
            
        # Ensure column order matches training set and drop extra columns
        return data[dataset['train'].columns]

    dataset['validation'] = process_eval_split('validation')
    dataset['test'] = process_eval_split('test')
    
    assert (len(dataset['train'].columns) == len(dataset['validation'].columns) == len(dataset['test'].columns))
    return dataset


def read_orig_dataset(orig_data, seed, split):
    """
    Splits the original dataset and returns the requested split.
    
    The split logic is:
    1. Initial 50/50 split. 
    2. The second half is further split 20/80 into Validation/Test.
    3. Train set combines the first half and the validation portion (effectively 60% Train, 40% Test).
    """
    
    # Stratified split simulation (randomized shuffle)
    data = orig_data.train_test_split(test_size=0.50, seed=seed)
    data2 = data['test'].train_test_split(test_size=0.80, seed=seed)
    
    dataset_dict = DatasetDict({
        'train': concatenate_datasets([data['train'], data2['train']]),
        'validation': data2['train'],
        'test': data2['test']
    })
    
    selected_split_data = dataset_dict[split]

    # Ensure an 'idx' column exists for tracking
    if 'idx' not in selected_split_data.column_names:
        selected_split_data = selected_split_data.add_column(name='idx', column=range(0, selected_split_data.num_rows))

    return selected_split_data