import re

# Mapping from integer label to string target
LABEL_2_TARGET = {
    'heart': {0: "No", 1: "Yes"},
    'creditg': {0: "No", 1: "Yes"},
    'diabetes': {0: "No", 1: "Yes"},
    'blood': {0: "No", 1: "Yes"},
    'bank': {0: "No", 1: "Yes"},
    'calhousing': {0: "No", 1: "Yes"},
    'car': {0: 'Unacceptable', 1: 'Acceptable', 2: 'Good', 3: 'Very good'},
    'student': {0: 'High', 1: 'Moderate', 2: 'Low'},
    'employee': {0: 'No', 1: 'Yes'},
    'jungle': {0: 'No', 1: 'Yes'},
    'health': {0: 'High', 1: 'Mid', 2: 'Low'},
    'income': {0: 'No', 1: 'Yes'}
}

# Mapping from string target to integer label
TARGET_2_LABEL = {
    'heart': {"No": 0, "Yes": 1},
    'creditg': {"No": 0, "Yes": 1},
    'diabetes': {"No": 0, "Yes": 1},
    'blood': {"No": 0, "Yes": 1},
    'bank': {"No": 0, "Yes": 1},
    'calhousing': {"No": 0, "Yes": 1},
    'car': {'Unacceptable': 0, 'Acceptable': 1, 'Good': 2, 'Very good': 3},
    'student': {'High': 0, 'Moderate': 1, 'Low': 2},
    'employee': {"No": 0, "Yes": 1},
    'jungle': {"No": 0, "Yes": 1},
    'health': {"High": 0, "Mid": 1, "Low": 2},
    'income': {"No": 0, "Yes": 1}
}

def target_2_label(target_str, dataset_name):
    """
    Convert a target string (e.g., 'Yes') to its corresponding integer label.
    """
    return TARGET_2_LABEL[dataset_name][target_str]

def extract_all_first_choices(dataset_name, responses):
    """
    Extracts the first prediction from the model responses.
    This function is used to parse the raw text output from the LLM into a clean label string.

    Supports responses in formats like:
    1. "Prediction of new instance: <label>"
    2. "Yes; No; Yes" (extracts the first one) or other separators like newline.
    
    Args:
        dataset_name: Name of the dataset (kept for API compatibility, unused in current logic).
        responses: List of string responses from the LLM.
        
    Returns:
        List of extracted prediction strings (capitalized).
    """
    preds = []

    for resp in responses:
        if not resp:
            preds.append(None)
            continue

        # 1. Try to extract content after specific phrase "Prediction of new instance:"
        prediction_match = re.search(r"Prediction of new instance:\s*(.+?)(?:\n|$)", resp, re.IGNORECASE)
        if prediction_match:
            pred_text = prediction_match.group(1).strip()
            # Remove possible trailing punctuation like . ! ? ; ,
            pred_text = re.sub(r"[.!?;,]+$", "", pred_text).strip()
            preds.append(pred_text.capitalize() if pred_text else None)
            continue
        
        # 2. Heuristic split by common separators (comma, semicolon, newline)
        # We take the first non-empty segment.
        parts = re.split(r"[,;\n]+", resp)
        found_pred = None
        
        for part in parts:
            # Clean up potential prefixes like 'Answer:', 'Label:', etc.
            # Using a regex to remove anything before a colon if present at the start
            lab_text = re.sub(r"^[^:：]*[:：]\s*", "", part).strip()
            
            # Remove trailing punctuation
            lab_text = re.sub(r"[.!?;,]+$", "", lab_text).strip()
            
            if lab_text:
                found_pred = lab_text.capitalize()
                break
        
        preds.append(found_pred)

    return preds