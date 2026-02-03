import pandas as pd

template_config_diabetes = {
    'pre': {
    }
}

template_config_blood = {
    'pre': {
    }
}

template_config_heart = {
    'pre': {
        'Sex': lambda x: 'male' if x == 'M' else 'female',
        'ChestPainType': lambda x: chest_paint_types_list[x],
        'FastingBS': lambda x: 'yes' if x == 1 else 'no',
        'ExerciseAngina': lambda x: 'yes' if x == 'Y' else 'no',
        'ST_Slope': lambda x: st_slopes[x],
        'RestingECG': lambda x: rest_ecg_results[x]
    }
}

template_config_calhousing = {
    'pre': {
    }
}

template_config_car = {
    'pre': {
        'buying': lambda x: prices_dict[x],
        'maint': lambda x: maint_dict[x],
        'doors': lambda x: doors_dict[x],
        'persons': lambda x: persons_dict[x],
        'lug_boot': lambda x: lug_boot_dict[x],
        'safety': lambda x: safety_dict[x]
    }
}

template_config_student = {
    'pre': {
    }
}

template_config_employee = {
    'pre': {
        'Education': lambda x: x.lower(),
        'City': lambda x: x,
        'PaymentTier': lambda x: payment_tier_dict[x],
        'Gender': lambda x: x.lower(),
        'EverBenched': lambda x: 'yes' if x.lower() == 'yes' else 'no',
    }
}

template_config_jungle = {
    'pre': {
        'white_piece0_file': lambda x: f"{int(x)}",
        'white_piece0_rank': lambda x: f"{int(x)}",
        'black_piece0_file': lambda x: f"{int(x)}",
        'black_piece0_rank': lambda x: f"{int(x)}",
    }
}

template_config_health = {
    'pre': {
    }
}

template_config_bank = {
    'pre': {
    }
}

template_config_creditg = {
    'pre': {
        'checking_status': lambda x: checking_status_dict[x],
        'credit_history': lambda x: credit_history_dict[x],
        'purpose': lambda x: purpose_dict[x],
        'savings_status': lambda x: savings_status_dict[x],
        'employment': lambda x: employment_dict[x],
        'personal_status': lambda x: personal_status_dict[x],
        'other_parties': lambda x: other_parties_dict[x],
        'property_magnitude': lambda x: property_magnitude_dict[x],
        'job': lambda x: job_dict[x],
        'own_telephone': lambda x: own_telephone_dict[x]
    }
}

template_config_income = {
    'pre': {
        'race': lambda r: None if r.lower() == 'other' else r,
        'marital_status': lambda ms: 'married' if ms.lower().startswith('married-') else
        ('never married' if ms.lower() == 'never-married' else ms.lower()),
        'native_country': lambda nc: 'United States' if nc in ['United-States', 'Outlying-US(Guam-USVI-etc)']
        else (None if pd.isna(nc) else ('South Korea' if nc.lower() == 'South' else nc)),
        'occupation': lambda o: occupation_dict_list.get(o, ''),
        'workclass': lambda w: workclass_dict_list.get(w, ''),
        'education': lambda e: education_dict_list.get(e)
    },
}

########################################################################################################################
# diabetes 8c0d size:768 binary-classification **** Some attributes contain many 0 values, some of which are meaningless
########################################################################################################################
# Used descriptions from: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
# and https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2245318/pdf/procascamc00018-0276.pdf

diabetes_task_description = "This dataset contains diagnostic measurements taken from the National Institute of Diabetes and Digestive and Kidney Diseases, with the objective of predicting whether a person has diabetes or not."

diabetes_role_prompt = "You are an expert in medical data analysis"
diabetes_answer_prompt = "Has not diabetes[0] or has diabetes[1]?"
diabetes_numerical_indices = [0, 1, 2, 3, 4, 5, 6, 7]

diabetes_feature_names = [
    ('Pregnancies', 'Number of times pregnant'),
    ('Glucose', 'Plasma glucose concentration at 2 hours in an oral glucose tolerance test (GTT)'),
    ('BloodPressure', 'Diastolic blood pressure (mm Hg)'),
    ('SkinThickness', 'Triceps skin fold thickness (mm)'),
    ('Insulin', '2-hour serum insulin (mu U/ml)'),
    ('BMI', 'Body mass index'),
    ('DiabetesPedigreeFunction', 'Diabetes pedigree function'),
    ('Age', 'Age')
]

except_numerical_value_diabetes = [] # Used to prevent numerical features from being discretized. Some features are numeric in type but categorical in meaning, and are not marked as such in the dataset file.


########################################################################################################################
# blood 4c0d size:748 binary-classification
########################################################################################################################
# Use description from: https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center

blood_task_description = "This dataset is taken from the Blood Transfusion Service Center, with the objective of predicting whether a person returned for another donation or not."

blood_role_prompt = "You are an expert in blood donation analysis"
blood_answer_prompt = "Will not donate[0] or will donate[1]?"
blood_numerical_indices = [0, 1, 2, 3]

blood_feature_names = [
    ('recency', 'Months since last donation'),
    ('frequency', 'Total number of donation'),
    ('monetary', 'Total blood donated'),
    ('time', 'Months since first donation'),
]

except_numerical_value_blood = []


########################################################################################################################
# heart 5c6d size:918 binary-classification **** Some attributes contain many 0 values, some of which are meaningless
########################################################################################################################
# Used descriptions from: https://www.kaggle.com/code/azizozmen/heart-failure-predict-8-classification-techniques

heart_task_description = "This dataset contains diagnostic measurements about individuals, with the objective of predicting whether a person has heart disease or not."

heart_role_prompt = "You are an expert in medical data analysis"
heart_answer_prompt = "Has not heart disease[0] or has heart disease[1]?"
heart_numerical_indices = []

heart_feature_names = [
    ('Age', 'Age of the patient'),
    ('Sex', 'Sex of the patient'),
    ('ChestPainType', 'Chest pain type'),
    ('RestingBP', 'Resting blood pressure'),
    ('Cholesterol', 'Serum cholesterol in mg/dl'),
    ('FastingBS', 'Fasting blood sugar > 120 mg/dl'),
    ('RestingECG', 'Resting electrocardiographic results'),
    ('MaxHR', 'Maximum heart rate achieved'),
    ('ExerciseAngina', 'Exercise induced angina'),
    ('Oldpeak', 'ST depression induced by exercise relative to rest'),
    ('ST_Slope', 'Slope of the peak exercise ST segment'),
]

chest_paint_types_list = {'TA': 'typical angina', 'ATA': 'atypical angina', 'NAP': 'non-anginal pain', 'ASY': 'asymptomatic'}
rest_ecg_results = {
    'Normal': 'normal',
    'ST': 'ST-T wave abnormality',
    'LVH': 'probable or definite left ventricular hypertrophy'
}
st_slopes = {'Up': 'upsloping', 'Flat': 'flat', 'Down': 'downsloping'}

except_numerical_value_heart = ['FastingBS']

########################################################################################################################
# calhousing 8c0d size:20640 binary-classification
########################################################################################################################
# Use description from: Pace and Barry (1997), "Sparse Spatial Autoregressions", Statistics and Probability Letters.

calhousing_task_description = "This dataset contains housing information per block group from the 1990 California Census, with the objective of predicting whether the house value is above the median or not."

calhousing_role_prompt = "You are an expert in real estate analysis"
calhousing_answer_prompt = "Below median value[0] or above median value[1]?"
calhousing_numerical_indices = [0, 1, 2, 3, 4, 5, 6, 7]

calhousing_feature_names = [
    ('median_income', 'median income'),
    ('housing_median_age', 'median age'),
    ('total_rooms', 'total rooms'),
    ('total_bedrooms', 'total bedrooms'),
    ('population', 'population'),
    ('households', 'households'),
    ('latitude', 'latitude'),
    ('longitude', 'longitude'),
]

except_numerical_value_calhousing = []

########################################################################################################################
# car 0c6d size:1728 multi-classification 
########################################################################################################################

car_task_description = "This dataset contains information about cars, with the objective of predicting the type of user's decision to buy this car: Unacceptable, Acceptable, Good, Very good."

car_role_prompt = "You are an expert in car evaluation"
car_answer_prompt = "Unacceptable[0], Acceptable[1], Good[2] or Very good[3]?"
car_numerical_indices = []

car_feature_names = [
    ('buying', 'Buying price'),
    ('maint', 'Maintenance price'),
    ('doors', 'Number of doors'),
    ('persons', 'Capacity in terms of persons to carry'),
    ('lug_boot', 'Size of luggage boot'),
    ('safety', 'Estimated safety of the car')
]

prices_dict = {'vhigh': 'very high', 'high': 'high', 'med': 'medium', 'low': 'low'}
maint_dict = {'vhigh': 'very high', 'high': 'high', 'med': 'medium', 'low': 'low'}
doors_dict = {'2': 'two', '3': 'three', '4': 'four', '5more': 'five or more'}
persons_dict = {'2': 'two', '4': 'four', 'more': 'more than four'}
lug_boot_dict = {'big': 'big', 'med': 'medium', 'small': 'small'}
safety_dict = {'high': 'high', 'med': 'medium', 'low': 'low'}

except_numerical_value_car = []

########################################################################################################################
# student: 6c0d size:2000 multi-classification
########################################################################################################################
# Use description from: https://www.kaggle.com/datasets/steve1215rogg/student-lifestyle-dataset

student_task_description = "This dataset contains detailed daily lifestyle and academic performance (GPA) for 2,000 students, with the objective of predicting the stress level of a student: High, Moderate, Low."

student_role_prompt = "You are an expert in student lifestyle data analysis"
student_answer_prompt = "High stress[0], moderate stress[1] or low stress[2]?"
student_numerical_indices = [0,1,2,3,4,5]

student_feature_names = [
    ('Study_Hours_Per_Day', 'Study time'),
    ('Extracurricular_Hours_Per_Day', 'Extracurricular time'),
    ('Sleep_Hours_Per_Day', 'Sleep time'),
    ('Social_Hours_Per_Day', 'Social time'),
    ('Physical_Activity_Hours_Per_Day', 'Physical activity time'),
    ('GPA', 'GPA')
]

except_numerical_value_student = []

########################################################################################################################
# Employee 3c5d size:4653 binary-classification
########################################################################################################################
# Use description from: https://www.kaggle.com/datasets/tawfikelmetwally/employee-dataset/data

employee_task_description = "This dataset contains information about employees in a company, with the objective of predicting whether an employee will leave the company or not."

employee_role_prompt = "You are an expert in human resources analysis"
employee_answer_prompt = "Stay[0] or Leave[1]?"
employee_numerical_indices = [1, 4, 7]

employee_feature_names = [
    ('Education', 'Educational qualification'),
    ('JoiningYear', 'Year of joining the company'),
    ('City', 'City where the employee is based or works'),
    ('PaymentTier', 'Salary tier'),
    ('Age', 'Age'),
    ('Gender', 'Gender'),
    ('EverBenched', 'If the employee has ever been temporarily without assigned work'),
    ('ExperienceInCurrentDomain', 'Years of experience in current domain')
]

payment_tier_dict = {
    1: 'low',
    2: 'medium',
    3: 'high'
}

except_numerical_value_employee = ['PaymentTier']

########################################################################################################################
# jungle 6c0d size:44819 binary-classification
########################################################################################################################
# Use description from: https://arxiv.org/abs/1604.07312

jungle_task_description = "This dataset contains 44,819 two pieces end game positions in the Dou Shou Qi game, with the objective of predicting whether the white player win this two pieces endgame of Jungle Chess or not."

jungle_role_prompt = "You are an expert of Dou Shou Qi game."
jungle_answer_prompt = "The black player win[0] or the white player win[1]?"
jungle_numerical_indices = [0,1,2,3,4,5]

jungle_feature_names = [
    ('white_piece0_strength', 'white piece strength'),
    ('white_piece0_file', 'white piece file'),
    ('white_piece0_rank', 'white piece rank'),
    ('black_piece0_strength', 'black piece strength'),
    ('black_piece0_file', 'black piece file'),
    ('black_piece0_rank', 'black piece rank')
]

except_numerical_value_jungle = ['white_piece0_file', 'white_piece0_rank', 'black_piece0_file', 'black_piece0_rank' ] # Coordinate information is not considered a numerical feature

########################################################################################################################
# Health 6c0d size:1014 multi-classification 
########################################################################################################################
# Use description from: https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data

health_task_description = "This dataset contains maternal health measurements for pregnant women in rural Bangladesh collected via an IoT-based risk monitoring system, with the objective of predicting the maternal health risk level: High, Mid, low."

health_role_prompt = "You are an expert in medical data analysis."
health_answer_prompt = "High risk[0], mid risk[1] or low risk[2]?"
health_numerical_indices = [0,1,2,3,4,5]

health_feature_names = [
    ('Age', 'Age'),
    ('SystolicBP', 'Systolic blood pressure'),
    ('DiastolicBP', 'Diastolic blood pressure'),
    ('BS', 'Blood sugar level'),
    ('BodyTemp', 'Body temperature'),
    ('HeartRate', 'Heart rate')
]

except_numerical_value_health = []

########################################################################################################################
# bank 7c9d size:45211 binary-classification
########################################################################################################################
# Use description from: https://archive.ics.uci.edu/ml/datasets/bank+marketing
# and https://www.openml.org/search?type=data&sort=runs&id=1461&status=active

bank_task_description = 'This dataset contains information of a direct marketing campaign from a Portugese banking institution, with the objective of predicting whether this client will subscribe to a term deposit or not'

bank_role_prompt = "You are an expert in bank marketing analysis"
bank_answer_prompt = "No[0] or Yes[1]?"
bank_numerical_indices = [0, 5, 9, 11, 12, 13, 14]

bank_feature_names = [
    ('age', 'age'),
    ('job', 'type of job'),
    ('marital', 'marital status'),
    ('education', 'education'),
    ('default', 'has credit in default?'),
    ('balance', 'average yearly balance'),
    ('housing', 'has housing loan?'),
    ('loan', 'has personal loan?'),
    ('contact', 'contact communication type'),
    ('day', 'last contact day of the month'),
    ('month', 'last contact month of year'),
    ('duration', 'last contact duration'),
    ('campagin', 'number of contacts performed during this campaign and for this client'),
    ('pdays', 'number of days that passed by after the client was last contacted from a previous campaign'),
    ('previous', 'number of contacts performed before this campaign and for this client'),
    ('poutcome', 'outcome of the previous marketing campaign'),
]

except_numerical_value_bank = []

########################################################################################################################
# creditg 7c13d size:1000 binary-classification 
########################################################################################################################
# Use description from: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

creditg_task_description = "This dataset contains financial and credit-related information, with the objective of predicting whether this applicant receive a credit"

creditg_role_prompt = "You are an expert in credit risk analysis"
creditg_answer_prompt = "Bad credit[0] or Good credit[1]?"
creditg_numerical_indices = [1, 4, 7, 10, 12, 15, 17]

creditg_feature_names = [
    ('checking_status', 'Status of existing checking account'),
    ('duration', 'Duration'),
    ('credit_history', 'Credit history'),
    ('purpose', 'Purpose'),
    ('credit_amount', 'Credit amount'),
    ('savings_status', 'Savings account/bonds'),
    ('employment', 'Present employment since'),
    ('installment_commitment', 'Installment rate in percentage of disposable income'),
    ('personal_status', 'Personal status and sex'),
    ('other_parties', 'Other debtors / guarantors'),
    ('residence_since', 'Present residence since'),
    ('property_magnitude', 'Property'),
    ('age', 'Age'),
    ('other_payment_plans', 'Other installment plans'),
    ('housing', 'Housing'),
    ('existing_credits', 'Number of existing credits'),
    ('job', 'Job'),
    ('num_dependents', 'Number of people being liable to provide maintenance for'),
    ('own_telephone', 'Telephone'),
    ('foreign_worker', 'Foreign worker')
]

checking_status_dict = {'<0': 'less than 0 DM', '0<=X<200': 'between 0 and 200 DM', '>=200': 'more than 200 DM', 'no checking': 'no checking account'}
credit_history_dict = {'no credits/all paid': 'no credits taken or all credits paid back duly', 'all paid': 'all credits at this bank paid back duly', 'existing paid': 'existing credits paid back duly till now', 'delayed previously': 'delay in paying off in the past', 'critical/other existing credit': 'critical account or other credits existing (not at this bank)'}
purpose_dict = {'new car': 'car (new)', 'used car': 'car (used)', 'furniture/equipment': 'furniture or equipment', 'radio/tv': 'radio or television', 'domestic appliance': 'domestic appliances', 'repairs': 'repairs', 'education': 'education', 'retraining': 'retraining', 'business': 'business', 'other': 'others'}
savings_status_dict = {'<100': 'less than 100 DM', '100<=X<500': 'between 100 and 500 DM', '500<=X<1000': 'between 500 and 1000 DM', '>=1000': 'more than 1000 DM', 'no known savings': 'unknown/ no savings account'}
employment_dict = {'unemployed': 'unemployed', '<1': 'less than 1 year', '1<=X<4': 'between 1 and 4 years', '4<=X<7': 'between 4 and 7 years', '>=7': 'more than 7 years'}
personal_status_dict = {'female div/dep/mar': 'female divorced or separated or married', 'male div/sep': 'male divorced or separated', 'male mar/wid': 'male married or widowed', 'male single': 'male single'}
other_parties_dict = {'none': 'none', 'co applicant': 'co-applicant', 'guarantor': 'guarantor'}
property_magnitude_dict = {'car': 'car or other, not in attribute 6', 'life insurance': 'building society savings agreement or life insurance', 'no known property': 'unknown or no property', 'real estate': 'real estate'}
job_dict = {'high qualif/self emp/mgmt': 'management or self-employed or highly qualified employee or officer', 'skilled': 'skilled employee or official', 'unemp/unskilled non res': 'unemployed or unskilled - non-resident', 'unskilled resident': 'unskilled - resident'}
own_telephone_dict = {'none': 'none', 'yes': 'yes and registered under the customers name'}

except_numerical_value_creditg = []

########################################################################################################################
# income 4c8d size:48842 binary-classification
########################################################################################################################
income_task_description = "This dataset contains demographic, employment, and financial information extracted from the 1994 US Census database, with the objective of predicting whether a person earn more than 50k dollars per year."

income_role_prompt = "You are an expert in financial analysis."
income_answer_prompt = "Less than 50k dollars[0] or more than 50k dollars[1]?"
income_numerical_indices = []

income_feature_names = [
    ('age', 'Age'),
    ('workclass', 'Work class'),
    ('education', 'Education'),
    ('marital_status', 'Marital status'),
    ('occupation', 'Occupation'),
    ('relationship', 'Relation to head of the household'),
    ('race', 'Race'),
    ('sex', 'Sex'),
    ('capital_gain', 'Capital gain last year'),
    ('capital_loss', 'Capital loss last year'),
    ('hours_per_week', 'Work hours per week'),
    ('native_country', 'Native country')
]

# Compiled from https://www2.census.gov/programs-surveys/demo/guidance/industry-occupation/1990-census-sic-codes.pdf and
# https://www2.census.gov/programs-surveys/demo/guidance/industry-occupation/2002-census-occupation-codes.xls
occupation_dict = {
    'Tech-support': 'in the technology and support sector',
    'Craft-repair': 'in the craft and repair sector',
    'Other-service': 'in the service sector',
    'Sales': 'in the sales sector',
    'Exec-managerial': 'in execution and management',
    'Prof-specialty': 'in a professional specialty',
    'Handlers-cleaners': 'in the cleaning and maintenance sector',
    'Machine-op-inspct': 'as a machine operator and inspector',
    'Adm-clerical': 'in office and administrative support',
    'Farming-fishing': 'in the agriculture, forestry, and fisheries sector',
    'Transport-moving': 'in the transportation, communication, and other public utilities sector',
    'Priv-house-serv': 'in their private household',
    'Protective-serv': 'in the protective services sector',
    'Armed-Forces': 'in the armed forces'
}
occupation_dict_list = {
    'Tech-support': 'technology and support sector',
    'Craft-repair': 'craft and repair sector',
    'Other-service': 'service sector',
    'Sales': 'sales sector',
    'Exec-managerial': 'execution and management',
    'Prof-specialty': 'professional specialty',
    'Handlers-cleaners': 'cleaning and maintenance sector',
    'Machine-op-inspct': 'machine operator and inspector',
    'Adm-clerical': 'office and administrative support',
    'Farming-fishing': 'agriculture, forestry, and fisheries sector',
    'Transport-moving': 'transportation, communication, and other public utilities sector',
    'Priv-house-serv': 'private household',
    'Protective-serv': 'protective services sector',
    'Armed-Forces': 'armed forces'
}
workclass_dict = {
    'Private': 'as a private sector employee',
    'Local-gov': 'for the local government',
    'State-gov': 'for the state government',
    'Federal-gov': 'for the federal government',
    'Self-emp-not-inc': 'as an owner of a non-incorporated business, professional practice, or farm',
    'Self-emp-inc': 'as a an owner of a incorporated business, professional practice, or farm',
    'Without-pay': 'without pay in a for-profit family business or farm',
    'Never-worked': 'never worked',
}
workclass_dict_list = {
    'Private': 'private sector employee',
    'Local-gov': 'local government',
    'State-gov': 'state government',
    'Federal-gov': 'federal government',
    'Self-emp-not-inc': 'owner of a non-incorporated business, professional practice, or farm',
    'Self-emp-inc': 'owner of a incorporated business, professional practice, or farm',
    'Without-pay': 'without pay in a for-profit family business or farm',
    'Never-worked': 'never worked',
}
# From: https://www.census.gov/content/dam/Census/library/publications/2007/dec/10_education.pdf
education_dict = {
    'Doctorate': 'has a doctoral degree',
    'Prof-school': 'has a professional degree',
    'Masters': 'has a master\'s degree',
    'Bachelors': 'has a bachelor\'s degree',
    'Assoc-acdm': 'has an associate\'s degree',
    'Assoc-voc': 'went to college for one or more years without a degree',
    'Some-college': 'went to college for less than one year',
    'HS-grad': 'is a high school graduate',
    '12th': 'finished 12th class without diploma',
    '11th': 'finished 11th class',
    '10th': 'finished 10th class',
    '9th': 'finished 9th class',
    '7th-8th': 'finished 8th class',
    '5th-6th': 'finished 6th class',
    '1st-4th': 'finished 4th class',
    'Preschool': 'completed no schooling'
}
education_dict_list = {
    'Doctorate': 'doctoral degree',
    'Prof-school': 'professional degree',
    'Masters': 'master\'s degree',
    'Bachelors': 'bachelor\'s degree',
    'Assoc-acdm': 'associate\'s degree',
    'Assoc-voc': 'college for one or more years without a degree',
    'Some-college': 'college for less than one year',
    'HS-grad': 'high school graduate',
    '12th': 'finished 12th class without diploma',
    '11th': 'finished 11th class',
    '10th': 'finished 10th class',
    '9th': 'finished 9th class',
    '7th-8th': 'finished 8th class',
    '5th-6th': 'finished 6th class',
    '1st-4th': 'finished 4th class',
    'Preschool': 'no schooling'
}
# From https://www.census.gov/programs-surveys/cps/technical-documentation/subject-definitions.html#householder
relationship_dict = {
    'Wife': 'and is the wife of the head of the household',
    'Own-child': 'and is a child of the head of the household',
    'Husband': 'and is the husband of the head of the household',
    'Not-in-family': 'and is not in a family',
    'Other-relative': 'and is an other relative of the head of the household',
    'Unmarried': 'and is not married to the head of the household'
}
relationship_dict_list = {
    'Wife': 'wife',
    'Own-child': 'own child',
    'Husband': 'husband',
    'Not-in-family': 'not in a family',
    'Other-relative': 'other relative',
    'Unmarried': 'unmarried'
}

except_numerical_value_income = []