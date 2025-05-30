# Convert old femr batch dataset to new input pairs of new femr dataset
# Input: database of all patients. 
# Output: a dataframe with birth_date, medical_tokens, time_tokens, patient_info tokens, patient_info time tokens, patient_id
# The ouput dataframe records each patients' medical events, time stamps corresponding to the medical events, patient informations, eg. birthdate, race, gender, etc.

import pandas as pd
import numpy as np
import os
import json
from femr.datasets import PatientDatabase
import datetime
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Extract the dataset from the femr database")
    parser.add_argument("--path_to_dataset", default = 'patient_dataset.csv', required=False, type=str, help="Path to new patient dataset to be saved")
    parser.add_argument('--path_to_patient_database', default = 'EHRSHOT_ASSETS/femr/extract', required=False, type=str, help="Path to the femr patient database")
    parser.add_argument('--path_to_medical_tokens', default = 'femr_vocab.json', required=False, type=str, help="Path to the femr medical tokens")
    parser.add_argument('--path_to_labels_dir', default = 'EHRSHOT_ASSETS/benchmark', required=False, type=str, help="Path to the benchmark labels")
    return parser.parse_args()

task_name_list =[
    "guo_los",
    "guo_readmission",
    "guo_icu",
    "new_hypertension",
    "new_hyperlipidemia",
    "new_pancan",
    "new_celiac",
    "new_lupus",
    "new_acutemi",
    "lab_thrombocytopenia",
    "lab_hyperkalemia",
    "lab_hyponatremia",
    "lab_anemia",
    "lab_hypoglycemia"
]
# 1, load the femr medical tokens, only keep the tokens that are with in the femr tokenizer
def load_medical_tokens(femr_token_dict_path = 'femr_vocab.json'):
    with open(femr_token_dict_path, 'r') as f:
        femr_dict = json.load(f)
    return femr_dict

# 2, load the patient database
def configure_database(path_to_database = "EHRSHOT_ASSETS/femr/extract"):
    return PatientDatabase(path_to_database)

# 3, load the patient labels and time
def load_patient_labels(path_to_labels_dir = "EHRSHOT_ASSETS/benchmark", task_dir = 'guo_los', label_file = 'labeled_patients.csv'):
    path_to_labels = os.path.join(path_to_labels_dir, task_dir, label_file)
    df_labels = pd.read_csv(path_to_labels)
    return df_labels

# 4, get the unique patient ids from a single label files. 
def get_unique_patient_ids(path_to_labels_dir = 'EHRSHOT_ASSETS/benchmark', task_dir = 'guo_los', label_file = 'labeled_patients.csv'):
    path_to_labels = os.path.join(path_to_labels_dir, task_dir, label_file)
    df_labels = pd.read_csv(path_to_labels)
    return list(df_labels['patient_id'].unique())

# 5, get the unique patient ids from all the label files.
def get_unique_patient_ids_all_tasks(path_to_labels_dir = 'EHRSHOT_ASSETS/benchmark', task_name_list = task_name_list, label_file = 'labeled_patients.csv'):
    patient_ids = []
    for task_name in task_name_list:
        current_patient_ids = get_unique_patient_ids(path_to_labels_dir, task_name, label_file)
        patient_ids.extend(current_patient_ids)
    return list(set(patient_ids))

# 6, given a patient id, extract the events, filter all the medical sequence that are in the femr tokenizer
def filter_medical_events(patient_id, database, femr_dict):
    patient_info = database[patient_id]
    assert patient_info.patient_id == patient_id
    event_list = database[patient_id].events
    patient_birthdate = database.get_patient_birth_date(patient_id)
    patient_birthdate = patient_birthdate.isoformat()

    medical_code_list = []
    time_list = []
    omop_person_info_list = []
    omop_time_info_list = []

    for i in range(len(event_list)):
        current_event = event_list[i]
        start_date = current_event.start
        start_date = start_date.isoformat()
        medical_code = current_event.code
        omop_table = current_event.omop_table
        if medical_code in femr_dict:
            medical_code_list.append(medical_code)
            time_list.append(start_date)
            assert start_date is not None
            assert medical_code is not None
        if omop_table == 'person':
            omop_person_info_list.append(medical_code)
            omop_time_info_list.append(start_date)
    
    patient = {}
    # patient['patient_id'] = patient_id
    patient['birth_date'] = patient_birthdate
    patient['medical_tokens'] = medical_code_list
    patient['time_tokens'] = time_list
    patient['person_info_tokens'] = omop_person_info_list
    patient['person_info_time_tokens'] = omop_time_info_list
    
    return patient

# 7, given a list of patient id, construct the dataset, return a dataframe with
def construct_dataset(patient_id_list, database, femr_dict, return_dataframe = True):
    dataset = {}
    for patient_id in tqdm(patient_id_list):
        patient_info = filter_medical_events(patient_id, database, femr_dict)
        dataset[patient_id] = patient_info
    if return_dataframe:
        dataset = pd.DataFrame(dataset).T
        dataset['patient_id'] = dataset.index
        dataset = dataset.reset_index(drop=True)
        return dataset
    else:
        return dataset

# 8, save the dataset with only patient id and 
def save_patient_dataset(patient_dataset, save_dir = 'new_femr_dataset/patient_info/'):
    for patient_id in tqdm(patient_dataset.keys()):
        patient_info = patient_dataset[patient_id]
        save_path = os.path.join(save_dir, str(patient_id) + '.json')
        with open(save_path, 'w') as f:
            json.dump(patient_info, f)

# 9, given a datatime string with format 
def convert_datetime_string_to_timestamp(datetime_string):
    date_format = '%Y-%m-%d %H:%M:%S'
    return datetime.datetime.strptime(datetime_string, date_format)

# 10, given patient's dataset, add the labels and the corresponding time points of a single medical task
def add_labels_to_dataset(dataset, path_to_labels_dir = 'EHRSHOT_ASSETS/benchmark', task_dir = 'guo_los', label_file = 'labeled_patients.csv'):
    path_to_labels = os.path.join(path_to_labels_dir, task_dir, label_file)
    df_labels = pd.read_csv(path_to_labels)
    for patient_id in tqdm(dataset.keys()):
        patient_info = dataset[patient_id]
        patient_info[task_dir] = {}

def main():
    args = parse_args()
    path_to_database = args.path_to_patient_database
    path_to_dataset = args.path_to_dataset
    path_to_medical_tokens = args.path_to_medical_tokens
    path_to_labels_dir = args.path_to_labels_dir

    femr_dict = load_medical_tokens(femr_token_dict_path=path_to_medical_tokens)
    database = configure_database(path_to_database)
    patient_id_list = get_unique_patient_ids_all_tasks(path_to_labels_dir=path_to_labels_dir)
    patient_dataset = construct_dataset(patient_id_list, database, femr_dict)
    save_patient_dataset(patient_dataset)


if __name__ == '__main__':
    main()