# preprocess the old femr dataset's label file. 
import pandas as pd
import numpy as np
import os
import json
from femr.datasets import PatientDatabase
import datetime
from tqdm import tqdm

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

def find_time_indices(patient_label_time_list, patient_all_time_list):
    results = []
    all_index = 0
    for label_time in patient_label_time_list:
        while all_index < len(patient_all_time_list) and patient_all_time_list[all_index] <= label_time:
            all_index += 1
        if all_index > 0:
            results.append(all_index - 1)
        else:
            raise ValueError("No time found")
        
    return results

def extract_patient_labels_index_by_task(patient_info_dir = 'new_femr_dataset/patient_info/', task_dir = 'EHRSHOT_ASSETS/benchmark/lab_anemia/', task_label_file = 'labeled_patients.csv'):
    df_labels = pd.read_csv(os.path.join(task_dir, task_label_file))
    patient_id_list = df_labels['patient_id'].unique()

    label_index_list = []

    for patient_id in tqdm(patient_id_list):
        patient_json_name = str(patient_id) + '.json'
        with open(patient_info_dir + patient_json_name) as f:
            patient_info = json.load(f)

        df_label_patient = df_labels[df_labels['patient_id'] == patient_id]

        patient_label_time_list = [datetime.datetime.fromisoformat(date) for date in df_label_patient['prediction_time'].values]

        patient_label_value_list = df_label_patient['value'].values

        assert len(patient_info['medical_tokens']) == len(patient_info['time_tokens'])

        patient_all_time_list = [datetime.datetime.fromisoformat(date) for date in patient_info['time_tokens']]

        label_index = find_time_indices(patient_label_time_list, patient_all_time_list)

        assert len(label_index) == len(patient_label_value_list)

        label_index_list.extend(label_index)

    df_labels_extracted = df_labels.copy()
    df_labels_extracted['index'] = label_index_list

    return df_labels_extracted

def main():
    labeled_dir = 'EHRSHOT_ASSETS/benchmark/'
    out_dir = 'new_femr_dataset/patient_label/'
    task_label_file = 'labeled_patients.csv'
    for task in tqdm(task_name_list):
        task_dir = os.path.join(labeled_dir, task)
        df_task = extract_patient_labels_index_by_task(patient_info_dir = 'new_femr_dataset/patient_info/', task_dir = task_dir, task_label_file = task_label_file)
        df_task.to_csv(os.path.join(out_dir, task + '.csv'), index = False)


if __name__ == "__main__":
    main()