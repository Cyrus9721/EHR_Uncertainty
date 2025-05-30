import femr.models.transformer
import torch
import femr.models.tokenizer
import femr.models.processor
import datetime

model_name = "StanfordShahLab/clmbr-t-base"

tokenizer = femr.models.tokenizer.FEMRTokenizer.from_pretrained(model_name)
batch_processor = femr.models.processor.FEMRBatchProcessor(tokenizer)

# Load model
model = femr.models.transformer.FEMRModel.from_pretrained(model_name)

patient_sample = {
    'patient_id': 30,
    'events': [{
        'time': datetime.datetime(2011, 5, 8),
        'measurements': [
            {'code': 'SNOMED/184099003'},
            {'code': 'Visit/IP'},
        ],
    },
    {
        'time': datetime.datetime(2012, 6, 9),
        'measurements': [
            {'code': 'Visit/OP'},
            {'code': 'SNOMED/3950001'}
        ],
    }]
}

patient_sample_0 = {
    'patient_id': 30,
    'events': [{
        'time': datetime.datetime(2011, 5, 8),
        'measurements': [
            {'code': 'SNOMED/184099003'},
            {'code': 'Visit/IP'},
        ],
    }]
}

raw_batch = batch_processor.convert_patient(patient_sample_0, tensor_type="pt")
batch = batch_processor.collate([raw_batch])

_, result = model(**batch)

print(result)

raw_batch = batch_processor.convert_patient(patient_sample, tensor_type="pt")
batch = batch_processor.collate([raw_batch])

_, result = model(**batch)

print(result)
