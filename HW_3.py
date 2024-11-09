import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering, default_data_collator, get_scheduler
from datasets import load_dataset
from accelerate import Accelerator, notebook_launcher
from huggingface_hub import Repository, get_full_repo_name, notebook_login
import evaluate
from tqdm.auto import tqdm
import numpy as np
import collections
import json

spoken_train = 'spoken_train-v1.1.json'
spoken_test = 'spoken_test-v1.1.json'
spoken_test_WER44 = 'spoken_test-v1.1_WER44.json'
spoken_test_WER54 = 'spoken_test-v1.1_WER54.json'

def reformat_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    
    examples = []
    for elem in data['data']:
        title = elem['title']

        for p in elem['paragraphs']:
            context = p['context']

            for qa in p['qas']:
                example = {}
                example['id'] = qa['id']
                example['title'] = title.strip()
                example['context'] = context.strip()
                example['question'] = qa['question'].strip()
                example['answers'] = {}
                example['answers']['answer_start'] = [answer["answer_start"] for answer in qa['answers']]
                example['answers']['text'] = [answer["text"] for answer in qa['answers']]
                examples.append(example)
    
    out_dict = {'data': examples}

    output = 'out_' + file
    with open(output, 'w') as f:
        json.dump(out_dict, f)

    return output

spoken_train = reformat_json(spoken_train)
spoken_test = reformat_json(spoken_test)
spoken_test_WER44 = reformat_json(spoken_test_WER44)
spoken_test_WER54 = reformat_json(spoken_test_WER54)

spoken_squad_dataset = load_dataset('json',
                                    data_files = { 'train': spoken_train,
                                                  'validation': spoken_test,         # NO NOISE: 22.73% WER
                                                  'test_WER44': spoken_test_WER44,   # NOISE V1: 44.22% WER
                                                  'test_WER54': spoken_test_WER54 }, # NOISE V2: 54.82% WER
                                    field = 'data')

model_name = "rein5/bert-base-uncased-finetuned-spoken-squad"
print("Model: " + model_name)

model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

length = 384 
stride = 128

def preprocess(examples):
    questions = [question.strip() for question in examples['question']]
    inputs = tokenizer(
        questions, 
        examples['context'],
        max_length = length,
        truncation = 'only_second',
        stride = stride, 
        return_overflowing_tokens = True,
        return_offsets_mapping=True, 
        padding = 'max_length'
    )

    offset_mapping = inputs.pop('offset_mapping')
    sample_map = inputs.pop('overflow_to_sample_mapping')
    answers = examples['answers']
    start_pos = []
    end_pos = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer['answer_start'][0]
        end_char = answer['answer_start'][0] + len(answer["text"][0])
        seq_id = inputs.sequence_ids(i)

        # find start and end of the text
        idx = 0
        while seq_id[idx] != 1: 
            idx += 1
        cstart = idx
        while seq_id[idx] == 1:
            idx += 1
        ctx_e = idx - 1

        if offset[cstart][0] > start_char or offset[ctx_e][1] < end_char:
            start_pos.append(0)
            end_pos.append(0)
        else:
            idx = cstart
            while idx <= ctx_e and offset[idx][0] <= start_char:
                idx += 1
            start_pos.append(idx - 1)

            idx = ctx_e
            while idx >= cstart and offset[idx][1] >= end_char:
                idx -= 1
            end_pos.append(idx + 1)
    
    inputs['start_positions'] = start_pos
    inputs['end_positions'] = end_pos
    return inputs

print("Preprocessing training data")
train_dataset = spoken_squad_dataset['train'].map(preprocess, batched = True, remove_columns=spoken_squad_dataset['train'].column_names)

def validate_examples(examples):
    questions = [q.strip() for q in examples['question']]
    inputs = tokenizer(
        questions, 
        examples['context'],
        max_length = length,
        truncation = 'only_second',
        stride = stride, 
        return_overflowing_tokens = True,
        return_offsets_mapping=True, 
        padding = 'max_length'
    )

    mapping = inputs.pop('overflow_to_sample_mapping')
    ex_id = []

    for i in range(len(inputs['input_ids'])):
        s_i = mapping[i]
        ex_id.append(examples["id"][s_i])

        sequence_ids = inputs.sequence_ids(i)
        offsets = inputs['offset_mapping'][i]
        inputs["offset_mapping"][i] = [ offset if sequence_ids[k] == 1 else None for k, offset in enumerate(offsets)]

    inputs['example_id'] = ex_id
    return inputs


print("Preprocessing test data")
validation_dataset = spoken_squad_dataset['validation'].map( validate_examples, batched = True, remove_columns=spoken_squad_dataset['validation'].column_names)

print("Preprocessing V1")
test_WER44 = spoken_squad_dataset['test_WER44'].map( validate_examples, batched = True, remove_columns=spoken_squad_dataset['test_WER44'].column_names)

print("Preprocessing V2")
test_WER54 = spoken_squad_dataset['test_WER54'].map( validate_examples, batched = True, remove_columns=spoken_squad_dataset['test_WER54'].column_names)

print(validation_dataset)

metric = evaluate.load("squad")

marker = 20
max_ans = 35

def compute_metrics(slogits, elogits, components, exes):
    exe_to_comp = collections.defaultdict(list)
    for idx, feature in enumerate(components): 
        exe_to_comp[feature["example_id"]].append(idx)
    
    predict = []
    for exe in tqdm(exes):
        exe_id = exe["id"]
        context = exe["context"]
        answers = []
        
        for feature_index in exe_to_comp[exe_id]: 
            start_logit = slogits[feature_index]
            end_logit = elogits[feature_index]
            offsets = components[feature_index]["offset_mapping"]
            
            start = np.argsort(start_logit)[-1: -marker - 1: -1].tolist()
            end = np.argsort(end_logit)[-1: -marker - 1: -1].tolist()
            for s in start: 
                for e in end: 
                    if offsets[s] is None or offsets[e] is None: 
                        continue
                    if e < s or e - s + 1 > max_ans: 
                        continue
                    
                    answer = {
                        "text": context[offsets[s][0] : offsets[e][1]],
                        "logit_score": start_logit[s] + end_logit[e]
                    }
                    answers.append(answer)
        
        if len(answers) > 0: 
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predict.append(
                {"id": exe_id, "prediction_text": best_answer["text"]}
            )
        else: 
            predict.append({"id": exe_id, "prediction_text": ""})
        
    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in exes]
    return metric.compute(predictions=predict, references=theoretical_answers)

train_dataset.set_format("torch")
valid = validation_dataset.remove_columns(["example_id", "offset_mapping"])
valid.set_format("torch")
test_WER44_mod = test_WER44.remove_columns(["example_id", "offset_mapping"])
test_WER44_mod.set_format("torch")
test_WER54_mod = test_WER54.remove_columns(["example_id", "offset_mapping"])
test_WER54_mod.set_format("torch")

print("Training Dataloader : ")
train_dataloader = DataLoader(
    train_dataset, 
    shuffle = True, 
    collate_fn = default_data_collator, 
    batch_size = 8
)

eval_dataloader = DataLoader( valid, collate_fn=default_data_collator, batch_size=8 )

test_WER44_dataloader = DataLoader( test_WER44_mod, collate_fn=default_data_collator, batch_size=8 )

test_WER54_dataloader = DataLoader( test_WER54_mod, collate_fn=default_data_collator, batch_size=8 )

output_dir = "bert-base-uncased-finetuned-spoken-squad"

upload_to_hub = False

def evaluate_model(model, dataloader, dataset, dataset_before_preprocessing, accelerator=None):
    if not accelerator: 
        accelerator = Accelerator(mixed_precision='fp16')
        model, dataloader = accelerator.prepare(
            model, dataloader
        )
    
    model.eval()
    slogits = []
    elogits = []
    for batch in tqdm(dataloader):
        with torch.no_grad(): 
            outputs = model(**batch)

        slogits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
        elogits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

    slogits = np.concatenate(slogits)
    elogits = np.concatenate(elogits)
    slogits = slogits[: len(dataset)]
    elogits = elogits[: len(dataset)]

    metrics = compute_metrics( slogits, elogits, dataset, dataset_before_preprocessing )
    return metrics

def train_model(model=model, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, epochs = 1):
    training_steps = epochs * len(train_dataloader)

    accelerator = Accelerator(mixed_precision='fp16')
    optimizer = AdamW(model.parameters(), lr = 2e-5)

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=training_steps,
    )

    status_ind = tqdm(range(training_steps))

    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            status_ind.update(1)

        # Epoch
        accelerator.print("Evaluation...")
        metrics = evaluate_model(model, eval_dataloader, validation_dataset, spoken_squad_dataset['validation'], accelerator)
        print(f"epoch {epoch}:", metrics)

print("Computing Test : ")
test = evaluate_model(model, eval_dataloader, validation_dataset, spoken_squad_dataset['validation'])
print("Computing Test V1 : ")
test_v1 = evaluate_model(model, test_WER44_dataloader, test_WER44, spoken_squad_dataset['test_WER44'])
print("Computing Test V2 : ")
test_v2 = evaluate_model(model, test_WER54_dataloader, test_WER54_dataset, spoken_squad_dataset['test_WER54'])

print("************** OUTPUT **************")
print("Test     - best match: " + str(test['exact_match']) + ", computed F1 Value: " + str(test['f1']))
print("Test V1  - best match: " + str(test_v1['exact_match']) + ", computed F1 Value: " + str(test_v1['f1']))
print("Test V2  - best match: " + str(test_v2['exact_match']) + ", computed F1 Value: " + str(test_v2['f1']))
print("************************************")