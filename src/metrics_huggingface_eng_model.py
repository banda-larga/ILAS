import argparse

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

parser = argparse.ArgumentParser("Metric")
parser.add_argument('--path', type=str, default='../Data/MLSum/test.csv')
parser.add_argument('--model', type=str, default='google/pegasus-xsum')
parser.add_argument('--batch-size', type=int, default=5)

args = parser.parse_args()


def main():
    device = test_gpu_settings()

    print('data reading...')
    df = pd.read_csv(args.path, )
    df.info()

    print("loading it->en model...")
    tokenizer_it_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-it-en")
    model_it_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-it-en")
    # if device == 'cuda':
    #     model_it_en = torch.nn.DataParallel(model_it_en)
    model_it_en = model_it_en.to(device)
    model_it_en.eval()

    print("loading en->it model...")
    tokenizer_en_it = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-it")
    model_en_it = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-it")
    # if device == 'cuda':
    #     model_en_it = torch.nn.DataParallel(model_en_it)
    model_en_it = model_en_it.to(device)
    model_en_it.eval()

    print("loading summarizer...")
    tokenizer_sum = AutoTokenizer.from_pretrained(args.model)
    model_sum = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    # if device == 'cuda':
    #     model_sum = torch.nn.DataParallel(model_sum)
    model_sum = model_sum.to(device)
    model_sum.eval()

    # Metric
    metric = load_metric("rouge")

    metric_accumulator = {
        'rouge1': [],
        'rouge2': [],
        'rougeL': [],
        'rougeLsum': [],
        'gen_len': []
    }

    # iterate on data
    for chunk in tqdm(chunker(df, args.batch_size)):
        it_source = list(chunk['source'].values)
        it_target = list(chunk['target'].values)

        # translate it->eng
        encoded_it_en = tokenizer_it_en(it_source, return_tensors="pt", padding=True, truncation=True, max_length=512)
        encoded_it_en = encoded_it_en.to(device)
        with torch.no_grad():
            output = model_it_en.generate(**encoded_it_en)
        en_source = tokenizer_it_en.batch_decode(output, skip_special_tokens=True)

        # summarize
        encoded_summary = tokenizer_sum(en_source, return_tensors="pt", padding=True, truncation=True, max_length=512)
        encoded_summary = encoded_summary.to(device)
        with torch.no_grad():
            output = model_sum.generate(**encoded_summary)
        summary = tokenizer_sum.batch_decode(output, skip_special_tokens=True)

        # translate en->it
        encoded_en_it = tokenizer_en_it(summary, return_tensors="pt", padding=True, truncation=True, max_length=512)
        encoded_en_it = encoded_en_it.to(device)
        with torch.no_grad():
            output = model_en_it.generate(**encoded_en_it)
        it_summary = tokenizer_en_it.batch_decode(output, skip_special_tokens=True)

        # print(len(it_summary), it_summary)

        # compute metric
        metrics = compute_metrics(it_summary, it_target, metric, output.cpu(), tokenizer_en_it)
        for k, v in metrics.items():
            metric_accumulator[k].append(v)

    for k, v in metric_accumulator.items():
        print(k, sum(v) / len(v))


def chunker(seq, size):
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]


def compute_metrics(decoded_preds, decoded_labels, metric, preds, tokenizer):
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def test_gpu_settings():
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print(f"Currently using GPU")
        return 'cuda'
    else:
        print("Currently using CPU")
        return 'cpu'


if __name__ == '__main__':
    main()
