import pandas as pd
import rouge_score
import rouge
import os
import torch
import sys
from tqdm.auto import tqdm
from typing import Callable, Optional
from dataclasses import dataclass, field
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import HfArgumentParser

@dataclass
class Arguments:
    model: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model for summarization"
        },
    )
    batch_size_t: Optional[int] = field(
        default=20,
        metadata={
            "help": "Batch_size for inference"
        },
    )
    batch_size_s: Optional[int] = field(
        default=4,
        metadata={
            "help": "Batch_size for inference"
        },
    )
    sample: Optional[int] = field(
        default=None,
        metadata={
            "help": "N. Samples for computing metrics"
        },
    )
    data_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input test file (a csv file)."
            }
        )

def word_len(text):
    words = text.split(" ")
    return len(words)

def compute_metrics(preds, labels, metric):
    # preds e labels lists
    result = metric.compute(predictions=preds, references=labels, use_stemmer=True)
    # few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def main():
    parser = HfArgumentParser([Arguments])
    model_args = parser.parse_args_into_dataclasses()[0]

    # args
    data_file = model_args.data_file
    batch_size = model_args.batch_size_t
    batch_size_sum = model_args.batch_size_s
    df = pd.read_csv(data_file)

    # sample if sample number is specified
    if model_args.sample is not None:
        sample = model_args.sample
        df = df.sample(n=sample, random_state=1)
        df = df.reset_index(drop=True)

    print("Dataframe:")
    print(df.head())
    print(f"\n[*] Data successfully loaded: {model_args.data_file}")

    '''
    Translate sources in italian
    '''

    # load model
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-it-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-it-en")
    model.eval()
    model.to('cuda')

    # get chunks
    sources = [df["source"].iloc[i] for i in range(df.shape[0])]
    print(f"[*] Data size: {len(sources)} summaries")
    input_chunks = [sources[i:i + batch_size] for i in range(0, len(sources), batch_size)]

    chunk_text = []
    print("[*] Translating chunks...")

    # translate
    progress_bar = tqdm(range(len(input_chunks)))
    for i in range(len(input_chunks)):
        encoded_input = tokenizer(input_chunks[i], return_tensors = "pt", padding = True, truncation = True).to('cuda')
        progress_bar.set_description(f"[*] Processing chunk {i}")
        with torch.no_grad():
            output = model.generate(**encoded_input)
        chunk = tokenizer.batch_decode(output, skip_special_tokens=True)
        progress_bar.update(1)
        chunk_text.extend(chunk)
    progress_bar.close()

    # create column and get avg length
    df["english_source"] = chunk_text
    lunghezze = [word_len(df.english_source.iloc[i]) for i in range(df.shape[0])]
    lunghezza_testi = sum(lunghezze)/len(lunghezze)
    print("[*] Num. words in translated source:", lunghezza_testi)

    '''
    Load model and summarize
    '''

    # get model for summarization
    sum_model = model_args.model
    # load model
    tokenizer = AutoTokenizer.from_pretrained(sum_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(sum_model)
    model.eval()
    model.to('cuda')
    print("\n[*] Pegasus successfully loaded, now summarizing...")

    # get chunks
    sources = [df["english_source"].iloc[i] for i in range(df.shape[0])]
    input_chunks = [sources[i:i + batch_size_sum] for i in range(0, len(sources), batch_size_sum)]

    # summarize
    chunk_text = []
    progress_bar = tqdm(range(len(input_chunks)), leave=True)
    for i in range(len(input_chunks)):
        encoded_input = tokenizer(input_chunks[i], return_tensors = "pt", padding=True, truncation=True).to('cuda')
        progress_bar.set_description(f"[*] Processing chunk {i}")
        with torch.no_grad():
            output = model.generate(
                **encoded_input
            )
        chunk = tokenizer.batch_decode(output, skip_special_tokens=True)
        progress_bar.update(1)
        chunk_text.extend(chunk)
    progress_bar.close()

    # create column and print updated df
    df['english_summary'] = chunk_text
    print("\nNow with English summaries:")
    print(df.head())

    # get avg length
    lunghezze = []
    lunghezze = [word_len(df.english_summary.iloc[i]) for i in range(df.shape[0])]
    lunghezza_testi = sum(lunghezze)/len(lunghezze)
    print("[*] Num. words in english summaries:", lunghezza_testi)

    '''
    Translate summaries in italian
    '''

    # load model
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-it")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-it")
    model.eval()
    model.to('cuda')

    # get chunks
    summaries = [df["english_summary"].iloc[i] for i in range(df.shape[0])]
    print(f"\n[*] Data size: {len(summaries)} summaries")
    input_chunks = [summaries[i:i + batch_size] for i in range(0, len(summaries), batch_size)]

    # translating
    chunk_text = []
    print("[*] Translating chunks...")
    progress_bar = tqdm(range(len(input_chunks)))
    for i in range(len(input_chunks)):
        encoded_input = tokenizer(input_chunks[i], return_tensors = "pt", padding = True, truncation = True).to('cuda')
        progress_bar.set_description(f"[*] Processing chunk {i}")
        with torch.no_grad():
            output = model.generate(**encoded_input)
        chunk = tokenizer.batch_decode(output, skip_special_tokens=True)
        progress_bar.update(1)
        chunk_text.extend(chunk)
    progress_bar.close()

    # create column and print updated df
    df["translated_summary"] = chunk_text
    print("\nNow with Translated summaries:")
    print(df.head())

    # compute metrics
    print("\n[*] Computing metrics")
    model_out = df["translated_summary"].tolist()
    reference = df["target"].tolist()
    # compute ROUGE metrics
    metric = load_metric("rouge")
    rouge_desc = ""
    rouge_metrics = compute_metrics(model_out, reference, metric)
    rouge_desc = " ".join([f"Eval {key}: {value} |" for key, value in rouge_metrics.items()])
    print(rouge_desc)

if __name__ == "__main__":
    main()
