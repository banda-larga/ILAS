[![Open Streamlit in HuggingFace spaces](https://img.shields.io/badge/Streamlit-Open%20in%20Spaces-blueviolet)](https://huggingface.co/spaces/ARTeLab/ARTeLab-SummIT)

# Summarization with T5 and Mbart

This repo contains the code for the experiments on training T5 and MBart models on italian language.

## Models & Results

<table>
    <thead>
        <tr>
            <th>Datsset</th>
            <th>Model</th>
            <th>Rouge 1</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3><a href="https://huggingface.co/datasets/ARTeLab/mlsum-it">MLSum</a></td>
            <td rowspan=2><a href="https://huggingface.co/ARTeLab/it5-summarization-mlsum">IT5-Base</a></td>
        </tr>
        <tr>
            <td >19.29</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/ARTeLab/mbart-summarization-mlsum">MBart</a></td>
            <td><b> 19.35</b></td>
        </tr>
        <tr>
            <td rowspan=3><a href="https://huggingface.co/datasets/ARTeLab/fanpage">Fanpage</a></td>
            <td rowspan=2><a href="https://huggingface.co/ARTeLab/it5-summarization-fanpage">IT5-Base</a></td>
        </tr>
        <tr>
            <td >33.83</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/ARTeLab/mbart-summarization-fanpage">MBart</a></td>
            <td> <b>36.50</b></td>
        </tr>
        <tr>
             <td rowspan=3><a href="https://huggingface.co/datasets/ARTeLab/ilpost">IlPost</a></td>
            <td rowspan=2><a href="https://huggingface.co/ARTeLab/it5-summarization-ilpost">IT5-Base</a></td>
        </tr>
        <tr>
            <td >33.78</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/ARTeLab/mbart-summarization-ilpost">MBart</a></td>
            <td> <b>39.91</b></td>
        </tr>
    </tbody>
</table>



## Datasets and models

Datasets and models are available on Huggingface on [ARTeLab](https://huggingface.co/ARTeLab).

## Eval Pegasus with translations
We used [google/pegasus-cnn_dailymail](https://huggingface.co/google/pegasus-cnn_dailymail) and [google/pegasus-xsum](https://huggingface.co/google/pegasus-xsum) as existing comparisons by translating the input to english with [Helsinki-NLP/opus-mt-it-en](https://huggingface.co/Helsinki-NLP/opus-mt-it-en) and the output to intalian with [Helsinki-NLP/opus-mt-en-it](https://huggingface.co/Helsinki-NLP/opus-mt-en-it).

```
CUDA_VISIBLE_DEVICES="2" nohup python src/metrics_huggingface_eng_model.py \
        --model google/pegasus-cnn_dailymail \
        --path "./Data/IlPost/test.csv" \
        > logs/pegasus-cnn_dailymail_ilpost.log  2>&1 &
```

## Eval Our models (or any italian summarization models from HugginFace)
It is possible to use this script to run a trained model on a custom file.csv to make simple comparisons.
```
python src/metrics_huggingface_it_model.py \
        --path ./Data/MLSum/test.csv \
        --batch-size 5 \
         --model ARTeLab/it5-summarization-fanpage
```

## Train
* Install requirements
```
pip install -r requirements-torch.txt
```
* train it5 from [gsarti/it5-base](https://huggingface.co/gsarti/it5-base) pretrained

```
# we use a Nvidia RTX 5000 with 16GB of RAM

CUDA_VISIBLE_DEVICES="0,1,2" nohup python src/run_summarization.py \     
        --output_dir /home/super/Models/summarization_mlsum2 \
        --model_name_or_path gsarti/it5-base \
        --tokenizer_name gsarti/it5-base \
        --train_file "./Data/MLSum/train.csv" \
        --validation_file "./MLSum/MLSum/val.csv" \
        --test_file "./Data/MLSum/test.csv" \
        --do_train --do_eval --do_predict \
        --logging_dir tensorboard/mlsum2 \
        --source_prefix "summarize: " \
        --predict_with_generate \
        --num_train_epochs 4 \
        --per_device_train_batch_size 2 \ 
        --per_device_eval_batch_size 2 \ 
        --overwrite_output_dir \
        --save_steps 500 \
        --save_total_limit 3 \
        --save_strategy="steps" \
        --max_source_length 512 --max_target_length 64 \
        > logs/mlsum2.log  2>&1 &
```

* train MBart from [facebook/mbart-large-cc25](https://huggingface.co/facebook/mbart-large-cc25) pretrained:

```
# we use a Nvidia RTX 5000 with 16GB of RAM
PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:25" CUDA_VISIBLE_DEVICES=0 nohup python src/run_summarization_mbart.py args.json > logs/mbart-fanpage2.log 2>&1 &
```

# Citation

More details and results in [published work](https://www.mdpi.com/2078-2489/13/5/228)

```
@Article{info13050228,
    AUTHOR = {Landro, Nicola and Gallo, Ignazio and La Grassa, Riccardo and Federici, Edoardo},
    TITLE = {Two New Datasets for Italian-Language Abstractive Text Summarization},
    JOURNAL = {Information},
    VOLUME = {13},
    YEAR = {2022},
    NUMBER = {5},
    ARTICLE-NUMBER = {228},
    URL = {https://www.mdpi.com/2078-2489/13/5/228},
    ISSN = {2078-2489},
    ABSTRACT = {Text summarization aims to produce a short summary containing relevant parts from a given text. Due to the lack of data for abstractive summarization on low-resource languages such as Italian, we propose two new original datasets collected from two Italian news websites with multi-sentence summaries and corresponding articles, and from a dataset obtained by machine translation of a Spanish summarization dataset. These two datasets are currently the only two available in Italian for this task. To evaluate the quality of these two datasets, we used them to train a T5-base model and an mBART model, obtaining good results with both. To better evaluate the results obtained, we also compared the same models trained on automatically translated datasets, and the resulting summaries in the same training language, with the automatically translated summaries, which demonstrated the superiority of the models obtained from the proposed datasets.},
    DOI = {10.3390/info13050228}
}
```
