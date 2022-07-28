### Usage:

Only on GPU

It uses "Helsinki-NLP/opus-mt-it-en" for it-to-en translation, selected model for summarization, and then "Helsinki-NLP/opus-mt-en-it" for en-to-it translation.

**batch_size_t**: batch_size for translating 

**batch_size_s**: batch_size for summarizing

**sample**: num_examples, if None it will take all of the data

```console
dom@toretto:~$ python metrics_iten.py \
                        --model google/pegasus-xsum \
                        --batch_size_t 20 \
                        --batch_size_s 5 \
                        --sample 400 \
                        --data_file "path_to_file/test.csv"
```
