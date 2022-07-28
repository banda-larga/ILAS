#!/usr/bin/env bash
# IT5
BATCH_SIZE=1
for DATASET in "fanpage" "MLSum" "IlPost"
do
  for MODEL in "mlsum" "ilpost" "fanpage"
  do
    python src/metrics_huggingface_it_model.py --path ./Data/${DATASET}/test.csv --batch-size ${BATCH_SIZE} --model ARTeLab/it5-summarization-${MODEL} > logs/predbeams5_${DATASET}_IT5_${MODEL}.log
  done
done

# MBart
BATCH_SIZE=1
for DATASET in "fanpage" "MLSum" "IlPost"
do
  for MODEL in "mlsum" "ilpost" "fanpage"
  do
    python src/metrics_huggingface_it_model.py --path ./Data/${DATASET}/test.csv --batch-size ${BATCH_SIZE} --model ARTeLab/mbart-summarization-${MODEL} > logs/predbeams5_${DATASET}_MBart_${MODEL}.log
  done
done