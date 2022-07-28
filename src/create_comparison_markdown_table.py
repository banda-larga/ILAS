import os

start = """
| Test Dataset | train Dataset | model | rouge1            |rouge2|rougeL|rougeL (sum)| gen len|
|---------------|---------------|-------|-------------------|---|---|---|---|
"""
folder = '../logs'
files = list(filter(lambda x: x.startswith('pred'), os.listdir(folder)))
files.sort()
print(start.strip())
for f in files:
    _, test_dataset, model, train_dataset = f.split('.')[0].split('_')
    file_path = os.path.join(folder, f)
    with open(file_path, 'r') as file:
        content = file.readlines()[-1]
    print(f'|{test_dataset}|{train_dataset}|{model}|', content.strip(), '|')
