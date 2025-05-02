from src.clap_score import clap_score
import pandas as pd
import sys
import os

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

baselines = ["audiocaps", "clotho"]
models = ["AudioLDM", "AudioLDM2", "MAA", "MAA2", "SAO", "Tango", "Tango2"]

steps = [10, 25, 50, 100, 150, 200]

for model in models:
    model_scores = []
    for base in baselines:
        for step in steps:
            generated_path = f'{parent_dir}/genaudios/{model}/{base}/{step}'
            csv_file_path = f'{parent_dir}/{base}_captions.csv'
            print(f'Computing CLAP score for model={model}, baseline={base}, steps={step} ...')
            df = pd.read_csv(csv_file_path)

            def rename_file(file_name, model, step):
                filename_parts = file_name.split('.wav')
                numeric_part = filename_parts[0]
                return f'{model}_{step}-steps-{numeric_part}'

            df['file_name'] = df['file_name'].apply(lambda x: rename_file(x, model, step))
            id2text = df.set_index('file_name')['caption'].to_dict()
            clp = clap_score(id2text, generated_path, audio_files_extension='.wav')
            score_item = clp.item()
            print(f'{model} {step} steps ({base}): CLAP score = {score_item}')
            model_scores.append({
                'model': model,
                'steps': step,
                'baseline': base,
                'clap_score': score_item
            })

    model_df = pd.DataFrame(model_scores)
    output_file_path = f"{current_dir}/output/{model}_clap_scores.csv"
    model_df.to_csv(output_file_path, index=False)
    print(f"CLAP scores for model '{model}' saved to '{output_file_path}'")
