"""Aggregate independently measured model result JSON files into a CSV.
Usage: python -m src.compare_results results/tfidf_public_test.json results/codebert_public_test.json
Does not invent or infer missing metrics.
"""
import argparse, csv, json
from pathlib import Path

def flatten(path):
    p=Path(path)
    data=json.loads(p.read_text(encoding='utf-8'))
    if 'metrics' in data and 'mean_f1' not in data:
        data=data['metrics']
    row={'experiment':p.stem}
    for task in ('weakness','MAT'):
        task_data=data.get(task,{})
        for key in ('precision','recall','f1','accuracy','positives'):
            row[f'{task}_{key}']=task_data.get(key,'')
    for key in ('mean_f1','inference_seconds_per_example','delta_vs_published_baseline'):
        row[key]=data.get(key,'')
    return row

def main():
    p=argparse.ArgumentParser()
    p.add_argument('files',nargs='+')
    p.add_argument('--output',default='results/comparison.csv')
    args=p.parse_args()
    rows=[flatten(path) for path in args.files]
    keys=list(rows[0].keys())
    dst=Path(args.output)
    dst.parent.mkdir(parents=True,exist_ok=True)
    with dst.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader();writer.writerows(rows)
    print('Wrote',dst,'with',len(rows),'experiment rows')

if __name__=='__main__':
    main()
