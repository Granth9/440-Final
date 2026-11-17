"""Read-only data audit: label distributions, duplicate hashes, project overlap.
Run: python -m src.audit_dataset --train-limit 3000 --test-limit 500
Omit limits for a complete audit. Does not train a model or tune on the public test.
"""
import argparse
import hashlib
from collections import Counter, defaultdict
from src.common import load_data, write_json

def signature(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def analyze(train, evaluation):
    def summary(rows):
        labels = Counter(''.join(map(str, x['Label'])) for x in rows)
        projects = Counter(x.get('Projectname') or 'UNKNOWN' for x in rows)
        hashes = Counter(signature(x['Function']) for x in rows)
        label_by_hash = defaultdict(set)
        for x in rows:
            label_by_hash[signature(x['Function'])].add(tuple(x['Label']))
        return {
            'rows': len(rows), 'label_counts': dict(sorted(labels.items())),
            'label_proportions': {k: round(v / len(rows), 6) for k,v in sorted(labels.items())} if rows else {},
            'weakness_positive': sum(int(x['Label'][0]) for x in rows),
            'satd_positive': sum(int(x['Label'][1]) for x in rows),
            'unique_functions': len(hashes),
            'duplicate_rows': sum(v-1 for v in hashes.values()),
            'conflicting_label_function_hashes': sum(len(v)>1 for v in label_by_hash.values()),
            'top_projects': projects.most_common(15),
        }
    tr_hash = {signature(x['Function']) for x in train}
    ev_hash = {signature(x['Function']) for x in evaluation}
    tr_projects = {x.get('Projectname') for x in train}
    ev_projects = {x.get('Projectname') for x in evaluation}
    return {'train': summary(train), 'public_evaluation': summary(evaluation),
            'overlap': {'exact_function_hashes': len(tr_hash & ev_hash),
                        'project_names': len(tr_projects & ev_projects)},
            'warning': 'Shared hashes/projects are descriptive, not evidence of mislabeled data. Do not use public evaluation labels for model selection.'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train-limit', type=int)
    p.add_argument('--test-limit', type=int)
    p.add_argument('--output',default='results/dataset_audit.json')
    args=p.parse_args()
    tr,ev=load_data(args.train_limit,args.test_limit)
    report=analyze(tr,ev)
    write_json(args.output,report)
    print('Audit saved:',args.output)
    print('Train examples:',report['train']['rows'])
    print('Evaluation examples:',report['public_evaluation']['rows'])
    print('Cross-split exact function hashes:',report['overlap']['exact_function_hashes'])

if __name__=='__main__':
    main()
