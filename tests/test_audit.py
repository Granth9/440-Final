from src.audit_dataset import analyze
from src.compare_results import flatten
import json

def test_duplicates_and_overlap():
    train=[{'Function':'a','Label':[0,0],'Projectname':'p'},
           {'Function':'a','Label':[0,0],'Projectname':'p'},
           {'Function':'b','Label':[1,0],'Projectname':'q'}]
    test=[{'Function':'b','Label':[1,0],'Projectname':'q'}]
    report=analyze(train,test)
    assert report['train']['duplicate_rows']==1
    assert report['overlap']['exact_function_hashes']==1
    assert report['train']['label_counts']['00']==2

def test_comparison_handles_missing_keys(tmp_path):
    p=tmp_path/'trial.json'
    p.write_text(json.dumps({'mean_f1':0.7,'weakness':{'f1':0.8},'MAT':{'f1':0.6}}))
    row=flatten(p)
    assert row['mean_f1']==0.7
    assert row['weakness_f1']==0.8
    assert row['MAT_recall']==''
