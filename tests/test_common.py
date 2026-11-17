import numpy as np
from src.common import unpack, metrics, choose_thresholds, partition_train
from src.train_tfidf import fit_model, probabilities

def test_labels_and_metric_order():
    cases = [{"Function": "foo", "Label": [1,0]}, {"Function": "bar", "Label": [0,1]}]
    x, y = unpack(cases)
    assert x == ["foo", "bar"]
    assert metrics(y,y)["mean_f1"] == 1.0

def test_thresholds():
    labels=np.array([[0,1],[1,0],[1,1],[0,0]])
    probs=np.array([[.1,.8],[.8,.1],[.9,.9],[.2,.2]])
    thresholds=choose_thresholds(labels,probs)
    assert len(thresholds)==2
    assert metrics(labels, (probs>=thresholds).astype(int))["mean_f1"]==1.0

def test_train_split():
    cases=[{"Function":f"def f{i}(): pass", "Label":[i%2,(i//2)%2]} for i in range(40)]
    fit,tune=partition_train(cases)
    assert len(fit)+len(tune)==40
    assert not set(map(id,fit)) & set(map(id,tune))

def test_small_model():
    cases=[{"Function": ("TODO FIXME debt " if i%2 else "safe clean ")+str(i),
            "Label":[i%2,i%2]} for i in range(20)]
    model=fit_model(cases,max_features=1000)
    assert probabilities(model,["TODO FIXME debt", "safe clean"]).shape==(2,2)
