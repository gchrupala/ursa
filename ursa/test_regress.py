from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays
from hypothesis import given, example
import numpy as np
import ursa.regress as R
from sklearn.datasets import load_iris
iris = load_iris()

def test_Regress_keys():
    X=iris.data[:,:2]
    Y=iris.data[:,2:]
    r = R.Regress(cv=3)
    r.fit(X, Y)
    report = r.report()
    assert "mse" in report
    assert "r2" in report
    assert "pearson_r" in report

    
def test_Regress_mse():
    X=iris.data[:,:2]
    Y=iris.data[:,2:]
    r = R.Regress(cv=3)
    r.fit(X, Y)
    report = r.report()
    assert report["mse"]["mean"] >= 0.0

        
