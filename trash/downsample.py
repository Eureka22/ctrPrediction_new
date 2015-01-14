from collections import namedtuple
import os.path
import numpy as np
import scipy.io
import common.time as time
from sklearn import cross_validation, preprocessing
from sklearn.metrics import roc_curve, auc
import json
import scipy.signal
from common.data import CachedDataLoader, makedirs


def downsample_mat_data(data_dir, target, component):
    dir = os.path.join(data_dir, target)
    done = False
    i = 0
    while not done:
        i += 1
        filename = '%s/%s_%s_segment_%04d.mat' % (dir, target, component, i)
        if os.path.exists(filename):
            mat_data = scipy.io.loadmat(filename)
            segdata =  mat_data['%s_segment_%d' % (component,i) ]
            ddata = segdata[0,0]
            data = ddata['data']
            print data.shape
            data = scipy.signal.resample(data,400*600,axis=-1)
            print data.shape
            #ddata['data'] = data
            segdata[0,0]['data'] = data
            segdata[0,0]['sampling_frequency'] = 400
            mat_data['%s_segment_%d' % (component,i) ] = segdata
            filename2 = '%s_downsample/%s_%s_segment_%04d.mat' % (dir, target, component, i)
            scipy.io.savemat(filename2,mat_data)
        else:
            if i == 1:
                raise Exception("file %s not found" % filename)
            done = True




if __name__ == "__main__":
    with open('SETTINGS.json') as f:
        settings = json.load(f)
    data_dir = str(settings['competition-data-dir'])
    targets = [
         #'Dog_1',
         #'Dog_2',
         #'Dog_3',
         #'Dog_4',
         #'Dog_5',
         'Patient_1',
         'Patient_2',
    ]
    components = ["test","interictal","preictal"]
    for target in targets:
        makedirs("%s/%s_downsample" % (data_dir, target))
        for component in components:
            downsample_mat_data(data_dir=data_dir, target=target, component=component)
