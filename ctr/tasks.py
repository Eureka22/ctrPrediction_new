from collections import namedtuple
import os.path
import numpy as np
import scipy.io
import common.time as time
from sklearn import cross_validation, preprocessing
from sklearn.metrics import roc_curve, auc
import random
from sklearn.isotonic import IsotonicRegression as IR
from csv import DictReader
TaskCore = namedtuple('TaskCore', ['cached_data_loader', 'data_dir', 'target', 'pipeline', 'classifier_name',
                                   'classifier', 'normalize', 'gen_ictal', 'cv_ratio'])

class Task(object):
    """
    A Task computes some work and outputs a dictionary which will be cached on disk.
    If the work has been computed before and is present in the cache, the data will
    simply be loaded from disk and will not be pre-computed.
    """
    def __init__(self, task_core):
        self.task_core = task_core

    def filename(self):
        raise NotImplementedError("Implement this")

    def run(self):
        print 'filename:',self.filename()
        return self.task_core.cached_data_loader.load(self.filename(), self.load_data)



class LoadTrainingDataTask(Task):
    """
    Load the interictal mat files 1 by 1, transform each 1-second segment through the pipeline
    and return data in the format {'X': X, 'Y': y}
    """
    def filename(self):
        return 'data_train_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name())

    def load_data(self):
        return parse_input_data(self.task_core.data_dir, self.task_core.target,  self.task_core.pipeline)



class LoadTestDataTask(Task):
    """
    Load the test mat files 1 by 1, transform each 1-second segment through the pipeline
    and return data in the format {'X': X}
    """
    def filename(self):
        return 'data_test_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name())

    def load_data(self):
        return parse_test_data(self.task_core.data_dir, self.task_core.target, self.task_core.pipeline)


class TrainingDataTask(Task):
    """
    Creating a training set and cross-validation set from the transformed ictal and interictal data.
    """
    def filename(self):
        return None  # not cached, should be fast enough to not need caching

    def load_data(self):
        train_data = LoadTrainingDataTask(self.task_core).run()
        return prepare_training_data(train_data, self.task_core.cv_ratio)


class CrossValidationScoreTask(Task):
    """
    Run a classifier over a training set, and give a cross-validation score.
    """
    def filename(self):
        return 'score_%s_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name(), self.task_core.classifier_name)

    def load_data(self):
        data = TrainingDataTask(self.task_core).run()
        classifier_data = train_classifier(self.task_core.classifier, data, normalize=self.task_core.normalize)
        del classifier_data['classifier'] # save disk space
        return classifier_data


class CrossValidationScoreFullTask(Task):
    """
    Run a classifier over a training set, and give a cross-validation score.
    """
    def filename(self):
        return 'score_%s_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name(), self.task_core.classifier_name)

    def load_data(self):
        data = TrainingDataTask(self.task_core).run()
        classifier_data = train_classifier(self.task_core.classifier, data, normalize=self.task_core.normalize, return_data=True)
        return classifier_data


class TrainClassifierTask(Task):
    """
    Run a classifier over the complete data set (training data + cross-validation data combined)
    and save the trained models.
    """
    def filename(self):
        return 'classifier_%s_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name(), self.task_core.classifier_name)

    def load_data(self):
        data = TrainingDataTask(self.task_core).run()
        return train_classifier(self.task_core.classifier, data, use_all_data=True, normalize=self.task_core.normalize)


class TrainClassifierwithCalibTask(Task):
    """
    Run a classifier over the complete data set (training data + cross-validation data combined)
    and save the trained models.
    """
    def filename(self):
        return 'classifier_%s_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name(), self.task_core.classifier_name)

    def load_data(self):
        data = TrainingDataTask(self.task_core).run()
        return train_classifier_with_calib(self.task_core.classifier, data, use_all_data=False, normalize=self.task_core.normalize)



class MakePredictionsTask(Task):
    """
    Make predictions on the test data.
    """
    def filename(self):
        return 'predictions_%s_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name(), self.task_core.classifier_name)

    def load_data(self):
        data = TrainingDataTask(self.task_core).run()
        y_classes = data['y_classes']
        del data

        classifier_data = TrainClassifierTask(self.task_core).run()
        test_data = LoadTestDataTask(self.task_core).run()
        X_test = flatten(test_data['X'])
        X_label = test_data['label']
        return make_predictions(self.task_core.target, X_label, X_test, y_classes, classifier_data)




def load_csv_data(data_dir, target):
    f_train = open(os.path.join(data_dir, target))
    return [row for t, row in enumerate(DictReader(f_train))]
    
    

def parse_input_data(data_dir, target, pipeline):
    csvdata = load_csv_data(data_dir, target)
    def process_raw_data(data):
        samplerow = data[0]
        train = 'click' in samplerow
        start = time.get_seconds()
        X = []
        y = []
        for row in data:
            if 'click' in row:
                if row['click'] == '1':
                    y.append(1)
                else:
                    y.append(0)
                del row['click']
            #print row
            transformed_data = pipeline.apply(row)
            
            X.append(transformed_data)
        print '(%ds)' % (time.get_seconds() - start)
        
        X = np.array(X)
        y = np.array(y)
        
        if train:
            return X, y
        else:
            return X

    data = process_raw_data(csvdata)
    
    if len(data) == 2:
        X, y = data
        return {
            'X': X,
            'y': y
        }
    else:
        X = data
        return {
            'X': X
        }


def parse_test_data(data_dir, target, pipeline):
    csvdata = load_csv_data(data_dir, target+"_test")
    def process_raw_data(data):
        samplerow = data[0]
        train = 'click' in samplerow
        start = time.get_seconds()
        X = []
        y = []
        label = []
        for row in data:
            if 'click' in row:
                if row['click'] == '1':
                    y.append(1)
                else:
                    y.append(0)
                del row['click']
            #print row
            label.append(row['id'])
            transformed_data = pipeline.apply(row)
            
            
            X.append(transformed_data)
        print '(%ds)' % (time.get_seconds() - start)
        
        X = np.array(X)
        y = np.array(y)
        
        if train:
            return label, X, y
        else:
            return label, X

    data = process_raw_data(csvdata)
    
    if len(data) == 3:
        label, X, y = data
        return {
            'label':label,
            'X': X,
            'y': y
        }
    else:
        label, X = data
        return {
            'label':label,
            'X': X
        }



# flatten data down to 2 dimensions for putting through a classifier
def flatten(data):
    if data.ndim > 2:
        return data.reshape((data.shape[0], np.product(data.shape[1:])))
    else:
        return data


# split up ictal and interictal data into training set and cross-validation set
def prepare_training_data(train_data, cv_ratio):
    print train_data
    start = time.get_seconds()
    def concat(a, b):
        return np.concatenate((a, b), axis=0)

    X_train = flatten(train_data['X'])
    y_train = train_data['y']
    X_train, y_train, X_cv, y_cv = split_train_random(X_train, y_train, cv_ratio)
    
   
    elapsedSecs = time.get_seconds() - start
    print "%ds" % int(elapsedSecs)

   
    y_classes = np.unique(concat(y_train, y_cv))
    
    print 'X_train:', np.shape(X_train)
    print 'y_train:', np.shape(y_train)
    print 'X_cv:', np.shape(X_cv)
    print 'y_cv:', np.shape(y_cv)
    print 'y_classes:', y_classes
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_cv': X_cv,
        'y_cv': y_cv,
        'y_classes': y_classes
    }


# split interictal segments at random for training and cross-validation
def split_train_random(X, y, cv_ratio):
    X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X, y, test_size=cv_ratio, random_state=0)
    return X_train, y_train, X_cv, y_cv



def train(classifier, X_train, y_train, X_cv, y_cv, y_classes):
    print "Training ..."
    print 'Dim', 'X', np.shape(X_train), 'y', np.shape(y_train), 'X_cv', np.shape(X_cv), 'y_cv', np.shape(y_cv)

    start = time.get_seconds()
  
    classifier.fit(X_train,y_train)
    print "Scoring..."
    S= score_classifier_auc(classifier, X_cv, y_cv, y_classes)
    score = S
    elapsedSecs = time.get_seconds() - start
    print "t=%ds score=%f" % (int(elapsedSecs), score)
    return score, S


# train classifier for predictions
def train_all_data(classifier, X_train, y_train, X_cv, y_cv):
    print "Training ..."
    X = np.concatenate((X_train, X_cv), axis=0)
    y = np.concatenate((y_train, y_cv), axis=0)
    print 'Dim', np.shape(X), np.shape(y)
    start = time.get_seconds()

    classifier.fit(X, y)
    #np.set_printoptions(threshold=np.nan)
    elapsedSecs = time.get_seconds() - start
    print "t=%ds" % int(elapsedSecs)


# sub mean divide by standard deviation
def normalize_data(X_train, X_cv):
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_cv = scaler.transform(X_cv)

    return X_train, X_cv

# depending on input train either for predictions or for cross-validation
def train_classifier(classifier, data, use_all_data=False, normalize=False, return_data = False):
    X_train = data['X_train']
    y_train = data['y_train']
    X_cv = data['X_cv']
    y_cv = data['y_cv']
    if normalize:
        X_train, X_cv = normalize_data(X_train, X_cv)
    if not use_all_data:
        score, S = train(classifier, X_train, y_train, X_cv, y_cv, data['y_classes'])
        if return_data:
            return {
                    'y_cv':y_cv.reshape((y_cv.size//drate,drate)).mean(axis=-1),
                    'pred':classifier.predict_proba(X_cv)[:,1].reshape((y_cv.size//drate,drate)).mean(axis=-1)
                    }
        else:
            return {
                'classifier': classifier,
                'score': score,
                'S_auc': S,
            }
    else:
        train_all_data(classifier, X_train, y_train, X_cv, y_cv)
        return {
            'classifier': classifier
        }



# use the classifier and make predictions on the test data
def make_predictions(target, X_label, X_test, y_classes, classifier_data):
    classifier = classifier_data['classifier']
    print X_test
    predictions_proba = classifier.predict_proba(X_test)
    
    #np.set_printoptions(threshold=np.nan)
    proba = predictions_proba[:,1];
    lines = []
    for i in range(len(proba)):
        S = proba[i]
        lines.append('%s,%d' % ( X_label[i], S))

    return {
        'data': '\n'.join(lines)
    }


# the scoring mechanism used by the competition leaderboard
def score_classifier_auc(classifier, X_cv, y_cv, y_classes):
    predictions = classifier.predict_proba(X_cv)
    S_predictions = []
    S_y_cv =  y_cv

    for i in range(len(predictions)):
        p = predictions[i]
        S= translate_prediction(p, y_classes)
        S_predictions.append(S)

    fpr, tpr, thresholds = roc_curve(S_y_cv, S_predictions)
    S_roc_auc = auc(fpr, tpr)
    return S_roc_auc

