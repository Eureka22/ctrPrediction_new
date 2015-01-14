import json
import os.path
import numpy as np
from common import time
from common.data import CachedDataLoader, makedirs
from common.pipeline import Pipeline
from ctr.transforms import TimeAliasing,RFFT, FFT, Slice, Magnitude, Log10, FFTWithTimeFreqCorrelation, MFCC, Resample, Stats, featureEngineering, \
    DaubWaveletStats, TimeCorrelation, FreqCorrelation, TimeFreqCorrelation, Hash
from ctr.tasks import TaskCore, CrossValidationScoreTask, MakePredictionsTask, TrainClassifierTask, TrainClassifierwithCalibTask, CrossValidationScoreFullTask

from ctr.scores import get_score_summary, print_results
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.lda import LDA
#from mail import send_message


def run_ctrprediction(build_target):
    """
    The main entry point for running seizure-detection cross-validation and predictions.
    Directories from settings file are configured, classifiers are chosen, pipelines are
    chosen, and the chosen build_target ('cv', 'predict', 'train_model') is run across
    all combinations of (targets, pipelines, classifiers)
    """

    with open('SETTINGS.json') as f:
        settings = json.load(f)

    data_dir = str(settings['competition-data-dir'])
    cache_dir = str(settings['data-cache-dir'])
    submission_dir = str(settings['submission-dir'])
    makedirs(submission_dir)
    cached_data_loader = CachedDataLoader(cache_dir)
    ts = time.get_millis()
    targets = [
        'train'
   ]
    pipelines = [
         Pipeline(pipeline=[featureEngineering(),Hash(24)]),
    ]
    classifiers = [
        # NOTE(mike): you can enable multiple classifiers to run them all and compare results
         (RandomForestClassifier(n_estimators=3, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf3mss1Bfrs0'),
        # (RandomForestClassifier(n_estimators=150, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf150mss1Bfrs0'),
        # (RandomForestClassifier(n_estimators=300, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf300mss1Bfrs0'),
        # (RandomForestClassifier(n_estimators=3000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf3000mss1Bfrs0'),
        # (GaussianNB(),'gbn'),
        # (BernoulliRBM(n_components=100),'dbn'),
        # (SVC(probability = True),'svc100'),
        # (LDA(),'lda'),

    ]
    cv_ratio = 0.1

    def should_normalize(classifier):
        clazzes = [LogisticRegression]
        return np.any(np.array([isinstance(classifier, clazz) for clazz in clazzes]) == True)

    def train_full_model(make_predictions):
        for pipeline in pipelines:
            for (classifier, classifier_name) in classifiers:
                print 'Using pipeline %s with classifier %s' % (pipeline.get_name(), classifier_name)
                guesses = ['id,click']
                classifier_filenames = []
                for target in targets:
                    task_core = TaskCore(cached_data_loader=cached_data_loader, data_dir=data_dir,
                                         target=target, pipeline=pipeline,
                                         classifier_name=classifier_name, classifier=classifier,
                                         normalize=should_normalize(classifier), gen_ictal=False,
                                         cv_ratio=cv_ratio)


                    if make_predictions:
                        predictions = MakePredictionsTask(task_core).run()
                        guesses.append(predictions['data'])
                    else:
                        task = TrainClassifierTask(task_core)
                        print "training"
                        task.run()
                        print "train_finished"
                        classifier_filenames.append(task.filename())


                if make_predictions:
                    filename = 'submission%d-%s_%s.csv' % (ts, classifier_name, pipeline.get_name())
                    filename = os.path.join(submission_dir, filename)
                    with open(filename, 'w') as f:
                        print >> f, '\n'.join(guesses)
                    print 'wrote', filename
                else:
                    print 'Trained classifiers ready in %s' % cache_dir
                    for filename in classifier_filenames:
                        print os.path.join(cache_dir, filename + '.pickle')


  

    def do_cross_validation():
        summaries = []
        for pipeline in pipelines:
            for (classifier, classifier_name) in classifiers:
                print 'Using pipeline %s with classifier %s' % (pipeline.get_name(), classifier_name)
                scores = []
                S_scores = []
                E_scores = []
                for target in targets:
                    print 'Processing %s (classifier %s)' % (target, classifier_name)

                    task_core = TaskCore(cached_data_loader=cached_data_loader, data_dir=data_dir,
                                         target=target, pipeline=pipeline,
                                         classifier_name=classifier_name, classifier=classifier,
                                         normalize=should_normalize(classifier), gen_ictal=False,
                                         cv_ratio=cv_ratio)

                    data = CrossValidationScoreTask(task_core).run()
                    score = data['score']
                    scores.append(score)

                    print '%.3f' % score, 'S=%.4f' % data['S_auc']
                    S_scores.append(data['S_auc'])

                if len(scores) > 0:
                    name = pipeline.get_name() + '_' + classifier_name
                    summary = get_score_summary(name, scores)
                    summaries.append((summary, np.mean(scores)))
                    print summary
                if len(S_scores) > 0:
                    name = pipeline.get_name() + '_' + classifier_name
                    summary = get_score_summary(name, S_scores)
                    print 'S', summary

            print_results(summaries)




    if build_target == 'cv':
        do_cross_validation()
    elif build_target == 'train_model':
        train_full_model(make_predictions=False)
    elif build_target == 'make_predictions':
        train_full_model(make_predictions=True)
    else:
        raise Exception("unknown build target %s" % build_target)
