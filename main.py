import cPickle as pkl
from tools import imputer, standardize, metrics
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Dense

events = ['sandy', 'boston', 'sochi', 'malaysia', 'bringback', 'columbianChemicals', 'passport', 'elephant',
          'underwater', 'livr', 'pigFish', 'eclipse', 'samurai', 'nepal', 'garissa', 'syrianboy', 'varoufakis']


def Twitter_RV_via_Google():
    print 'verify tweets leveraging Google'
    with open('CLCP/Twitter_CLCP_via_Google.pkl', 'rb') as f:
        (X, Y), (cv_task, cv_event) = pkl.load(f)

    # task
    train, test = cv_task[0]
    train_X, train_Y = X[train], Y[train]
    test_X, test_Y = X[test], Y[test]

    f1 = train_test(train_X, train_Y, test_X, test_Y)
    print f1

    # event
    eid=0
    avg=[]
    for train, test in cv_event:
        print events[eid], eid+1
        eid+=1
        train_X, train_Y = X[train], Y[train]
        test_X, test_Y = X[test], Y[test]

        f1 = train_test(train_X, train_Y, test_X, test_Y)
        print f1
        avg.append(f1)
    print 'avg: ', np.average(avg)


def Twitter_RV_via_Baidu():
    print 'verify tweets leveraging Baidu'
    with open('CLCP/Twitter_CLCP_via_Baidu.pkl', 'rb') as f:
        (X, Y), (cv_task, cv_event) = pkl.load(f)

    # task
    train, test = cv_task[0]
    train_X, train_Y = X[train], Y[train]
    test_X, test_Y = X[test], Y[test]

    f1 = train_test(train_X, train_Y, test_X, test_Y)
    print f1

    # event
    eid=0
    avg=[]
    for train, test in cv_event:
        print events[eid], eid + 1
        eid+=1
        train_X, train_Y = X[train], Y[train]
        test_X, test_Y = X[test], Y[test]

        f1 = train_test(train_X, train_Y, test_X, test_Y)
        if f1 is None:
            print 'without source rumors!'
        else:
            print f1
            avg.append(f1)
    print 'avg: ', np.average(avg)


def Baidu_RV_via_Transfer():
    print 'verify Baidu webpages via transfer learning'
    with open('CLCP/Baidu_CLCP_via_Google.pkl', 'rb') as f:
        (test_X, test_Y), (_, cv_event) = pkl.load(f)

    with open('CLCP/Twitter_CLCP_via_Google.pkl', 'rb') as f:
        (train_X, train_Y), (_, _) = pkl.load(f)

    # pre-trained a model on CCMR Twitter dataset and test on each event in CCMR Baidu
    eid=0
    avg=[]
    rands=[]
    for train, test in cv_event:
        print events[eid], eid + 1
        eid+=1
        x, y = test_X[test], test_Y[test]
        f1 = train_test(train_X, train_Y, x, y)
        f1_rand=metrics(y, np.random.randint(2, size=y.size))
        if f1 is None:
            print 'without target rumors!'
        else:
            print 'Random: ', f1_rand
            print f1
            avg.append(f1)
            rands.append((f1_rand))

    print 'Random avg: ', np.average(rands)
    print 'avg: ', np.average(avg)


def Twitter_RV_via_All():
    print 'verify tweets leveraging Google and Baidu'
    with open('CLCP/Twitter_CLCP_via_All.pkl', 'rb') as f:
        (X, Y), (cv_task, cv_event) = pkl.load(f)

    # task
    train, test = cv_task[0]
    train_X, train_Y = X[train], Y[train]
    test_X, test_Y = X[test], Y[test]

    f1 = train_test(train_X, train_Y, test_X, test_Y)
    print f1

    # event
    eid=0
    avg=[]
    for train, test in cv_event:
        print events[eid], eid+1
        eid+=1
        train_X, train_Y = X[train], Y[train]
        test_X, test_Y = X[test], Y[test]

        f1 = train_test(train_X, train_Y, test_X, test_Y)
        print f1
        avg.append(f1)
    print 'avg: ', np.average(avg)




def build_model(hidden, dropout):
    model = Sequential()
    model.add(Dense(hidden, kernel_initializer='normal', input_dim=10, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(hidden, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_test(train_X, train_Y, test_X, test_Y):
    np.random.seed(38)
    if np.all(np.isnan(test_X)):
        return
    train_X = imputer(train_X)
    train_X = standardize(train_X)
    test_X = imputer(test_X)
    test_X = standardize(test_X)

    dropout = 0.5
    epochs = 23
    batch_size = 10
    hidden = 20
    clf = build_model(hidden=hidden, dropout=dropout)
    clf.fit(train_X, train_Y, epochs, batch_size, verbose=False)
    return metrics(test_Y, clf.predict_classes(test_X))


if __name__ == '__main__':
    Twitter_RV_via_Google()
    Twitter_RV_via_Baidu()
    Baidu_RV_via_Transfer()
    Twitter_RV_via_All()
