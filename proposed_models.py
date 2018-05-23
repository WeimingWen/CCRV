import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from tools import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

crossplatform = ['dist mean', 'dist var', 'agree mean', 'agree var', 'disagree mean',
                 'disagree var', 'discuss mean', 'discuss var', 'unrelated mean', 'unrelated var']
tweetfeats = ['itemLength', 'numWords', 'containsQuestionMark', 'numQuestionMark', 'containsExclamationMark',
              'numExclamationMark', 'containsHappyEmo', 'containsSadEmo', 'containsFirstOrderPron',
              'containsSecondOrderPron', 'containsThirdOrderPron', 'numUppercaseChars', 'numPosSentiWords',
              'numNegSentiWords', 'numMentions', 'numHashtags', 'numURLs', 'retweetCount']
tweetadd = ['numNouns', 'numSlangs', 'hasColon', 'hasPlease', 'hasExternalLink', 'wotTrust',
            'readability', 'urlIndegree', 'urlHarmonic', 'alexaCountryRank', 'alexaDeltaRank', 'alexaPopularity',
            'alexaReachRank']
userfeats = ['numFriends', 'numFollowers', 'FolFrieRatio', 'timesListed', 'hasURL', 'isVerified', 'numTweets']
useradd = ['hasBio', 'hasLocation', 'hasExistingLocation', 'numFavorites', 'wotTrustUser', 'numMediaContent',
           'accountAge',
           'hasProfileImg', 'hasHeaderImg', 'tweetRatio', 'indegree', 'harmonic', 'alexaCountryRank',
           'alexaDeltaRank', 'alexaPopularity', 'alexaReachRank', 'FolFrieRatio', 'timesListed']


def get_top_index(importance, names):
    order = []
    for i in xrange(len(importance)):
        order.append((names[i], importance[i]))
    order = sorted(order, key=lambda x: x[1], reverse=True)
    print 'the most importance 6 features for RF:'
    print order[:6]


def SVM(X, Y, x, y,weight=None, feat=None):
    np.random.seed(777)
    clf = SVC(kernel='rbf', C=1, gamma=0.1, random_state=777)

    clf.fit(X, Y)
    if weight:
        if feat == '1':
            names = crossplatform
        elif feat == '2':
            names = crossplatform + tweetfeats + tweetadd + userfeats + useradd
        get_top_index(clf.feature_importances_, names)

    y_pred = clf.predict(x)

    return metrics(y, y_pred)


# def NB(X, Y, x, y, weight=False, feat=None):
#
#     clf.fit(X, Y)
#         if feat == '1':
#             names = crossplatform
#         elif feat == '2':
#             names = tweetfeats + tweetadd + userfeats + useradd
#         elif feat == '3':
#             names = crossplatform + tweetfeats + tweetadd + userfeats + useradd
#         get_top_index(clf.feature_importances_, names)
#
#     y_pred = clf.predict(x)
#
#     return metrics(y, y_pred)


def RF(X, Y, x, y, frac=0.1, weight=None, feat=None):
    np.random.seed(777)
    clf = RandomForestClassifier(n_estimators=300, criterion='entropy', max_depth=None,
                                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=frac,
                                 max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0,
                                 min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1,
                                 random_state=7, verbose=0, warm_start=False, class_weight=None)

    clf.fit(X, Y)
    if weight:
        if feat == '1':
            names = crossplatform
        elif feat == '2':
            names = tweetfeats + tweetadd + userfeats + useradd
        elif feat == '3':
            names = crossplatform + tweetfeats + tweetadd + userfeats + useradd
        get_top_index(clf.feature_importances_, names)

    y_pred = clf.predict(x)

    return metrics(y, y_pred)


def NN(X, Y, x, y, dropout, weight=True, feat='1', epochs=20, batch_size=21):
    np.random.seed(70)
    # creat model
    def create_larger(input_dim):
        model = Sequential()
        model.add(Dense(input_dim, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(10, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(dropout))
        # model.add(Dense(20, kernel_initializer='normal', activation='relu'))
        # model.add(Dropout(dropout))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    dim = X[0, :].size
    model = create_larger(dim)
    model.fit(X, Y, epochs, batch_size, verbose=False)

    if True:
        # if feat == '1':
        #     names = crossplatform
        # elif feat == '2':
        #     names = crossplatform + tweetfeats + tweetadd + userfeats + useradd
        #     print len(names)
        # for layer in model.layers:
        #     print layer.get_weights() #layer.get_config(),
        names = crossplatform + tweetfeats + tweetadd + userfeats + useradd
        layer_1 = np.array(model.layers[-3].get_weights()[0])
        layer_2 = np.array(model.layers[-1].get_weights()[0])
        order = []
        for i in xrange(layer_1[:, 0].size):
            w = np.sum(layer_1[i, :] * layer_2[:, 0])
            order.append((names[i], w))
        order = sorted(order, key=lambda x: abs(x[1]), reverse=True)
        print 'the most importance 6 features for NN:'
        print order[:6]

    y_pred = model.predict_classes(x)  # predict only for x!

    return metrics(y, y_pred)


def MCG(train, test, frac1, frac2):
    np.random.seed(777)

    def average_feats(data):
        data = np.array(data, dtype=float)
        # replace all with 0
        for i in xrange(data[0, :].size):
            if np.all(np.isnan(data[:, i])):
                data[:, i].fill(0)
        data = imputer(data)
        data = standardize(data)
        return [np.average(data[:, i]) for i in xrange(data[0, :].size)]

    def vote_label(label):
        fake = 0
        real = 0
        for y in label:
            if y == 'fake':
                fake += 1
            else:
                real += 1
        if real > fake:
            return 0
        else:
            return 1

    topic_dict = {}

    def get_input(elembase):
        for elem in elembase:
            if elem['image_id'][0] in topic_dict:
                name = elem['image_id'][0]
                topic_dict[name]['feats'].append(elem['agree'] + elem['dist'])
                topic_dict[name]['label'].append(elem['label'])
            else:
                name = elem['image_id'][0]
                topic_dict[name] = {}
                topic_dict[name]['feats'] = [elem['agree'] + elem['dist']]
                topic_dict[name]['label'] = [elem['label']]
        X = []
        Y = []
        for name in topic_dict:
            filled_feats = average_feats(topic_dict[name]['feats'])
            X.append(filled_feats)
            topic_dict[name]['filled_feats'] = np.array(filled_feats).reshape(1, -1)
            labels = topic_dict[name]['label']
            Y.append(vote_label(labels))
        return X, Y

    X, Y = get_input(train)
    x, y = get_input(test)
    clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None,
                                 min_samples_split=2, min_samples_leaf=1,
                                 min_weight_fraction_leaf=frac2,
                                 max_features=None, random_state=777, max_leaf_nodes=None,
                                 min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None,
                                 presort=False)
    clf.fit(X, Y)
    # y_pred=clf.predict(x)
    # F1 = f1_score(y, y_pred)
    # print 'f1 for decision tree: ' + str(F1)
    for name in topic_dict:
        prob = clf.predict_proba(topic_dict[name]['filled_feats'])[0][0]
        topic_dict[name]['prob'] = prob

    # random forest
    def topic_feats_pre(elembase):
        data = []
        label = []
        for elem in elembase:
            data.append([topic_dict[elem['image_id'][0]]['prob']] + elem['dist'] + elem['agree'])
            label.append(readf(elem['label']))

        label = np.array(label, dtype=float)
        data = np.array(data, dtype=float)
        # replace all with 0
        for i in xrange(data[0, :].size):
            if np.all(np.isnan(data[:, i])):
                data[:, i].fill(1)
        data = imputer(data)
        data = standardize(data)
        return data, label

    X, Y = topic_feats_pre(train)
    x, y = topic_feats_pre(test)
    clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None,
                                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=frac1,
                                 max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0,
                                 min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1,
                                 random_state=777, verbose=0, warm_start=False, class_weight=None)

    clf.fit(X, Y)
    y_pred = clf.predict(x)

    return metrics(y, y_pred)


#
# def baselines():
#     certh = {}
#     uos = {}
#     mcg = {}
#     df_output = pd.read_csv('resources/baselines.csv')
#     out = df_output.as_matrix().tolist()
#     for row in out[1:]:
#         certh[row[0]] = map(float, [row[4], row[5]])
#         uos[row[0]] = map(float, [row[8], row[9]])
#         mcg[row[0]] = map(float, [row[12], row[13]])
#     return certh, uos, mcg


if __name__ == '__main__':
    baselines()
