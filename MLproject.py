import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib, statistics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")



class classification():
    def __init__(self, path='Train_data.csv', clf_opt='lr', no_of_selected_features=None):
        self.path = path
        self.clf_opt = clf_opt
        self.no_of_selected_features = no_of_selected_features
        if self.no_of_selected_features != None:
            self.no_of_selected_features = int(self.no_of_selected_features)

        # Selection of classifiers

    def classification_pipeline(self):
        # AdaBoost
        if self.clf_opt == 'ab':
            print('\n\t### Training AdaBoost Classifier ### \n')
            be1 = svm.SVC(kernel='linear', class_weight='balanced', probability=True)
            be2 = LogisticRegression(solver='liblinear', class_weight='balanced')
            be3 = DecisionTreeClassifier(max_depth=50)
            clf = AdaBoostClassifier(algorithm='SAMME.R', n_estimators=100)
            clf_parameters = {
                'clf__base_estimator': (be1,be2,be3),
                'clf__random_state': (0,5,10),
            }

            # Decision Tree
        elif self.clf_opt == 'dt':
            print('\n\t### Training Decision Tree Classifier ### \n')
            clf = DecisionTreeClassifier(random_state=40)
            clf_parameters = {
                'clf__criterion': ('gini', 'entropy'),
                'clf__max_features': ('auto', 'sqrt', 'log2'),
                'clf__max_depth': (10,40,45,60),
                'clf__ccp_alpha': (0.009,0.01,0.05,0.1),
            }

            # K-nearest neighbors
        elif self.clf_opt == 'knn':
            print('\n\t### Training KNN ### \n')
            clf = KNeighborsClassifier()
            clf_parameters = { 'clf__n_neighbors': (3, 5, 7),
                               'clf__metric': ('euclidean', 'manhattan'),
            }


            # Logistic Regression
        elif self.clf_opt == 'lr':
            print('\n\t### Training Logistic Regression Classifier ### \n')
            clf = LogisticRegression(solver='liblinear', class_weight='balanced')
            clf_parameters = {
                'clf__random_state': (0, 10),
            }

            # Multinomial Naive Bayes
        elif self.clf_opt == 'nb':
            print('\n\t### Training Multinomial Naive Bayes Classifier ### \n')
            clf = MultinomialNB(fit_prior=True, class_prior=None)
            clf_parameters = {
                'clf__alpha': (0, 1),
            }

            # Random Forest
        elif self.clf_opt == 'rf':
            print('\n\t ### Training Random Forest Classifier ### \n')
            clf = RandomForestClassifier(max_features=None, class_weight='balanced')
            clf_parameters = {
                'clf__criterion': ('entropy','gini'),
                'clf__n_estimators': (30,50,100),
                'clf__max_depth': (10,20,30,50,100,200),
            }

            # Support Vector Machine
        elif self.clf_opt == 'svm':
            print('\n\t### Training SVM Classifier ### \n')
            clf = svm.SVC(class_weight='balanced', probability=True)
            clf_parameters = {
                'clf__C': (0.1,1,50,100),
                'clf__kernel': ('poly','rbf','linear'),
            }
        else:
            print('Select a valid classifier \n')
            sys.exit(0)
        return clf, clf_parameters

    # Statistics of individual classes
    def get_class_statistics(self, labels):
        class_statistics = Counter(labels)
        print('\n Class \t\t Number of Instances \n')
        for item in list(class_statistics.keys()):
            print('\t' + str(item) + '\t\t\t' + str(class_statistics[item]))

    # Load the data
    def get_data(self):

        reader = pd.read_csv("Train_Data.csv")
        leader = pd.read_csv("Traindata_classlabels.csv")

        data = reader

        # scaler = MinMaxScaler()
        labels = leader[('price_range')]
        # scaled_data = pd.DataFrame(scaler.fit_transform(data))

        self.get_class_statistics(labels)


        # Encode multiple features using one-hot encoding
        encoder = OneHotEncoder()
        encoded_features = encoder.fit_transform(
            data[['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']])

        data = data.drop(columns=['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi'], axis=1)
        data = pd.concat([data, pd.DataFrame(encoded_features.toarray())], axis=1)

        print()
        print("missing values")
        print(data.isnull().sum())
        print(data.head())
        print(data.shape)
        print()
        print(data.info())

        return data, labels


    def classification(self):
        # Get the data
        data, labels = self.get_data()
        data = np.asarray(data)

        # Experiments using training data only during training phase (dividing it into training and validation set)
        skf = StratifiedKFold(n_splits=5)
        predicted_class_labels = [];
        actual_class_labels = [];
        count = 0;
        probs = [];
        for train_index, test_index in skf.split(data, labels):
            X_train = [];
            y_train = [];
            X_test = [];
            y_test = []
            for item in train_index:
                X_train.append(data[item])
                y_train.append(labels[item])
            for item in test_index:
                X_test.append(data[item])
                y_test.append(labels[item])
            count += 1
            print('Training Phase ' + str(count))
            clf, clf_parameters = self.classification_pipeline()
            pipeline = Pipeline([
                ('feature_selection', SelectKBest(chi2, k=self.no_of_selected_features)),  # k=1000 is recommended
                #        ('feature_selection', SelectKBest(mutual_info_classif, k=self.no_of_selected_features)),
                ('clf', clf), ])
            grid = GridSearchCV(pipeline, clf_parameters, scoring='f1_micro', cv=5)
            grid.fit(X_train, y_train)
            clf = grid.best_estimator_
            print('\n\n The best set of parameters of the pipeline are: ')
            print(clf)
            predicted = clf.predict(X_test)
            predicted_probability = clf.predict_proba(X_test)
            for item in predicted_probability:
                probs.append(float(max(item)))
            for item in y_test:
                actual_class_labels.append(item)
            for item in predicted:
                predicted_class_labels.append(item)
        confidence_score = statistics.mean(probs) - statistics.variance(probs)
        confidence_score = round(confidence_score, 3)
        print('\n The Probablity of Confidence of the Classifier: \t' + str(confidence_score) + '\n')
        #joblib.dump(clf, 'svm1.joblib')
        # Evaluation
        class_names = list(Counter(labels).keys())
        class_names = [str(x) for x in class_names]
        # print('\n\n The classes are: ')
        # print(class_names)



        print('\n ##### Classification Report on Training Data ##### \n')
        print(classification_report(actual_class_labels, predicted_class_labels, target_names=class_names))

        pr = precision_score(actual_class_labels, predicted_class_labels, average='macro')
        print('\n Precision:\t' + str(pr))

        rl = recall_score(actual_class_labels, predicted_class_labels, average='macro')
        print('\n Recall:\t' + str(rl))

        fm = f1_score(actual_class_labels, predicted_class_labels, average='macro')
        print('\n F1-Score:\t' + str(fm))

        print("\n Confusion Matrix :\t")
        conf_mat = confusion_matrix(actual_class_labels, predicted_class_labels)
        print(conf_mat)

        # Plotting the Confusion Matrix as a Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_mat, annot=True, cmap='Blues', fmt='g')  # Creating the heatmap
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title("Confusion Matrix :")
        plt.show()

clf = classification('Train_Data.csv', clf_opt="svm",
                     no_of_selected_features=6)

clf.classification()
