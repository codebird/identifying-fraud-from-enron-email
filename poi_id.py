#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from my_funcs import repair_nan
import time
import pprint
############################################################################
#                           column names                                   #
############################################################################
#'salary', 'to_messages', 'deferral_payments', 'total_payments', 
#'exercised_stock_options', 'bonus', 'restricted_stock', 
#'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 
#'expenses', 'loan_advances', 'from_messages', 'other', 
#'from_this_person_to_poi', 'poi', 'director_fees', 'deferred_income', 
#'long_term_incentive', 'from_poi_to_this_person'
############################################################################
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# You will need to use more features
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments',
'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 
'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 
'expenses', 'loan_advances', 'from_messages', 'other', 
'from_this_person_to_poi', 'director_fees', 'deferred_income', 
'long_term_incentive', 'from_poi_to_this_person', 
'stocks_money', 'di_lti', 'other_bonus', 'exp_la']

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r"))
#remove outliers
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

for item in my_dataset:
	my_dataset[item]['stocks_money']=\
repair_nan(my_dataset[item]['exercised_stock_options'], 1)*\
repair_nan(my_dataset[item]['total_stock_value'], 1)
	
	my_dataset[item]['di_lti']=\
repair_nan(my_dataset[item]['long_term_incentive'], 1)*\
repair_nan(my_dataset[item]['deferred_income'], 1)

	my_dataset[item]['exp_la']=\
repair_nan(my_dataset[item]['expenses'], 0)*\
repair_nan(my_dataset[item]['loan_advances'], 0)

	my_dataset[item]['other_bonus']=\
repair_nan(my_dataset[item]['other'])+repair_nan(my_dataset[item]['bonus'])

	my_dataset[item]['sal_exp_df']=\
repair_nan(my_dataset[item]['salary'])+repair_nan(my_dataset[item]['expenses'])\
+repair_nan(my_dataset[item]['director_fees'])\
/repair_nan(my_dataset[item]['total_payments'], 1)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

labels, features = targetFeatureSplit(data)

#import and apply SelectKBest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
selector=SelectKBest(f_classif, k=22)
selector.fit(features, labels)

#Create new feature list based on the the features we got from selectKBest
new_feature_list=['poi']
k_fea_list={}
for key, item in enumerate(selector.get_support()):
	if item:
		new_feature_list.append(features_list[key+1])
		k_fea_list[features_list[key+1]]=selector.scores_[key]

#sort the features by importance and print the list out
#k_fea_list=sorted(k_fea_list.items(), key=lambda x: x[1], reverse=True)
#pprint.pprint(k_fea_list)
new_feature_list.remove('restricted_stock')
new_feature_list.remove('shared_receipt_with_poi')
new_feature_list.remove('other_bonus')
new_feature_list.remove('to_messages')
new_feature_list.remove('deferral_payments')

features_list=new_feature_list

#Format the data for it to be ready for predictions
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#log start time
start_time=time.time()
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#create the classifier
clf=LogisticRegression(C=11500, fit_intercept=False)

#predict
test_classifier(clf, my_dataset, features_list)
print 'Time needed: ', time.time()-start_time, 'seconds'

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.
dump_classifier_and_data(clf, my_dataset, features_list)
