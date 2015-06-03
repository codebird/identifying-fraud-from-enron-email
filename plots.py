#!/usr/bin/python
import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from my_funcs import *
import matplotlib.pyplot as plt
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
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments',
'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 
'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 
'expenses', 'loan_advances', 'from_messages', 'other', 
'from_this_person_to_poi', 'director_fees', 'deferred_income', 'sal_exp_df',
'long_term_incentive', 'from_poi_to_this_person', 'stocks_money', 'di_lti',\
 'other_bonus', 'exp_la']

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r"))
#remove outliers
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

my_dataset = data_dict

plot_2_vars('salary', 'bonus', 'poi', my_dataset, True,\
'Salary',
'Bonus')

plot_2_vars('salary', 'exercised_stock_options', 'poi', my_dataset, True,\
'Salary',
'Exercised Stock Options')

plot_2_vars('to_messages', 'from_messages', 'poi', my_dataset, True,\
'To Messages',
'From Messages')

