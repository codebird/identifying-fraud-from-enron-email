import sys
sys.path.append( "../tools/" )
from os import listdir
from os.path import isfile, join
from parse_out_email_text import parseOutText
from parse_email_from import parse_email_from
from sklearn.feature_extraction.text import TfidfVectorizer
from poi_email_addresses import poiEmails
import pickle


#loop through POI names and create the directory names that 
#are linked to these POIs
with open('poi_names.txt', 'r')as f:
	line=f.readline()
	line=f.readline()
	line=f.readline()
	interesting_dirs=[]
	while(line):
		line_list=line.split(')')
		if(line_list[0].strip()=='(y'):
			name=line_list[1].strip()
			last_name, first_name=name.split(', ')
			dir_name=last_name.lower()+'-'+first_name[0].lower()
			interesting_dirs.append(dir_name)
		line=f.readline()

#Get all POI emails
poi_emails=poiEmails()
	
#set the path and folder of interest
path='enron_mail/maildir/'
inbox='/inbox/'

#create a list of the texts in the emails by looping through the interesting 
#directories, loop through all the files in the inbox, parse out the text only
# and add it to our word_data list
words_data=[]
for directory in interesting_dirs:
	this_path=path+directory+inbox
	for f in listdir(this_path):
		processed_text=''
		if isfile(this_path+f):
			with open(this_path+f, 'r') as email:
				from_who=parse_email_from(email)
				if(from_who in poi_emails):
					print from_who
					text=parseOutText(email)
					words_data.append(text)

#dump words data into a pickle file not to have to run this script each time
#you need the data
pickle.dump(words_data, open('email_between_pois.pkl', 'w'))

#Load word_data from the pickle file
words_data=pickle.load(open('email_between_pois.pkl', 'rb'))

#Try tfidf vectorizer on the data to see what it comes out with. I used 3% 
#document frequency, thinking that interesting words, shouldn't be words that
#appear in too many emails.
tfidf=TfidfVectorizer(max_df=0.03, sublinear_tf=True, stop_words='english')
features=tfidf.fit_transform(words_data)
#print out the words we got.
print tfidf.get_feature_names()
