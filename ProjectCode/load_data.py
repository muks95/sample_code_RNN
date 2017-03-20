import os
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
from bs4 import BeautifulSoup
import re
import csv
from nltk.corpus import stopwords

work_path = '/Users/Balaji/PycharmProjects/project/'
zip_folder = 'dataset'
unzip_folder = 'dataset_unzipped'

t1 = pd.read_csv("train_output/Packet1Concensus.csv", dtype=str, header=0, delimiter=",", quoting=2)
t2 = pd.read_csv("train_output/Packet2Concensus.csv", dtype=str, header=0, delimiter=",", quoting=2)
t3 = pd.read_csv("train_output/Packet3Concensus.csv", dtype=str, header=0, delimiter=",", quoting=2)

train_filenames = []
train_output = []

test_filenames = []
test_output = []

c = 0
for i in t1['file']:
    if (i == 'nan'):
        continue
    train_filenames.append(i)
    c += 1
for i in t1['output']:
    if (i == 'N'):
        train_output.append(0)
    else:
        train_output.append(1)

print c
c = 0
for i in t2['file']:
    if (i == 'nan'):
        continue
    train_filenames.append(i)
    c += 1
print c
c = 0
for i in t2['output']:
    if (i == 'N'):
        train_output.append(0)
    else:
        train_output.append(1)
train_filenames.pop()
train_output.pop()

for i in t3['file']:
    test_filenames.append(i)
for i in t3['output']:
    if (i == 'N'):
        test_output.append(0)
    else:
        test_output.append(1)


def unzip_dataset():
    for x in os.listdir(work_path + os.sep + zip_folder):
        p = work_path + os.sep + zip_folder + os.sep + x
        if p.endswith(".zip"):
            zip_temp = zipfile.ZipFile(p, 'r')
            zip_temp.extractall(unzip_folder)
            zip_temp.close()


def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review, "html.parser").get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return (" ".join(meaningful_words))


def get_train_data():
    posts = []
    for i in train_filenames:
        tree = ET.parse(unzip_folder + os.sep + str(i) + ".xml")
        root = tree.getroot()
        temp = []
        for child_of_root in root:
            for child in child_of_root:
                if (child.tag == 'body'):
                    if (child.text is not None):
                        temp.append(child.text)
        posts.append(temp)
    return posts


def get_test_data():
    posts = []
    for i in test_filenames:
        tree = ET.parse(unzip_folder + os.sep + str(i) + ".xml")
        root = tree.getroot()
        temp = []
        for child_of_root in root:
            for child in child_of_root:
                if (child.tag == 'body'):
                    if (child.text is not None):
                        temp.append(child.text)  # place where we should process the words
        posts.append(temp)
    return posts

def get_data():
	train_data = get_train_data()
	test_data = get_test_data()

	for i in range(0, len(train_data)):
	    train_data[i] = " ".join(train_data[i])
	for i in range(0, len(test_data)):
	    test_data[i] = " ".join(test_data[i])
	return train_data,test_data,train_output,test_output
"""
def read_files():
    posts = []
    c = 0
    user_id = []
    for i in os.listdir(work_path + os.sep + unzip_folder):
        for j in os.listdir(work_path + os.sep + unzip_folder + os.sep + i):
            xml_file = work_path + os.sep + unzip_folder + os.sep + i + os.sep + j
            if xml_file.endswith(".xml"):
                tree = ET.parse(xml_file)
                root = tree.getroot()
                for child_of_root in root:
                    for child in child_of_root:
                        if (child.tag == 'user'):
                            user_id.append(child.attrib)
                        if (child.tag == 'body'):
                            temp = []
                            if (child.text is not None):
                                temp = (child.text).split()  # place where we should process the words
                            posts.append(temp)
                    c += 1
    print "Total No. of posts loaded : ", c
    return posts, user_id


posts, user_id = read_files()
print len(posts)


print "Cleaning and parsing the posts for training...\n"
num_posts = len(train_data)
clean_train_posts = []
for i in xrange(0, num_posts):
    if ((i + 1) % 1000 == 0):
        print "Review %d of %d\n" % (i + 1, num_posts)
    clean_train_posts.append(review_to_words(str(train_data[i].split())))

print "Creating the bag of words for training dataset...\n"
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=2000)
train_data_features = vectorizer.fit_transform(clean_train_posts)
train_data_features = train_data_features.toarray()

print "Cleaning and parsing the posts for testing...\n"
num_posts = len(test_data)
clean_test_posts = []
for i in xrange(0, num_posts):
    if ((i + 1) % 1000 == 0):
        print "Review %d of %d\n" % (i + 1, num_posts)
    clean_test_posts.append(review_to_words(str(train_data[i].split())))

print "Creating the bag of words for testing dataset...\n"
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=2000)
test_data_features = vectorizer.fit_transform(clean_test_posts)
test_data_features = test_data_features.toarray()

print "Training the random forest..."
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


clf = svm.SVC(tol=0.000001)
clf.fit(train_data_features, train_output)

forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data_features, train_output)

result = forest.predict(test_data_features)
result1 = clf.predict(test_data_features)
p = 0
p1 = 0
for i in range(0, len(result)):
    print result[i], " ", test_output[i]
    if (result[i] == test_output[i]):
        p += 1
    if (result1[i] == test_output[i]):
        p1 += 1

accuracy = p / float(len(result))
accuracy1 = p1 / float(len(result))
print "Prediction accuracy :",
print accuracy
print accuracy1
"""
