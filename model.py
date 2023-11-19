
import pandas as pd # use for data manipulation and analysis
import numpy as np # use for multi-dimensional array and matrix

import seaborn as sns # use for high-level interface for drawing attractive and informative statistical graphics 
import matplotlib.pyplot as plt # It provides an object-oriented API for embedding plots into applications
%matplotlib inline 
# It sets the backend of matplotlib to the 'inline' backend:
import time # calculate time 

from sklearn.linear_model import LogisticRegression # algo use to predict good or bad
from sklearn.naive_bayes import MultinomialNB # nlp algo use to predict good or bad

from sklearn.model_selection import train_test_split # spliting the data between feature and target
from sklearn.metrics import classification_report # gives whole report about metrics (e.g, recall,precision,f1_score,c_m)
from sklearn.metrics import confusion_matrix # gives info about actual and predict
from nltk.tokenize import RegexpTokenizer # regexp tokenizers use to split words from text  
from nltk.stem.snowball import SnowballStemmer # stemmes words
from sklearn.feature_extraction.text import CountVectorizer # create sparse matrix of words using regexptokenizes  
from sklearn.pipeline import make_pipeline # use for combining all prerocessors techniuqes and algos

from PIL import Image # getting images in notebook
# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator# creates words colud

from bs4 import BeautifulSoup # use for scraping the data from website
from selenium import webdriver # use for automation chrome 
import networkx as nx # for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.

import pickle# use to dump model 

import warnings # ignores pink warnings 
warnings.filterwarnings('ignore')
Did some surfing and found some websites offering malicious links. And found some datasets
phishing_data1 = pd.read_csv('phishing_urls.csv',usecols=['domain','label'],encoding='latin1', error_bad_lines=False)
phishing_data1.columns = ['URL','Label']
phishing_data2 = pd.read_csv('phishing_data.csv')
phishing_data2.columns = ['URL','Label']
phishing_data3 = pd.read_csv('phishing_data2.csv')
phishing_data3.columns = ['URL','Label']
for l in range(len(phishing_data1.Label)):
    if phishing_data1.Label.loc[l] == '1.0':
        phishing_data1.Label.loc[l] = 'bad'
    else:
        phishing_data1.Label.loc[l] = 'good'
Concatenate All datasets in one.
frames = [phishing_data1, phishing_data2, phishing_data3]
phishing_urls = pd.concat(frames)
#saving dataset
phishing_urls.to_csv(r'phishing_site_urls.csv', index = False)
Loading the main dataset.
phish_data = pd.read_csv('phishing_site_urls.csv')
* You can download dataset from my Kaggle Profile here
phish_data.head()
URL	Label
0	nobell.it/70ffb52d079109dca5664cce6f317373782/...	bad
1	www.dghjdgf.com/paypal.co.uk/cycgi-bin/webscrc...	bad
2	serviciosbys.com/paypal.cgi.bin.get-into.herf....	bad
3	mail.printakid.com/www.online.americanexpress....	bad
4	thewhiskeydregs.com/wp-content/themes/widescre...	bad
phish_data.tail()
URL	Label
549341	23.227.196.215/	bad
549342	apple-checker.org/	bad
549343	apple-iclods.org/	bad
549344	apple-uptoday.org/	bad
549345	apple-search.info	bad
phish_data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 549346 entries, 0 to 549345
Data columns (total 2 columns):
URL      549346 non-null object
Label    549346 non-null object
dtypes: object(2)
memory usage: 8.4+ MB
About dataset
Data is containg 5,49,346 unique entries.
There are two columns.
Label column is prediction col which has 2 categories A. Good - which means the urls is not containing malicious stuff and this site is not a Phishing Site. B. Bad - which means the urls contains malicious stuffs and this site isa Phishing Site.
There is no missing value in the dataset.
phish_data.isnull().sum() # there is no missing values
URL      0
Label    0
dtype: int64
Since it is classification problems so let's see the classes are balanced or imbalances
#create a dataframe of classes counts
label_counts = pd.DataFrame(phish_data.Label.value_counts())
#visualizing target_col
sns.set_style('darkgrid')
sns.barplot(label_counts.index,label_counts.Label)
<matplotlib.axes._subplots.AxesSubplot at 0x15e4f1821f0>

Preprocessing
Now that we have the data, we have to vectorize our URLs. I used CountVectorizer and gather words using tokenizer, since there are words in urls that are more important than other words e.g ‘virus’, ‘.exe’ ,’.dat’ etc. Lets convert the URLs into a vector form.
RegexpTokenizer
A tokenizer that splits a string using a regular expression, which matches either the tokens or the separators between tokens.
tokenizer = RegexpTokenizer(r'[A-Za-z]+')
phish_data.URL[0]
'nobell.it/70ffb52d079109dca5664cce6f317373782/login.SkyPe.com/en/cgi-bin/verification/login/70ffb52d079109dca5664cce6f317373/index.php?cmd=_profile-ach&outdated_page_tmpl=p/gen/failed-to-load&nav=0.5.1&login_access=1322408526'
# this will be pull letter which matches to expression
tokenizer.tokenize(phish_data.URL[0]) # using first row
['nobell',
 'it',
 'ffb',
 'd',
 'dca',
 'cce',
 'f',
 'login',
 'SkyPe',
 'com',
 'en',
 'cgi',
 'bin',
 'verification',
 'login',
 'ffb',
 'd',
 'dca',
 'cce',
 'f',
 'index',
 'php',
 'cmd',
 'profile',
 'ach',
 'outdated',
 'page',
 'tmpl',
 'p',
 'gen',
 'failed',
 'to',
 'load',
 'nav',
 'login',
 'access']
print('Getting words tokenized ...')
t0= time.perf_counter()
phish_data['text_tokenized'] = phish_data.URL.map(lambda t: tokenizer.tokenize(t)) # doing with all rows
t1 = time.perf_counter() - t0
print('Time taken',t1 ,'sec')
Getting words tokenized ...
Time taken 4.382057999999915 sec
phish_data.sample(5)
URL	Label	text_tokenized
123763	harborexpressservices.com/xlrmp/check/	bad	[harborexpressservices, com, xlrmp, check]
520402	cds-chartreuse.fr/locales/sancho.tar.gz	bad	[cds, chartreuse, fr, locales, sancho, tar, gz]
545878	adminzv.ru/xaRAUXHZ9r/oPOtm6hU7.php	bad	[adminzv, ru, xaRAUXHZ, r, oPOtm, hU, php]
84771	www.hootech.com/WinTail/	good	[www, hootech, com, WinTail]
60932	bellsouthpwp.net/g/y/gypsyfairy32425/	good	[bellsouthpwp, net, g, y, gypsyfairy]
SnowballStemmer
Snowball is a small string processing language, gives root words
stemmer = SnowballStemmer("english") # choose a language
print('Getting words stemmed ...')
t0= time.perf_counter()
phish_data['text_stemmed'] = phish_data['text_tokenized'].map(lambda l: [stemmer.stem(word) for word in l])
t1= time.perf_counter() - t0
print('Time taken',t1 ,'sec')
Getting words stemmed ...
Time taken 178.71395260000008 sec
phish_data.sample(5)
URL	Label	text_tokenized	text_stemmed
314008	detroitlionsticket.com/	good	[detroitlionsticket, com]	[detroitlionsticket, com]
502478	pertclinic.com/qsipz9g	bad	[pertclinic, com, qsipz, g]	[pertclin, com, qsipz, g]
423029	revolutionmyspace.com/pictures-1/ami_dolenz	good	[revolutionmyspace, com, pictures, ami, dolenz]	[revolutionmyspac, com, pictur, ami, dolenz]
236352	ryanscottfrost.com/	good	[ryanscottfrost, com]	[ryanscottfrost, com]
227770	photobucket.com/images/matsumoto%20jun/	good	[photobucket, com, images, matsumoto, jun]	[photobucket, com, imag, matsumoto, jun]
print('Getting joiningwords ...')
t0= time.perf_counter()
phish_data['text_sent'] = phish_data['text_stemmed'].map(lambda l: ' '.join(l))
t1= time.perf_counter() - t0
print('Time taken',t1 ,'sec')
Getting joiningwords ...
Time taken 0.7240019999999276 sec
phish_data.sample(5)
URL	Label	text_tokenized	text_stemmed	text_sent
342349	freebase.com/view/en/new_zealand_womens_nation...	good	[freebase, com, view, en, new, zealand, womens...	[freebas, com, view, en, new, zealand, women, ...	freebas com view en new zealand women nation s...
353662	homefinder.com/CA/Torrance/61177522d_5500_Torr...	good	[homefinder, com, CA, Torrance, d, Torrance, B...	[homefind, com, ca, torranc, d, torranc, blvd, b]	homefind com ca torranc d torranc blvd b
483946	ib.adnxs.com/tt?id=2063435&size=728x90&referre...	bad	[ib, adnxs, com, tt, id, size, x, referrer, ht...	[ib, adnx, com, tt, id, size, x, referr, http,...	ib adnx com tt id size x referr http www world...
519398	system-check-abevbrye.in/	bad	[system, check, abevbrye, in]	[system, check, abevbry, in]	system check abevbry in
153443	bsbiz.eu/?p=278	good	[bsbiz, eu, p]	[bsbiz, eu, p]	bsbiz eu p
Visualization
1. Visualize some important keys using word cloud

#sliceing classes
bad_sites = phish_data[phish_data.Label == 'bad']
good_sites = phish_data[phish_data.Label == 'good']
bad_sites.head()
URL	Label	text_tokenized	text_stemmed	text_sent
0	nobell.it/70ffb52d079109dca5664cce6f317373782/...	bad	[nobell, it, ffb, d, dca, cce, f, login, SkyPe...	[nobel, it, ffb, d, dca, cce, f, login, skype,...	nobel it ffb d dca cce f login skype com en cg...
1	www.dghjdgf.com/paypal.co.uk/cycgi-bin/webscrc...	bad	[www, dghjdgf, com, paypal, co, uk, cycgi, bin...	[www, dghjdgf, com, paypal, co, uk, cycgi, bin...	www dghjdgf com paypal co uk cycgi bin webscrc...
2	serviciosbys.com/paypal.cgi.bin.get-into.herf....	bad	[serviciosbys, com, paypal, cgi, bin, get, int...	[serviciosbi, com, paypal, cgi, bin, get, into...	serviciosbi com paypal cgi bin get into herf s...
3	mail.printakid.com/www.online.americanexpress....	bad	[mail, printakid, com, www, online, americanex...	[mail, printakid, com, www, onlin, americanexp...	mail printakid com www onlin americanexpress c...
4	thewhiskeydregs.com/wp-content/themes/widescre...	bad	[thewhiskeydregs, com, wp, content, themes, wi...	[thewhiskeydreg, com, wp, content, theme, wide...	thewhiskeydreg com wp content theme widescreen...
good_sites.head()
URL	Label	text_tokenized	text_stemmed	text_sent
18231	esxcc.com/js/index.htm?us.battle.net/noghn/en/...	good	[esxcc, com, js, index, htm, us, battle, net, ...	[esxcc, com, js, index, htm, us, battl, net, n...	esxcc com js index htm us battl net noghn en r...
18232	wwweira¯&nvinip¿ncH¯wVö%ÆåyDaHðû/ÏyEùuË\nÓ6...	good	[www, eira, nvinip, ncH, wV, yDaH, yE, u, rT, ...	[www, eira, nvinip, nch, wv, ydah, ye, u, rt, ...	www eira nvinip nch wv ydah ye u rt u g m i xz...
18233	'www.institutocgr.coo/web/media/syqvem/dk-óij...	good	[www, institutocgr, coo, web, media, syqvem, d...	[www, institutocgr, coo, web, media, syqvem, d...	www institutocgr coo web media syqvem dk ij r ...
18234	Yìê‡koãÕ»Î§DéÎl½ñ¡ââqtò¸/à; Í	good	[Y, ko, D, l, qt]	[y, ko, d, l, qt]	y ko d l qt
18236	ruta89fm.com/images/AS@Vies/1i75cf7b16vc<Fd16...	good	[ruta, fm, com, images, AS, Vies, i, cf, b, vc...	[ruta, fm, com, imag, as, vie, i, cf, b, vc, f...	ruta fm com imag as vie i cf b vc f d b g sd v...
create a function to visualize the important keys from url
def plot_wordcloud(text, mask=None, max_words=400, max_font_size=120, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'com','http'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='white',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    mask = mask)
    wordcloud.generate(text)
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'green', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  
data = good_sites.text_sent
data.reset_index(drop=True, inplace=True)
common_text = str(data)
common_mask = np.array(Image.open('star.png'))
plot_wordcloud(common_text, common_mask, max_words=400, max_font_size=120, 
               title = 'Most common words use in good urls', title_size=15)

data = bad_sites.text_sent
data.reset_index(drop=True, inplace=True)
common_text = str(data)
common_mask = np.array(Image.open('comment.png'))
plot_wordcloud(common_text, common_mask, max_words=400, max_font_size=120, 
               title = 'Most common words use in bad urls', title_size=15)

Download more various type of images here

2. Visualize internal links, it will shows all redirect links.

Scrape any website
First, setting up the Chrome webdriver so we can scrape dynamic web pages.
Chrome webdriver
WebDriver tool use for automated testing of webapps across many browsers. It provides capabilities for navigating to web pages, user input and more
browser = webdriver.Chrome(r"chromedriver.exe")
You can download chromedriver.exe from my github here

After set up the Chrome driver create two lists.
First list named list_urls holds all the pages you’d like to scrape.
Second, create an empty list where you’ll append links from each page.
list_urls = ['https://www.ezeephones.com/','https://www.ezeephones.com/about-us'] #here i take phishing sites 
links_with_text = []
I took some phishing site to see were the hackers redirect(on different link) us.
Use the BeautifulSoup library to extract only relevant hyperlinks for Google, i.e. links only with '<'a'>' tags with href attributes.
BeautifulSoup
It is use for getting data out of HTML, XML, and other markup languages.
for url in list_urls:
    browser.get(url)
    soup = BeautifulSoup(browser.page_source,"html.parser")
    for line in soup.find_all('a'):
        href = line.get('href')
        links_with_text.append([url, href])
Turn the URL’s into a Dataframe
After you get the list of your websites with hyperlinks turn them into a Pandas DataFrame with columns “from” (URL where the link resides) and “to” (link destination URL)
df = pd.DataFrame(links_with_text, columns=["from", "to"])
df.head()
from	to
0	https://www.ezeephones.com/	None
1	https://www.ezeephones.com/	https://www.ezeephones.com/
2	https://www.ezeephones.com/	/cart
3	https://www.ezeephones.com/	/category/notch-phones
4	https://www.ezeephones.com/	/category/Deals - Of The Day
Step 3: Draw a graph
Finally, use the aforementioned DataFrame to visualize an internal link structure by feeding it to the Networkx method from_pandas_edgelist first and draw it by calling nx.draw
GA = nx.from_pandas_edgelist(df, source="from", target="to")
nx.draw(GA, with_labels=False)

Creating Model
CountVectorizer
CountVectorizer is used to transform a corpora of text to a vector of term / token counts.
#create cv object
cv = CountVectorizer()
#help(CountVectorizer())
feature = cv.fit_transform(phish_data.text_sent) #transform all text which we tokenize and stemed
feature[:5].toarray() # convert sparse matrix into array to print transformed features
array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0]], dtype=int64)
* Spliting the data
trainX, testX, trainY, testY = train_test_split(feature, phish_data.Label)
LogisticRegression
Logistic Regression is a Machine Learning classification algorithm that is used to predict the probability of a categorical dependent variable. In logistic regression, the dependent variable is a binary variable that contains data coded as 1 (yes, success, etc.) or 0 (no, failure, etc.). In other words, the logistic regression model predicts P(Y=1) as a function of X.
# create lr object
lr = LogisticRegression()
lr.fit(trainX,trainY)
LogisticRegression()
lr.score(testX,testY)
0.9636514559077306
.*** Logistic Regression is giving 96% accuracy, Now we will store scores in dict to see which model perform best**

Scores_ml = {}
Scores_ml['Logistic Regression'] = np.round(lr.score(testX,testY),2)
print('Training Accuracy :',lr.score(trainX,trainY))
print('Testing Accuracy :',lr.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(lr.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(lr.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")
Training Accuracy : 0.9782480479795345
Testing Accuracy : 0.9636514559077306

CLASSIFICATION REPORT

              precision    recall  f1-score   support

         Bad       0.90      0.97      0.93     36597
        Good       0.99      0.96      0.97    100740

    accuracy                           0.96    137337
   macro avg       0.95      0.96      0.95    137337
weighted avg       0.97      0.96      0.96    137337


CONFUSION MATRIX
<matplotlib.axes._subplots.AxesSubplot at 0x1ec84c387c8>

MultinomialNB
Applying Multinomial Naive Bayes to NLP Problems. Naive Bayes Classifier Algorithm is a family of probabilistic algorithms based on applying Bayes' theorem with the “naive” assumption of conditional independence between every pair of a feature.
# create mnb object
mnb = MultinomialNB()
mnb.fit(trainX,trainY)
MultinomialNB()
mnb.score(testX,testY)
0.9574550194048217
*** MultinomialNB gives us 95% accuracy**

Scores_ml['MultinomialNB'] = np.round(mnb.score(testX,testY),2)
print('Training Accuracy :',mnb.score(trainX,trainY))
print('Testing Accuracy :',mnb.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(mnb.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(mnb.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")
Training Accuracy : 0.9741437687040817
Testing Accuracy : 0.9574550194048217

CLASSIFICATION REPORT

              precision    recall  f1-score   support

         Bad       0.91      0.94      0.92     38282
        Good       0.98      0.97      0.97     99055

    accuracy                           0.96    137337
   macro avg       0.94      0.95      0.95    137337
weighted avg       0.96      0.96      0.96    137337


CONFUSION MATRIX
<matplotlib.axes._subplots.AxesSubplot at 0x1ec84f9c908>

acc = pd.DataFrame.from_dict(Scores_ml,orient = 'index',columns=['Accuracy'])
sns.set_style('darkgrid')
sns.barplot(acc.index,acc.Accuracy)
<matplotlib.axes._subplots.AxesSubplot at 0x1ec84f71348>

*** So, Logistic Regression is the best fit model, Now we make sklearn pipeline using Logistic Regression**

pipeline_ls = make_pipeline(CountVectorizer(tokenizer = RegexpTokenizer(r'[A-Za-z]+').tokenize,stop_words='english'), LogisticRegression())
##(r'\b(?:http|ftp)s?://\S*\w|\w+|[^\w\s]+') ([a-zA-Z]+)([0-9]+) -- these tolenizers giving me low accuray 
trainX, testX, trainY, testY = train_test_split(phish_data.URL, phish_data.Label)
pipeline_ls.fit(trainX,trainY)
Pipeline(steps=[('countvectorizer',
                 CountVectorizer(stop_words='english',
                                 tokenizer=<bound method RegexpTokenizer.tokenize of RegexpTokenizer(pattern='[A-Za-z]+', gaps=False, discard_empty=True, flags=<RegexFlag.UNICODE|DOTALL|MULTILINE: 56>)>)),
                ('logisticregression', LogisticRegression())])
pipeline_ls.score(testX,testY) 
0.9674450439430016
print('Training Accuracy :',pipeline_ls.score(trainX,trainY))
print('Testing Accuracy :',pipeline_ls.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(pipeline_ls.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(pipeline_ls.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")
Training Accuracy : 0.9808911941244002
Testing Accuracy : 0.9674450439430016

CLASSIFICATION REPORT

              precision    recall  f1-score   support

         Bad       0.92      0.97      0.94     36841
        Good       0.99      0.97      0.98    100496

    accuracy                           0.97    137337
   macro avg       0.95      0.97      0.96    137337
weighted avg       0.97      0.97      0.97    137337


CONFUSION MATRIX
<matplotlib.axes._subplots.AxesSubplot at 0x1ec83fca588>

pickle.dump(pipeline_ls,open('phishing.pkl','wb'))
loaded_model = pickle.load(open('phishing.pkl', 'rb'))
result = loaded_model.score(testX,testY)
print(result)
0.9674450439430016
*That’s it. See, it's that simple yet so effective. We get an accuracy of 98%. That’s a very high value for a machine to be able to detect a malicious URL with. Want to test some links to see if the model gives good predictions? Sure. Let's do it

* Bad links => this are phishing sites
yeniik.com.tr/wp-admin/js/login.alibaba.com/login.jsp.php
fazan-pacir.rs/temp/libraries/ipad
www.tubemoviez.exe
svision-online.de/mgfi/administrator/components/com_babackup/classes/fx29id1.txt

* Good links => this are not phishing sites
www.youtube.com/
youtube.com/watch?v=qI0TQJI3vdU
www.retailhellunderground.com/
restorevisioncenters.com/html/technology.html
predict_bad = ['yeniik.com.tr/wp-admin/js/login.alibaba.com/login.jsp.php','fazan-pacir.rs/temp/libraries/ipad','tubemoviez.exe','svision-online.de/mgfi/administrator/components/com_babackup/classes/fx29id1.txt']
predict_good = ['youtube.com/','youtube.com/watch?v=qI0TQJI3vdU','retailhellunderground.com/','restorevisioncenters.com/html/technology.html']
loaded_model = pickle.load(open('phishing.pkl', 'rb'))
#predict_bad = vectorizers.transform(predict_bad)
# predict_good = vectorizer.transform(predict_good)
result = loaded_model.predict(predict_bad)
result2 = loaded_model.predict(predict_good)
print(result)
print("*"*30)
print(result2)
['bad' 'bad' 'bad' 'bad']
******************************
['good' 'good' 'good' 'good']
