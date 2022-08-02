#%%LST model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV 

# Load the dataset.
df_feeds = pd.read_csv("train.csv")
df_feeds['discourse_text'].isnull().sum() # checks for NAs

import string
string.punctuation 

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import word_tokenize
from collections import Counter
from itertools import chain

# Remove stopwords from corpus stopwords dictionary to prevent commonly used language or text such as 'a', 'the', etc, that may affect the predictive performance.
stop = set(stopwords.words('english')) 
df_feeds['discourse_text'].replace("[^a-zA-Z]"," ", regex=True, inplace =True)  # match strings that contains non-letter and replace with black to remove string.punctuations.
df_feeds['discourse_text'] = df_feeds['discourse_text'].str.lower() # converts strings to lower case.
print(df_feeds['discourse_text'])
df_feeds['discourse_text'] = df_feeds['discourse_text'].apply(lambda x: [item for item in str(x).split() if item not in stop])
freq = pd.Series(Counter(chain.from_iterable(df_feeds['discourse_text']))).sort_values(ascending=False) # count the frequencies of words.


# Lemmatize all tokens into a new list to prevent overfitting of dataset when verb, nouns, adverb or adjectives are not as concerning on the impact of effectiveness of an argument.
wordnet_lem = WordNetLemmatizer()
df_feeds['discourse_text'] = df_feeds['discourse_text'].apply(
                    lambda lst:[wordnet_lem.lemmatize(word, pos='v') for word in lst])
df_feeds['discourse_text'] = df_feeds['discourse_text'].apply(
                    lambda lst:[wordnet_lem.lemmatize(word, pos='n') for word in lst])
df_feeds['discourse_text'] = df_feeds['discourse_text'].apply(
                    lambda lst:[wordnet_lem.lemmatize(word, pos='r') for word in lst])
df_feeds['discourse_text'] = df_feeds['discourse_text'].apply(
                    lambda lst:[wordnet_lem.lemmatize(word, pos='a') for word in lst])
freq_final = pd.Series(Counter(chain.from_iterable(df_feeds['discourse_text']))).sort_values(ascending=False)



# Visualize top 30 word frequencies.
import seaborn as sns 
import matplotlib.pyplot as plt
top_words = freq_final.head(30).reset_index()

sns.reset_orig()
plt.figure(figsize = (8,6))
my_palette = sns.color_palette("colorblind") # variations of default palette: deep, muted, pastel, bright, dark, colorblind. 
plt.style.use('seaborn-colorblind')
sns.set(rc={'figure.figsize':(10,5)})
sns.barplot(x=top_words.iloc[:,1], y=top_words.iloc[:,0],data=top_words , alpha = 0.6).set_title('Top 20 count of word frequencies')
plt.xlabel('Frequencies')
plt.ylabel('Words')
plt.show()

# Visualize data to see if the dataset has a normal distribution
counts = df_feeds.discourse_effectiveness.value_counts()
print(counts)
print("\nPredicting only 0 = {:.2f}% accuracy".format(counts[0] / sum(counts) * 100))
print("\nPredicting only 1 = {:.2f}% accuracy".format(counts[1] / sum(counts) * 100))
print("\nPredicting only 2 = {:.2f}% accuracy".format(counts[2] / sum(counts) * 100))

df_counts = pd.DataFrame(counts).reset_index()
plt.figure(figsize = (10,5))
sns.barplot(x=df_counts['index'],y=df_counts['discourse_effectiveness'],data=df_counts,alpha = 0.6).set_title('Total count of Effectiveness')
plt.xlabel('Effectiveness')
plt.ylabel('Count')
plt.show()
# Seems like there are more Adequate feedbacks than Effective and Ineffective, indicating that the data may be imbalanced and may be prone to lower prediction accuracy. 

# One-hot-encoding by creating dummies to categorical data.
df_effects = pd.get_dummies(df_feeds.iloc[:,4])
df_feeds = pd.concat([df_feeds, df_effects], axis=1) # combine dummy rows.

# Train-test validation approach.
X_train, X_test, y_train, y_test = train_test_split(df_feeds['discourse_text'].values, df_feeds[['Adequate', 'Effective','Ineffective']].values, stratify=df_feeds['discourse_effectiveness'],test_size=0.1, random_state=100)


# Tokenize 500 most common words to prevent overftitting from noise text that least occur.
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=500, oov_token='x')
word_index = tokenizer.word_index
count_words = tokenizer.word_counts
tokenizer.fit_on_texts(X_train) 
tokenizer.fit_on_texts(X_test)
seq_train = tokenizer.texts_to_sequences(X_train)
seq_test = tokenizer.texts_to_sequences(X_test)
pad_train = pad_sequences(seq_train) 
pad_test = pad_sequences(seq_test)

# Shuffle train set after splitting to improve or avoid overfitting and ensure data are representatives.
from sklearn.utils import shuffle
pad_train, y_train = shuffle(pad_train, y_train)
print(pad_train[9])


import tensorflow as tf
from tensorflow.keras import regularizers
import keras
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential 
import tensorflow
from tensorflow.keras.layers import Dropout

# Create LST model and add neuron layers to easily define relationship of the output classess. Last layer based on number of desired categorical/classes output of interest.
# Linear regularization used in optimizer settings to prevent overfitting of data.
lst_mod = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=500, output_dim=100),
    tf.keras.layers.SpatialDropout1D(0.3),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation='softmax')
])

optimizer = keras.optimizers.Adam(lr=0.001)
lst_mod.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
lst_mod.summary()

# Implementing early stopper to prevent overfitting of data that may occur on validation accuracy.
class earlystop(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}): 
    if(logs.get('val_accuracy')>0.855):
      print("Accuracy has reached > 85.5%!") 
      self.model.stop_training = True
es = earlystop() 

# After numerous trial runs, epochs = 20 provides constant increase in test accuracy. If epochs > 20, LST model may overfit, where train accuracy graduaually increasing while validation accuracy decreases.
lst_mod1 = lst_mod.fit(pad_train, y_train, epochs=20, callbacks=[es],
            validation_data=(pad_test, y_test), verbose=1, batch_size=100)

# Visualize overview accuracy and validation accuracy of individual epochs of the LST model.
# Note: Saving LST model into a variable allows the model to be visualized on graph, else it would return an error where history is not callable.
plt.figure(figsize = (15,10))
plt.plot(lst_mod1.history['accuracy'])
plt.plot(lst_mod1.history['val_accuracy'])
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# Predict LST model on padded test set.
y_predict = lst_mod.predict(pad_test, verbose=0)
print(y_predict)


# Confusion matrix 
import sklearn.metrics as metrics
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
test_cm = metrics.confusion_matrix(y_test.argmax(axis=1), y_predict.argmax(axis=1))
print(test_cm)
test_score = metrics.accuracy_score(y_test.argmax(axis=1), y_predict.argmax(axis=1))
print(test_score)
# overall accuracy ~60%++ depending on the random train-test split.
test_report = metrics.classification_report(y_test.argmax(axis=1), y_predict.argmax(axis=1))
print(test_report)
# It can be seen that basesd on classification report, 0: 'Adequate' has the highest overall accuracy in terms of precision (actual true positive ouf of predicted positive), recall (true positive rate) and f1-score (mean of precision and recall, taking consideration of false positive and false negatives)
# It can also be observed that 'Adequate' has the highest number of observations supporting its accuracy.
# Followed by 1:'Effective' with the second highest accuracy and 2: 'Ineffective' as the least overall accuracy. 


# Visualize Confusion Matrix
test_cm = pd.DataFrame(test_cm, range(3), range(3))
plt.figure(figsize = (10,8))
sns.set(font_scale=2)
sns.heatmap(test_cm, annot=True, annot_kws={"size": 21},fmt='d')
plt.show()

# Export final test.csv dataset for submission.
df_test_predict = pd.DataFrame(y_predict, columns=['Adequate','Effective','Ineffective'])
final_test_dataset = pd.merge(df_feeds, df_test_predict, how = 'right', left_index= True ,right_index =True)
final_test_dataset = final_test_dataset[['discourse_id', 'Adequate_y', 'Effective_y','Ineffective_y']]
final_test_dataset.rename(columns={'Adequate_y' : 'Adequate', 'Effective_y' : 'Effective', 'Ineffective_y' : 'Ineffective'}, inplace=True)
final_test_dataset.to_csv('D:/Kaggle/Predictive arguments/test.csv')




