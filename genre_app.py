from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import os
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding
from tensorflow.keras.layers import SpatialDropout1D, concatenate
from tensorflow.keras.layers import GRU, Bidirectional
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import text, sequence

app = Flask(__name__)

# set objects to be populated in training
model = None
mlb = None
tokenizer = None

# set constants
embedding_file = 'wiki-news-300d-1M.vec'

max_features = 30000
maxlen = 100
embed_size = 300

batch_size = 128
epochs = 20

@app.route('/genres/train', methods=['POST'])
def train():
	decode = request.data.decode('utf-8')
	train = pd.read_csv(io.StringIO(decode))
	
	# create list of unique genres
	genres = set()
	for g in train['genres'].unique():
		if len(g.split(' ')) == 1:
			genres.add(g)
		if len(g.split(' ')) > 1:
			for g_ in g.split(' '):
				genres.add(g_)
	genres = list(genres)

	# create multi label binarizer
	global mlb
	mlb = MultiLabelBinarizer()
	mlb.fit([genres])

	# change genre format from Drama Horror Thriller to [Drama, Horror, Thriller]
	train['genres'] = train['genres'].str.split(" ", expand=False)
	# change genre format from [Drama, Horror, Thriller]
	# to [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, ...
	train['genres'] = train['genres'].apply(lambda x: mlb.transform([x]).flatten())

	# create X_train, y_train
	X_train = train["synopsis"].values
	y_train = np.stack(train['genres'].values)

	# tokenize X_train
	global tokenizer
	tokenizer = text.Tokenizer(num_words=max_features)
	tokenizer.fit_on_texts(list(X_train))
	X_train = tokenizer.texts_to_sequences(X_train)
	x_train = sequence.pad_sequences(X_train, maxlen=maxlen)

	# create embedding
	def get_coefs(word, *arr):
		return word, np.asarray(arr, dtype='float32')
	embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' '))
				for o in open(embedding_file))
	word_index = tokenizer.word_index
	nb_words = min(max_features, len(word_index))
	embedding_matrix = np.zeros((nb_words, embed_size))		
	for word, i in word_index.items():
		if i >= max_features:
			continue
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector

	# create model
	global model
	inp = Input(shape=(maxlen, ))
	x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
	x = SpatialDropout1D(0.2)(x)
	x = Bidirectional(GRU(80, return_sequences=True))(x)
	a = GlobalAveragePooling1D()(x)
	m = GlobalMaxPooling1D()(x)
	conc = concatenate([a, m])
	outp = Dense(len(genres), activation="sigmoid")(conc)
	model = Model(inputs=inp, outputs=outp)
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	# train model
	X_t, X_v, y_t, y_v = train_test_split(x_train, y_train, train_size=0.90)
	early_stopping = EarlyStopping(monitor='val_loss', patience=2,
				       restore_best_weights=True, verbose=1)
	hist = model.fit(X_t, y_t, batch_size=batch_size,
			 epochs=epochs, validation_data=(X_v, y_v),
			 verbose=1, callbacks=[early_stopping])

	return 'model trained'


@app.route('/genres/predict', methods=['POST'])
def pred():
	decode = request.data.decode('utf-8')
	test = pd.read_csv(io.StringIO(decode))
	
	# use trained model to make predictions 
	ids = test['movie_id'].values
	synopsis = test['synopsis'].values
	seq = tokenizer.texts_to_sequences(synopsis)
	pad_seq = sequence.pad_sequences(seq, maxlen=maxlen)
	y_pred = model.predict(pad_seq, batch_size=1024, verbose=1)

	# create top 5 predictions for each test case
	sort_idx = np.argsort(y_pred, axis=1)
	genres_pred = []
	genres_prob = []
	for i, a in enumerate(sort_idx):
		a_idx = a[::-1][0:5]
		genres_pred.append([mlb.classes_[g] for g in a_idx])
		genres_prob.append(y_pred[i][a_idx])

	# create output dataframe
	out = pd.DataFrame({'movie_id': ids,
			    'predicted_genres': genres_pred,
			    'probs': genres_prob,
			    'synopsis': synopsis})
	out['predicted_genres'] = out['predicted_genres'].apply(lambda x: ' '.join(x))
	out = out.drop(columns=['probs', 'synopsis']) 

	# save as csv
	out.to_csv('out.csv', index=False)


if __name__ == '__main__':
    app.run(debug=True)
