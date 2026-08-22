from sklearn.feature_extraction.text import TfidfVectorizer
corpus=[ 
    'this is the first document.',
    'this document is the second document.',
    'and this is the third one.',
    'is this the first document?'
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(X.toarray())
print(vectorizer.get_feature_names_out())
