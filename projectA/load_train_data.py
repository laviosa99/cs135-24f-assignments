import numpy as np
import pandas as pd
import os
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.metrics
import sklearn.model_selection
import sklearn.feature_extraction

from nltk.corpus import stopwords


if __name__ == '__main__':
    data_dir = 'data_reviews'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))

    N, n_cols = x_train_df.shape
    print("Shape of x_train_df: (%d, %d)" % (N,n_cols))
    print("Shape of y_train_df: %s" % str(y_train_df.shape))

    # Print out the first five rows and last five rows
    tr_text_list = x_train_df['text'].values.tolist()
    rows = np.arange(0, 5)
    for row_id in rows:
        text = tr_text_list[row_id]
        print("row %5d | y = %d | %s" % (row_id, y_train_df.values[row_id,0], text))

    print("...")
    rows = np.arange(N - 5, N)
    for row_id in rows:
        text = tr_text_list[row_id]
        print("row %5d | y = %d | %s" % (row_id, y_train_df.values[row_id,0], text))


def make_sentiment_analyzer_pipeline(degree=1, alpha=1.0):
    pipeline = sklearn.pipeline.Pipeline(
        steps=[
         ('vectorizer', sklearn.feature_extraction.text.CountVectorizer(
             lowercase=True, # make the text uniformly lowercase
             stop_words=stopwords.words('english'), # remove filler words ('a', 'the', etc.) present in the stopwords nltk library
             analyzer='word', # breakdown text into words for feature analysis
             min_df=0.10, # ignore words with a frequency strictly lower than 10%
             max_df=0.50, # ignore words with a frequency strictly higher than 50%
             token_pattern=r'\b\w+\b' # removes punctuation and numbers
             )
            ),
         ('log_regr', sklearn.linear_model.LogisticRegression(penalty='l2')),
        ])

# make pipeline
pipe = make_sentiment_analyzer_pipeline()

# make hyperparameter C grid (regularization strength)
C_grid = np.asarray([0.0001, 0.01, 1, 100, 1000000])
param_grid = {'C_grid': C_grid}
ncols = len(C_grid)

# perform grid search and fit the model
grid_search = sklearn.model_selection.GridSearchCV(pipe, param_grid=param_grid, cv=5, scoring='roc_auc')

grid_search.fit(x_train_df, y_train_df.values.ravel())

 # Print best parameter and best score
print(f'Best C: {grid_search.best_params_["log_regr__C"]}')
print(f'Best AUROC score: {grid_search.best_score_}')

# Plot the performance of different regularization strengths
scores = grid_search.cv_results_['mean_test_score']

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(C_grid, scores, marker='o')
ax.set_xscale('log')
ax.set_xlabel('C (Inverse Regularization Strength)')
ax.set_ylabel('Mean AUROC')
ax.set_title('Effect of C on AUROC (5-fold CV)')
plt.show()