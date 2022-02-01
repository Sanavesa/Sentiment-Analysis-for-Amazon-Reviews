# Sentiment Analysis for Amazon Reviews
The goal of this project was to train a few models to classify customer sentiment given a product review, which was trained on Amazon Reviews [here](https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Kitchen_v1_00.tsv.gz). This dataset is extremely huge, which helped out in the training phase.

## Data Pre-processing
As is known, models have difficulty learning meaningful patterns from raw, unprocessed data. Thus, I cleaned the data and ensured the data was balanced so that the learning has no problems. The following steps were taken to clean the data:
<ul>
  <li>Lowercase all the reviews</li>
  <li>Remove all HTML tags and URLs</li>
  <li>Remove non-English alphanumeric characters and extra spaces</li>
  <li>Expand contractions (won't became will not)</li>
  <li>Remove stop words</li>
  <li>Perform lemmatization</li>
</ul>

The average character length of the reviews were reduced from 309 to 183 characters (down by 40%).

Next, I randomly sampled 100k instances of each class (positive and negative sentiment) to be the final processed dataset. By doing this, the model will be fitted on a high-quality dataset that is well-balanced.

## Feature Extraction
So far the reviews are still in their textual format. By utilizing Sklearn, I was able to extract features from the reviews using TF-IDF, which is a value that is intended to reflect how important a word is to a dataset. This step converted each review from its textual format into an array of floats.

## Training
A total of 4 models were trained using Sklearn: Perceptron, Linear SVM, Logistic Regression, and Multinomial Naive Bayes. The following are the prediction results against the testing set after training the models:

| Model | Precision | Recall | Accuracy | F1 score |
| ----- | -------- | --------- | ------ | -------- |
| Perceptron | 85% | 89% | 85% | 84 |
| Multinomial Naive Bayes | 87% | 87% | 87% | 87 |
| Linear SVC | 89% | 90% | 89% | 89 |
| Logistic Regression | 90% | 90% | 90% |  90 |
