# Amazon Fine Food Reviews

**Data includes**:

* Reviews from Oct 1999 - Oct 2012
* 568,454 reviews
* 256,059 users
* 74,258 products
* 260 users with > 50 reviews

**Columns**
* Id
* ProductId - unique identifier for the product
* UserId - unqiue identifier for the user
* ProfileName
* HelpfulnessNumerator - number of users who found the review helpful
* HelpfulnessDenominator - number of users who indicated whether they found the review helpful or not
* Score - rating between 1 and 5
* Time - timestamp for the review
* Summary - brief summary of the review
* Text - text of the review

**Objective**: In order to train a classifier to detect whether the review is positive or negative we need to have a label column. In the given set such info not available but, fortunately, we got a column of scores/rating which is close approximation of user liked or not liked the product. Scores less than 3 (Rating 2 or 1) we are pretty that label is negative and for more than 3 (Rating 3 or 4) it's positive. But, we are sure about examples with score 3 (kind of neutral). so we drop those rows.



## Feature tranformation:
summary = 'count_letters', 'count_word', 'count_unique_word'