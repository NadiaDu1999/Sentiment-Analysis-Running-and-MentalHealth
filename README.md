# Running Trend Analysis on Reddit

## 1. Introduction

In the era of social media, we often find ourselves consumed by the lives of others, constantly comparing their highlights with our reality. This habit can take a toll on our mental health. However, like every coin has two sides, social media can also be a source of inspiration. For me personally, it became the motivation to start exercising and take better care of my body. I began running as a way to improve my mental health, and I can confidently say that running has truly changed my life.

In this paper, I imagine myself as part of the marketing team for a non-profit running organization, working to promote a campaign that inspires more people to run. One of the key objectives of this project is to understand how running affects people’s mental well-being and what motivates them to start or continue running. To explore this, I analyse Reddit posts that discuss running and mental health. I also examine the broader context in which people talk about running on Reddit. Additionally, I look at the sentiments expressed, both positive and negative, as a way to understand strong opinions and emotional connections people have with running.

## 2. Data Collection

The data was collected from Reddit using its official API through PRAW (Python Reddit API Wrapper), a Python library that provides easy access to Reddit’s data. To focus on discussions specifically related to running and mental health, I selected seven relevant subreddits: r/running, r/BeginnersRunning,r/
beginnerrunning, r/Marathon_Training, r/AdvancedRunning, r/XXRunning, and r/firstmarathon. I used the some keywords, such as "mental", “therapy” and “burnout”, to narrow down the scope of the search, avoiding unrelated running topics and ensuring relevance to mental health. The plan was to retrieve up to 500 posts from each subreddit, totalling 3,500 posts. However, only 925 posts were successfully saved into the final JSON file. This discrepancy is likely due to the fact that not all subreddits had 500 posts that matched the keyword criteria within the available Reddit archive, especially for smaller or less active subreddits. Additionally, PRAW returns only a subset of all possible results due to Reddit’s search constraints and API limits, which can also contribute to the lower number of retrieved posts (LinkedIn, 2024).

## 3. Data Pre-Processing and Exploration 

##### 3.1 Conversion to lower case

To maintain uniformity and reduce case-related variability in the text data, all text was converted to lowercase (Gorale, 2024). This normalization step ensures that words like "Running" and "running" are treated identically during analysis.

##### 3.2 Noise reduction
To reduce textual noise, punctuation marks and special characters such as . , ; : ? ! " ' - _ ( ) [ ] {
} were removed. This was done using Python’s string.punctuation, which provides a convenient list of common punctuation symbols to filter out irrelevant characters from the data.

##### 3.3 Tokenization

Tokenization is the process of splitting text into individual elements or tokens (usually words). For instance, the sentence "I love running" becomes ["I", "love", "running"]. Instead of the
standard word_tokenize() function, I used NLTK’s TweetTokenizer, which is better suited for informal or social media text. It handles contractions like "I'm" correctly by tokenizing it as ["I", "'m"], preserving meaning in casual language formats. However, if we skip the noise reduction process—which includes steps like removing punctuation and special characters—tokens such as commas and quotation marks (e.g., ",", "“", etc.) will still appear in the list, leading to less meaningful analysis. The example below illustrates this difference clearly.

##### 3.4 Stop word removal

Commonly used words such as "the", "is", and "and" contribute little to semantic analysis and were removed to reduce noise. The stopword list was based on NLTK’s default set, with additional domain- specific terms like "rt", "via", "...", "’", "http", and "https" also excluded. These additions are particularly useful for cleaning Reddit posts, where such tokens are frequent but not meaningful.

##### 3.5 Normalization

To further refine the text, lemmatization was applied to reduce words to their base or dictionary forms. Unlike stemming, which may produce non-standard or partial roots, lemmatization ensures that the output is meaningful and context-aware. For example:

Original Words: running, runner, runs After Stemming: run, run, run

After Lemmatization: running, runner, run

This also explains why the word “run” appeared more frequently in the stemming results
(over 8,000 times) compared to lemmatization (just over 5,000 times). In stemming, different word forms like “runner” or “running” are all reduced to the root “run”, regardless of context. In contrast, lemmatization retains more accurate, context-aware word forms, which leads to a more balanced distribution of related terms.

![Bi(tri)grams](https://github.com/NadiaDu1999/Sentiment-Analysis-Running-and-MentalHealth/blob/main/Bi(Tri)%20grams.png)
Figure 1: Comparison of Top Terms: Stemming vs Lemmatization

This distinction highlights the benefit of lemmatization: while stemming treats all forms as identical, lemmatization preserves important differences—such as between run and runner, which have distinct meanings (Piduguralla, 2023). To evaluate the effectiveness of both approaches, a comparative analysis was performed and summarized in Table 2. The results demonstrate that lemmatization produces more accurate, meaningful, and human-readable outputs, making it more suitable for sentiment analysis in this context.

###### The output from normalization by Stemming and Lemmatizer 
![lemma_stemming](https://github.com/NadiaDu1999/Sentiment-Analysis-Running-and-MentalHealth/blob/main/Lemma%20vs.%20Stemming.png)

As expected, before applying any cleaning or tokenization, the total raw word count was 499,669. After the cleaning and tokenization process, the word count was significantly reduced — down
to 243,693 using stemming and 248,501 using lemmatization. This substantial reduction highlights how much unnecessary or redundant content (such as punctuation, stopwords, or repeated word forms) was present in the original text, and demonstrates the importance of preprocessing for meaningful analysis. For the purpose of this report, I have chosen to proceed with the lemmatization approach, as it retains context-aware word forms and provides more accurate representations of the original content.

## 4. Analysis Approach

##### 4.1 N-grams

In this report, I used N-gram language modeling with NLTK to explore the common themes in discussions around running and mental health. I generated both bigrams (2-word
combinations) and trigrams (3-word combinations) to capture the most frequent word patterns and phrases. This approach helped me find not just the important words, but also how people often use these words together in conversations (Fathima, 2024). It gave me a clearer picture of what people are really talking about when it comes to running and mental health.

##### 4.2 Sentiment Analysis

In this process, for word counting method, stopwords and punctuation are removed to clean the text, similar to the earlier data preprocessing steps. The text is then tokenized, and each word is checked against a list of known positive and negative words. However, Vader method is designed to work well on raw social media text, emojis, slang, and punctuation which often carry strong emotional cues, so cleaning, lowercasing, and lemmatizing are not applied in Vader method. There are two approaches of sentiment analysis performed in this report.

**4.2.1 Word Counting Method**

Word count-based analysis helps measure the overall emotion or tone in written content. It’s especially useful in survey analysis to understand how people feel about a particular topic (Displayr, 2025). This method assigns each text a sentiment score based on how many words match those in a predefined sentiment dictionary. The analysis provides four key scores – proportion of negative, neutral and positive sentiment in the text and the compound score that serves as the final indicator of sentiment, balancing all three proportions to reflect the general mood of the content.

**4.2.2 Vader Method**

Vader offers a more nuanced approach to measuring the emotional tone of text compared to traditional word-count methods. Instead of simply classifying words as positive or negative, Vader assesses the intensity of emotions expressed in the sentence. It does so by applying various rules to account for factors such as Intensifiers, such as "really", "very", which amplify the sentiment of surrounding words, Capitalization, which conveys a stronger emotional weight, and Punctuation, particularly exclamation marks (!), which further intensify the sentiment (Domaleski, 2024).

This report will perform sentiment analysis using two methods: word counting and Vader approach. The purpose is to compare the results from both methods to provide a clearer understanding of sentiment trends. However, more focus will be placed on the Vader analysis due to its higher efficiency and accuracy in processing text data.

##### 4.3 Topic Modelling

Topic modeling was performed using Latent Dirichlet Allocation (LDA), a generative probabilistic model. LDA assumes that each document is composed of a combination of various topics, and that each word in the document is generated from one of these underlying topics. The objective of LDA is to uncover the hidden topic structure by estimating the distribution of topics within documents and the distribution of words within those topics, based on the observed text data (Datta, 2024).

As a marketing researcher, I used LDA to explore the broader context in which people discuss running online. This approach allows me to identify hidden themes and conversations beyond surface- level keywords, offering valuable insights into consumer interests and behaviors. By analyzing how topics are distributed across time, LDA helps uncover emerging trends and shifts in public sentiment. These insights are instrumental in shaping targeted marketing campaigns and ensuring that promotional strategies align with evolving interests within the running community.

## 5. Analysis

##### 5.1 N-grams Analysis

The bigram analysis reveals a strong focus on marathon training, physical health, and mental well- being in Reddit conversations. To capture more context and emotional tone, a trigram analysis was also performed. While much of the discussion is positive and goal-oriented, trigrams like "feel like shit" and "mental health issue" show that users also express the struggles of training, including fatigue and emotional challenges. The presence of running-related terms and brands highlights a shared connection within the running community.

**The number of common bi-grams and tri-grams.**

![common_bitri](https://github.com/NadiaDu1999/Sentiment-Analysis-Running-and-MentalHealth/blob/main/Common%20bi(tri)grams.png)













