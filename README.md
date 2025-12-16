# 🏃🏻‍♀️‍➡️💪🏽 Reddit Data Mining for Running and Mental Health Insights

![cover](https://github.com/NadiaDu1999/Sentiment-Analysis-Running-and-MentalHealth/blob/main/cover.png)

## ✨ 1. Introduction

In the era of social media, we often find ourselves consumed by the lives of others, constantly comparing their highlights with our reality. This habit can take a toll on our mental health. However, like every coin has two sides, social media can also be a source of inspiration. For me personally, it became the motivation to start exercising and take better care of my body. I began running as a way to improve my mental health, and I can confidently say that running has truly changed my life.

In this paper, I imagine myself as part of the marketing team for a non-profit running organization, working to promote a campaign that inspires more people to run. One of the key objectives of this project is to understand how running affects people’s mental well-being and what motivates them to start or continue running. To explore this, I analyse Reddit posts that discuss running and mental health. I also examine the broader context in which people talk about running on Reddit. Additionally, I look at the sentiments expressed, both positive and negative, as a way to understand strong opinions and emotional connections people have with running.

## 💻 2. Data Collection

The data was collected from Reddit using its official API through PRAW (Python Reddit API Wrapper), a Python library that provides easy access to Reddit’s data. To focus on discussions specifically related to running and mental health, I selected seven relevant subreddits: r/running, r/BeginnersRunning,r/
beginnerrunning, r/Marathon_Training, r/AdvancedRunning, r/XXRunning, and r/firstmarathon. I used the some keywords, such as "mental", “therapy” and “burnout”, to narrow down the scope of the search, avoiding unrelated running topics and ensuring relevance to mental health. The plan was to retrieve up to 500 posts from each subreddit, totalling 3,500 posts. However, only 925 posts were successfully saved into the final JSON file. This discrepancy is likely due to the fact that not all subreddits had 500 posts that matched the keyword criteria within the available Reddit archive, especially for smaller or less active subreddits. Additionally, PRAW returns only a subset of all possible results due to Reddit’s search constraints and API limits, which can also contribute to the lower number of retrieved posts (LinkedIn, 2024).

## 3. 📖 Data Pre-Processing and Exploration 

**3.1 Conversion to lower case**

To maintain uniformity and reduce case-related variability in the text data, all text was converted to lowercase (Gorale, 2024). This normalization step ensures that words like "Running" and "running" are treated identically during analysis.

**3.2 Noise reduction**
To reduce textual noise, punctuation marks and special characters such as . , ; : ? ! " ' - _ ( ) [ ] {
} were removed. This was done using Python’s string.punctuation, which provides a convenient list of common punctuation symbols to filter out irrelevant characters from the data.

**3.3 Tokenization**

Tokenization is the process of splitting text into individual elements or tokens (usually words). For instance, the sentence "I love running" becomes ["I", "love", "running"]. Instead of the
standard word_tokenize() function, I used NLTK’s TweetTokenizer, which is better suited for informal or social media text. It handles contractions like "I'm" correctly by tokenizing it as ["I", "'m"], preserving meaning in casual language formats. However, if we skip the noise reduction process—which includes steps like removing punctuation and special characters—tokens such as commas and quotation marks (e.g., ",", "“", etc.) will still appear in the list, leading to less meaningful analysis. The example below illustrates this difference clearly.

**3.4 Stop word removal**

Commonly used words such as "the", "is", and "and" contribute little to semantic analysis and were removed to reduce noise. The stopword list was based on NLTK’s default set, with additional domain- specific terms like "rt", "via", "...", "’", "http", and "https" also excluded. These additions are particularly useful for cleaning Reddit posts, where such tokens are frequent but not meaningful.

**3.5 Normalization**

To further refine the text, lemmatization was applied to reduce words to their base or dictionary forms. Unlike stemming, which may produce non-standard or partial roots, lemmatization ensures that the output is meaningful and context-aware. For example:

Original Words: running, runner, runs After Stemming: run, run, run

After Lemmatization: running, runner, run

This also explains why the word “run” appeared more frequently in the stemming results
(over 8,000 times) compared to lemmatization (just over 5,000 times). In stemming, different word forms like “runner” or “running” are all reduced to the root “run”, regardless of context. In contrast, lemmatization retains more accurate, context-aware word forms, which leads to a more balanced distribution of related terms.

![Bi(tri)grams](https://github.com/NadiaDu1999/Sentiment-Analysis-Running-and-MentalHealth/blob/main/Bi(Tri)%20grams.png)
Figure 1: Comparison of Top Terms: Stemming vs Lemmatization

This distinction highlights the benefit of lemmatization: while stemming treats all forms as identical, lemmatization preserves important differences—such as between run and runner, which have distinct meanings (Piduguralla, 2023). To evaluate the effectiveness of both approaches, a comparative analysis was performed and summarized in Table 2. The results demonstrate that lemmatization produces more accurate, meaningful, and human-readable outputs, making it more suitable for sentiment analysis in this context.

Table 1: The output from normalization by Stemming and Lemmatizer
![lemma_stemming](https://github.com/NadiaDu1999/Sentiment-Analysis-Running-and-MentalHealth/blob/main/Lemma%20vs.%20Stemming.png)

As expected, before applying any cleaning or tokenization, the total raw word count was 499,669. After the cleaning and tokenization process, the word count was significantly reduced — down
to 243,693 using stemming and 248,501 using lemmatization. This substantial reduction highlights how much unnecessary or redundant content (such as punctuation, stopwords, or repeated word forms) was present in the original text, and demonstrates the importance of preprocessing for meaningful analysis. For the purpose of this report, I have chosen to proceed with the lemmatization approach, as it retains context-aware word forms and provides more accurate representations of the original content.

## 🔎 4. Analysis Approach

**4.1 N-grams**

In this report, I used N-gram language modeling with NLTK to explore the common themes in discussions around running and mental health. I generated both bigrams (2-word
combinations) and trigrams (3-word combinations) to capture the most frequent word patterns and phrases. This approach helped me find not just the important words, but also how people often use these words together in conversations (Fathima, 2024). It gave me a clearer picture of what people are really talking about when it comes to running and mental health.

**4.2 Sentiment Analysis**

In this process, for word counting method, stopwords and punctuation are removed to clean the text, similar to the earlier data preprocessing steps. The text is then tokenized, and each word is checked against a list of known positive and negative words. However, Vader method is designed to work well on raw social media text, emojis, slang, and punctuation which often carry strong emotional cues, so cleaning, lowercasing, and lemmatizing are not applied in Vader method. There are two approaches of sentiment analysis performed in this report.

##### 4.2.1 Word Counting Method

Word count-based analysis helps measure the overall emotion or tone in written content. It’s especially useful in survey analysis to understand how people feel about a particular topic (Displayr, 2025). This method assigns each text a sentiment score based on how many words match those in a predefined sentiment dictionary. The analysis provides four key scores – proportion of negative, neutral and positive sentiment in the text and the compound score that serves as the final indicator of sentiment, balancing all three proportions to reflect the general mood of the content.

##### 4.2.2 Vader Method

Vader offers a more nuanced approach to measuring the emotional tone of text compared to traditional word-count methods. Instead of simply classifying words as positive or negative, Vader assesses the intensity of emotions expressed in the sentence. It does so by applying various rules to account for factors such as Intensifiers, such as "really", "very", which amplify the sentiment of surrounding words, Capitalization, which conveys a stronger emotional weight, and Punctuation, particularly exclamation marks (!), which further intensify the sentiment (Domaleski, 2024).

This report will perform sentiment analysis using two methods: word counting and Vader approach. The purpose is to compare the results from both methods to provide a clearer understanding of sentiment trends. However, more focus will be placed on the Vader analysis due to its higher efficiency and accuracy in processing text data.

**4.3 Topic Modelling**

Topic modeling was performed using Latent Dirichlet Allocation (LDA), a generative probabilistic model. LDA assumes that each document is composed of a combination of various topics, and that each word in the document is generated from one of these underlying topics. The objective of LDA is to uncover the hidden topic structure by estimating the distribution of topics within documents and the distribution of words within those topics, based on the observed text data (Datta, 2024).

As a marketing researcher, I used LDA to explore the broader context in which people discuss running online. This approach allows me to identify hidden themes and conversations beyond surface- level keywords, offering valuable insights into consumer interests and behaviors. By analyzing how topics are distributed across time, LDA helps uncover emerging trends and shifts in public sentiment. These insights are instrumental in shaping targeted marketing campaigns and ensuring that promotional strategies align with evolving interests within the running community.

## 📌 5. Analysis

**5.1 N-grams Analysis**

The bigram analysis reveals a strong focus on marathon training, physical health, and mental well- being in Reddit conversations. To capture more context and emotional tone, a trigram analysis was also performed. While much of the discussion is positive and goal-oriented, trigrams like "feel like shit" and "mental health issue" show that users also express the struggles of training, including fatigue and emotional challenges. The presence of running-related terms and brands highlights a shared connection within the running community.

Table 2: The number of common bi-grams and tri-grams.
![common_bitri](https://github.com/NadiaDu1999/Sentiment-Analysis-Running-and-MentalHealth/blob/main/Common%20bi(tri)grams.png)

**5.2 Sentiment Analysis**

##### 5.2.1 Word Counting Sentiment Analysis

From the word count analysis, we identified 22,560 positive words, 14,947 negative words, and 210,884 neutral words. Figure 2 presents a word cloud showcasing the most common positive, negative, and neutral terms that appeared in the comments. However, the words captured in this analysis are quite generic, making it challenging to fully understand the underlying context or sentiment of the discussions. This highlights the limitation of the word count approach in capturing the nuances of language, especially in a diverse and informal setting like Reddit comments.

![pos_neg_neu](https://github.com/NadiaDu1999/Sentiment-Analysis-Running-and-MentalHealth/blob/main/pos%3Aneg%3Aneu.png)
Figure 2: Word Cloud represents the common positive, negative and neutral words

Moreover, another significant limitation of the word count approach is its inability to capture the context, negation, sarcasm, and emotional intensity behind the words. This often leads to misclassification of the overall tone of a message. For example, both "GREAT" and "okay" might be counted as positive words with the same weight (+1), despite "GREAT" expressing a much stronger sentiment. The method fails to differentiate between varying intensities or emotional strengths, which can result in an inaccurate assessment of the tone In this case, the sentiment of the comment might be calculated as 0 (positive words minus negative words), yet, in reality, the comment carries a distinctly positive tone.

##### 5.2.2 Vader Sentiment Analysis

The overall sentiment score, based on the Vader analysis, is 0.4, indicating a predominantly positive sentiment across the dataset. Below are the top 3 positive sentiments derived from the analysis. These excerpts primarily consist of comments from Reddit users, and the sentiments reflect encouragement and motivation for other runners.

![vader1](https://github.com/NadiaDu1999/Sentiment-Analysis-Running-and-MentalHealth/blob/main/Vader1.png)

The most negative sentiments, as identified by Vader compound scores, reflect deeply personal and emotionally charged experiences shared by runners. These include coping with disappointing race outcomes, enduring physical pain, recovering from unexpected injuries unrelated to running, and dealing with traumatic incidents such as dog attacks on trails. Interestingly, amidst these challenges, some individuals also shared how their setbacks led them to explore new interests—such as cycling— highlighting a shift in focus and resilience in adapting to changing circumstances.

![vader2](https://github.com/NadiaDu1999/Sentiment-Analysis-Running-and-MentalHealth/blob/main/Vader2.png)

In figure 3A, the word count method shows extreme positive sentiment spikes, while Vader remains more stable. Early data (pre-2014) is missing in the monthly trend, likely due to limited Reddit activity or topic relevance. However, figure 3B shows overall positive yearly sentiment from 2014 to 2016. A dip into negative sentiment appears only in the word count method around 2020–2022, suggesting it may be more sensitive to changes in language tone.

![sentiment](https://github.com/NadiaDu1999/Sentiment-Analysis-Running-and-MentalHealth/blob/main/Sentiment.png)
Figure 3: Monthly Sentiment Trends (Figure 3A) and Yearly Sentiment Trend (Figure 3B)

**5.3 Topic Summary**

During the initial phase of topic modeling, I overlooked the importance of removing website links from the dataset. As a result, the generated word clouds prominently featured irrelevant tokens like "http" and "com," which distorted the topic interpretations. I later realized that cleaning the data to remove common URL components such as "http," "www," and "com" was essential, as many comments included links to external websites. Once the data was properly cleaned, I proceeded to determine the optimal number of topics for the LDA model. My first attempt with 3 topics yielded vague and uninformative results, such as a topic dominated by generic words like “feel” and “like”, which lacked actionable insights. Therefore, I manually experimented with different numbers of topics and assessed the coherence of the outputs. Eventually, I settled on using 9 topics, displaying the top 15 words per topic, and limiting the vocabulary size to 2000 features for clearer interpretations.

Table 4: The summarized conclusions for each topic.
![topic_summary](https://github.com/NadiaDu1999/Sentiment-Analysis-Running-and-MentalHealth/blob/main/Topic_summary.png)

![wordCloud](https://github.com/NadiaDu1999/Sentiment-Analysis-Running-and-MentalHealth/blob/main/TopicModelling.png)
Figure 4: Word Cloud represents topic modelling

## 📝 6. Conclusion

This report explored how running relates to mental well-being and what motivates people to run. The findings from sentiment analysis indicate that, overall, people express a positive sentiment toward running. However, there are certain months where negative sentiments are more prominent. A closer look into the top posts during those periods reveals that negative experiences, such as traumatic events like dog attacks on trails, or a shift in interest toward other activities like cycling may contribute to these sentiments. Topic modelling further supports the richness of online discussions surrounding running. A variety of themes emerged, including marathons, running clubs, music, and deeply personal topics such as parenting, pregnancy, postpartum experiences, and even health challenges like cancer. These discussions suggest that running is often intertwined with significant life events and transformations. N-gram analysis also showed that Reddit is a space where people share their emotions, encourage others, and reflect on their running journeys. This community-driven engagement highlights how running is not only a physical activity but also a powerful source of emotional connection and support.


## 🗞️ 7. References

[1] C. Gorale, "Applying Lowercase Before and After Tokenization in NLP", Medium.com, 2024. [Online]. Available: https://cgorale111.medium.com/applying-lowercase-before-and-after- tokenization-in-nlp- 67f50462b06f#:~:text=Consistency%3A%20Applying%20lowercase%20conversion%20before,maki ng%20the%20tokens%20more%20uniform. [Accessed: 10- April- 2025].

[2] Displayr, “How to Count the Number of Positive and Negative Terms Use to Calculate Sentiment Scores,” 2025. [Online]. Available: https://help.displayr.com/hc/en-us/articles/7163382749967-How- to-Count-the-Number-of-Positive-and-Negative-Terms-Use-to-Calculate-Sentiment- Scores#:~:text=Sentiment%20analysis%20is%20a%20way,a%20negative%20to%20positive%20scal e). [Accessed 15- April- 2025].

[3] J. Chan, " COSC2671 | Social Media and Network Analytics, Lab Notes (Internal) [Online] [Private]. [Accessed: 15- Aug- 2019].

[4] J. Domaleski, "Basic Sentiment Analysis Using R with VADER", Medium.com, 2024. [Online]. Available: https://blog.marketingdatascience.ai/basic-sentiment-analysis-using-r-with-vader- 4eecb738566f. [Accessed: 16- April- 2025].

[5] Linkedin, “What are the most effective ways to collect data from Reddit?”, 2024. [Online]. Available: https://www.linkedin.com/advice/1/what-most-effective-ways-collect-data-from-reddit- ybgtf. [Accessed 15- April- 2025].

[6] P. Datta, "Analyzing Text Data with Topic Modeling: Latent Dirichlet Allocation (LDA) Explained", Medium.com, 2024. [Online]. Available: https://medium.com/@pinakdatta/understanding-lda-unveiling-hidden-topics-in-text-data- 9bbbd25ae162. [Accessed: 16- April- 2025].

[7] S. Fathima, "Explore NGram Analysis to Understand Language Patterns", MarkovML.com, 2024. [Online]. Available: https://www.markovml.com/blog/ngram- analysis#:~:text=N%2Dgram%20in%20Natural%20Language,comes%20next%20in%20a%20sentenc e. [Accessed: 15- April- 2025].

[8] S. Piduguralla, “Understanding the Difference Between Stemming and Lemmatization”, Medium.com, 2023. [Online]. Available: https://medium.com/@tejaswaroop2310/understanding-the- difference-between-stemming-and-lemmatization-dbfdfed98df0. [Accessed: 10- April- 2025].














