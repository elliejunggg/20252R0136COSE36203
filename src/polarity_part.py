# Drop unnecessary columns and not verified data
df_final = df_preprocessed.drop(['Title', 'Name', 'Review Date', 'Type of Traveller', 'Month Flown', 'Route'], axis=1).query("Verified == 'True'")
df_final = df_final.assign(Recommended=lambda x: x['Recommended'].map({'yes': 1, 'no': 0}))
df_final = df_final.reset_index(drop=True)

# Since we only kept the verified data(rows), drop the column as we don't need it anymore
df_final = df_final.drop(['Verified'], axis=1)
df_final

# Drop unnecessary columns and not verified data
df_final = df_preprocessed.drop(['Title', 'Name', 'Review Date', 'Type of Traveller', 'Month Flown', 'Route'], axis=1).query("Verified == 'True'")
df_final = df_final.assign(Recommended=lambda x: x['Recommended'].map({'yes': 1, 'no': 0}))
df_final = df_final.reset_index(drop=True)

# Since we only kept the verified data(rows), drop the column as we don't need it anymore
df_final = df_final.drop(['Verified'], axis=1)
df_final

# Use vader to evaluated sentiment of reviews
def evalSentences(sentences, to_df=False, columns=[]):
    # Instantiate an instance to access SentimentIntensityAnalyzer class
    # from vader in nltk
    sid = SentimentIntensityAnalyzer()
    pdlist = []
    if to_df:
        for sentence in tqdm(sentences):
            ss = sid.polarity_scores(sentence)
            pdlist.append([sentence]+[ss['compound']])
        reviewDf = pd.DataFrame(pdlist)
        reviewDf.columns = columns
        return reviewDf

    else:
        for sentence in tqdm(sentences):
            print("\n" + sentence)
            ss = sid.polarity_scores(sentence)
            for k in sorted(ss):
                print('{0}: {1}, '.format(k, ss[k]), end='')
            print()
            
            
reviews = df_final['Reviews'].values

# Dataframe with airlines, reviews and vader scores
df_vader = evalSentences(df_final['Reviews'].values, to_df=True, columns=['review','vader_score'])
df_vader = pd.concat([df_final['Airline'], df_vader], axis=1)
df_vader