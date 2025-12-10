# Add columns from vader dataframe to the final dataframe
df_final = pd.concat([df_final, df_vader[['vader_score']]], axis=1)
df_final

# Before applying any models, text data (preprocessed_review) should be converted as numerical features
# Apply text vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df_final['processed_final'])
y = df_final['Recommended']

# Divide the dataset using train_test_split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("-------Shape of train-test data---------")
print(f'Train feature shape: {X_train.shape}')
print(f'Test feature shape: {X_test.shape}')

# Divide the dataset using train_test_split (80% train, 20% test) for VADER
X_vader = df_final[['vader_score']]
y_vader = df_final['Recommended']

X_train_vader, X_test_vader, y_train_vader, y_test_vader = train_test_split(X_vader, y_vader, test_size=0.2, random_state=42)

print("-------Shape of train-test VADER data---------")
print(f'Train feature shape: {X_train_vader.shape}')
print(f'Test feature shape: {X_test_vader.shape}')