"""# **Class Comparison**"""

# Build Logistic Regression model
logreg_final = LogisticRegression(random_state = 42)

# Train the model
logreg_final.fit(X_train, y_train)

# Predict on the whole dataset
X_all = vectorizer.transform(df_final['Reviews'])
df_final['pos_rate'] = logreg_final.predict(X_all)

# Predict probability on test set
df_final['pos_prob'] = logreg_final.predict_proba(X_all)[:, 1]

# Create new dataframe
df_class = (
    df_final.groupby(['Airline', 'Class'])
    .agg(
        positive_rate = ('pos_rate', 'mean'),
        positive_probability = ('pos_prob', 'mean'),
        )
    .reset_index()
    .sort_values(['Airline','Class'])
)

df_class

# Combine positive_rate score and positive_probability score
df_class['positive_score'] = df_class['positive_rate'] * df_class['positive_probability']

# Print the dataframe for each class
for cls in df_class['Class'].unique():
    print(f'\n===== Top Airlines Ranking for {cls}=====\n')

    df_rank = df_class[df_class['Class'] == cls].sort_values('positive_score', ascending=False).reset_index(drop=True)
    df_rank['Rank'] = df_rank.index + 1
    df_rank = df_rank.set_index('Rank')
    display(df_rank[['Airline', 'positive_score']])

# Display top 5 Airlines for each class
for cls in df_class['Class'].unique():
  top5 = df_class[df_class['Class'] == cls].sort_values('positive_score', ascending=False).head(5)

  plt.figure(figsize=(5, 5))
  plt.bar(top5['Airline'], top5['positive_score'])
  plt.title(f'Top 5 Airlines for {cls}')
  plt.xlabel('Airline')
  plt.ylabel('Positive Score')
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.show()

"""#**Compare the Details**"""

# Only extract columns with numeric value
num_cols = df_final.select_dtypes(include=['int64']).columns.tolist()

# Drop Recommended and pos_rate column (unneeded)
remove_cols = ['Recommended', 'pos_rate']
num_cols = [col for col in num_cols if col not in remove_cols]

ranking_per_column = {}

for col in num_cols:
  df_avg = df_final.groupby('Airline')[col].mean().sort_values(ascending=False).reset_index()

  df_avg['Rank'] = df_avg.index + 1
  df_avg = df_avg.set_index('Rank')
  ranking_per_column[col] = df_avg

for col, table in ranking_per_column.items():
  print(f'\n===== Top Airlines Ranking for {col} =====\n')
  display(table)

# Display top 5 Airlines for each characteristics
for col, table in ranking_per_column.items():
  top5 = table.head(5)

  plt.figure(figsize=(5, 5))
  plt.bar(top5['Airline'], top5[col])
  plt.title(f'Top 5 Airlines for {col}')
  plt.xlabel('Airline')
  plt.ylabel(col)
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.show()