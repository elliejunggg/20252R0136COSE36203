# Build the Bernoulli Naive Bayes model
bnb = BernoulliNB()

# Train the model
bnb.fit(X_train, y_train)

# Predict on test set
y_pred_bnb = bnb.predict(X_test)

# Compute accuracy score
acc_bnb = accuracy_score(y_test, y_pred_bnb)

# Print evaluation results
print("=== Bernoulli Naive Bayes Evaluation Results ===")
print("Accuracy:", acc_bnb)
print(classification_report(y_test, y_pred_bnb))

# Plot confusion matrix as heatmap
cm = confusion_matrix(y_test, y_pred_bnb)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Bernoulli Naive Bayes Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
