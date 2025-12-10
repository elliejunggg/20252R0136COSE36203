# Build the Linear Support Vector Machine model
lsvm = svm.LinearSVC()

# Train the model
lsvm.fit(X_train, y_train)

# Predict on test set
y_pred_lsvm = lsvm.predict(X_test)

# Compute accuracy score
acc_lsvm = accuracy_score(y_test, y_pred_lsvm)

# Print evaluation results
print("=== Linear Support Vector Machine Evaluation Results ===")
print("Accuracy:", acc_lsvm)
print(classification_report(y_test, y_pred_lsvm))

# Plot confusion matrix as heatmap
cm = confusion_matrix(y_test, y_pred_lsvm)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Linear Support Vector Machine Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
