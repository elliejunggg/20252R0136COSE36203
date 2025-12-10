# Feature scaling: ensures all the features are on a similar scale
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rfc.fit(X_train_scaled, y_train)

# Predict on test set
y_pred_rfc = rfc.predict(X_test_scaled)

# Compute accuracy score
acc_rfc = accuracy_score(y_test, y_pred_rfc)

# Print evaluation results
print("=== Random Forest Classifier Evaluation Results ===")
print("Accuracy:", acc_rfc)
print(classification_report(y_test, y_pred_rfc))

# Plot confusion matrix as heatmap
cm = confusion_matrix(y_test, y_pred_rfc)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Random Forest Classifier Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()