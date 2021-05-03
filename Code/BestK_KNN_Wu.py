from sklearn.metrics import accuracy_score
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train_std, y_train)
y_pred = clf.predict(X_test_std)
y_pred_proba = clf.predict_proba(X_test_std)
conf_matrix = confusion_matrix(y_test, y_pred)

accuracy = []
for i in range(1, 50):
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(X_train_std, y_train)
    y_pred = clf.predict(X_test_std)
    accuracy.append(accuracy_score(y_test, y_pred))

highest_K_accuracy = max(accuracy)

best_K = accuracy.index(max(accuracy)) + 1

# plot the K loop
plt.figure(figsize=(12,8))
plt.plot(np.arange(1,50), accuracy, color = 'navy', ls='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('')
plt.xlabel('K value')
plt.ylabel('Accuracy')
