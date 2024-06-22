import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from art.attacks.evasion import FastGradientMethod, DeepFool, CarliniL2Method, ZooAttack
from art.estimators.classification import SklearnClassifier
from art.utils import to_categorical
import time
import pandas as pd
import warnings

# Ignore future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load MNIST data
mnist = fetch_openml('mnist_784')
X, y = mnist.data, mnist.target.astype(np.int)
X = X / 255.0
y = to_categorical(y, 10)

# Reduce dataset size for efficiency
X_train, X_test, y_train, y_test = train_test_split(X[:1000], y[:1000], test_size=0.2, random_state=42)

# Convert to NumPy arrays
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

# Define models
models = {
    "Logistic Regression": LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
                                              class_weight='balanced', random_state=None, solver='lbfgs', max_iter=10, 
                                              multi_class='ovr', verbose=0, warm_start=False, n_jobs=None),
    "SVM": SVC(C=1.0, kernel='linear', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=True, 
               tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, 
               decision_function_shape='ovr', random_state=None),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Train models
for name, model in models.items():
    model.fit(X_train, np.argmax(y_train, axis=1))

# Create ART classifiers
art_classifiers = {
    "Logistic Regression": SklearnClassifier(model=models["Logistic Regression"], clip_values=(0, 1)),
    "SVM": SklearnClassifier(model=models["SVM"]),
    "Random Forest": SklearnClassifier(model=models["Random Forest"]),
    "Gradient Boosting": SklearnClassifier(model=models["Gradient Boosting"])
}

# Define attacks
attacks = {
    "FGSM": FastGradientMethod,
    "DeepFool": DeepFool,
    "C&W": CarliniL2Method,
    "ZOO": ZooAttack
}

# Evaluate models under attacks
results = []

for model_name, classifier in art_classifiers.items():
    for attack_name, AttackClass in attacks.items():
        # Check if the attack is suitable for the model
        if model_name in ["Random Forest", "Gradient Boosting"] and attack_name in ["FGSM", "DeepFool", "C&W"]:
            continue
        
        start_time = time.time()
        
        # Generate adversarial examples
        if attack_name == "FGSM":
            attack = AttackClass(classifier, eps=0.5)
        elif attack_name == "DeepFool":
            attack = AttackClass(classifier, max_iter=30, verbose=True)
        elif attack_name == "C&W":
            attack = AttackClass(classifier, confidence=0.0, targeted=False, learning_rate=0.01, max_iter=10, binary_search_steps=5, initial_const=0.01, batch_size=1, verbose=False)
        elif attack_name == "ZOO":
            attack = AttackClass(classifier, confidence=0.0, targeted=False, learning_rate=0.01, max_iter=10, binary_search_steps=5, initial_const=0.01, batch_size=1, verbose=False)
            X_test_subset = X_test[:100]
            adv_examples = attack.generate(x=X_test_subset)
            y_test_subset = y_test[:100]
        else:
            attack = AttackClass(estimator=classifier, eps=0.5)
            adv_examples = attack.generate(x=X_test)
        
        if attack_name != "ZOO":
            adv_examples = attack.generate(x=X_test)

        end_time = time.time()
        
        # Evaluate adversarial examples
        if attack_name == "ZOO":
            y_pred_adv = np.argmax(classifier.predict(adv_examples), axis=1)
            y_true_subset = np.argmax(y_test_subset, axis=1)
            
            accuracy_adv = accuracy_score(y_true_subset, y_pred_adv)
            f1_adv = f1_score(y_true_subset, y_pred_adv, average='macro')
            recall_adv = recall_score(y_true_subset, y_pred_adv, average='macro')
            success_rate = np.mean(y_true_subset != y_pred_adv)
        else:
            y_pred_adv = np.argmax(classifier.predict(adv_examples), axis=1)
            y_true = np.argmax(y_test, axis=1)
            
            accuracy_adv = accuracy_score(y_true, y_pred_adv)
            f1_adv = f1_score(y_true, y_pred_adv, average='macro')
            recall_adv = recall_score(y_true, y_pred_adv, average='macro')
            success_rate = np.mean(y_true != y_pred_adv)
        
        y_pred = np.argmax(classifier.predict(X_test), axis=1)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        
        execution_time = end_time - start_time
        
        results.append({
            "Model": model_name,
            "Attack": attack_name,
            "Accuracy": accuracy,
            "Accuracy After Attack": accuracy_adv,
            "F1 Score": f1,
            "F1 Score After Attack": f1_adv,
            "Recall": recall,
            "Recall After Attack": recall_adv,
            "Success Rate": success_rate,
            "Execution Time (s)": execution_time
        })

# Display results
results_df = pd.DataFrame(results)
print(results_df)

# Plotting results
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Accuracy drop plot
for model_name in models.keys():
    subset = results_df[results_df["Model"] == model_name]
    axes[0].plot(subset["Attack"], subset["Accuracy After Attack"], label=model_name)

axes[0].set_title('Accuracy Drop After Attack')
axes[0].set_xlabel('Attack Method')
axes[0].set_ylabel('Accuracy After Attack')
axes[0].legend()

# Execution time plot
execution_times = results_df.pivot(index='Attack', columns='Model', values='Execution Time (s)')
execution_times.plot(kind='bar', ax=axes[1])
axes[1].set_title('Execution Time of Adversarial Attacks')
axes[1].set_xlabel('Attack Method')
axes[1].set_ylabel('Time (s)')

plt.tight_layout()
plt.show()
plt.savefig("attack_performanc")