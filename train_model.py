import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

students = pd.read_csv("data/student_data.csv").drop_duplicates().dropna()
courses = pd.read_csv("data/courses_master.csv")

students["branch"] = students["branch"].str.strip().str.upper()
students["interest_area"] = students["interest_area"].str.strip()
students["skills_list"] = students["skills"].apply(
    lambda x: [s.strip() for s in str(x).split(",") if s.strip()]
)

ohe = OneHotEncoder(handle_unknown="ignore")
cat_features = ohe.fit_transform(students[["branch", "interest_area"]])

mlb = MultiLabelBinarizer()
skill_features = csr_matrix(mlb.fit_transform(students["skills_list"]))

X = hstack([cat_features, skill_features])
y = students["course_taken"]

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, students.index, test_size=0.2, random_state=42, stratify=y
)

# Baseline Model
train_df = students.loc[idx_train]
test_df = students.loc[idx_test]
popular_by_branch = train_df.groupby("branch")["course_taken"].agg(lambda s: s.value_counts().index[0]).to_dict()
global_popular = train_df["course_taken"].value_counts().index[0]
baseline_preds = [popular_by_branch.get(b, global_popular) for b in test_df["branch"]]

print("\n--- MODEL COMPARISON (Copy into Table 5.2 of your report) ---")
print(f"Baseline -> Accuracy: {accuracy_score(y_test, baseline_preds):.2f} | "
      f"F1-Score: {f1_score(y_test, baseline_preds, average='weighted', zero_division=0):.2f}")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
}

best_model = None
best_acc = -1

for name, clf in models.items():
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    acc = accuracy_score(y_test, preds)
    top3 = top_k_accuracy_score(y_test, probs, k=3, labels=clf.classes_)
    f1 = f1_score(y_test, preds, average="weighted")
    print(f"{name} -> Accuracy: {acc:.2f} | Top-3 Accuracy: {top3:.2f} | F1-Score: {f1:.2f}")
    if acc >= best_acc:
        best_acc = acc
        best_model = clf

courses["combined_text"] = courses["course_name"] + " " + courses["domain"] + " " + courses["description"]
tfidf = TfidfVectorizer(stop_words="english")
course_tfidf_matrix = tfidf.fit_transform(courses["combined_text"])

bundle = {
    "model": best_model,
    "ohe": ohe,
    "mlb": mlb,
    "tfidf": tfidf,
    "course_tfidf_matrix": course_tfidf_matrix,
    "courses_df": courses
}
joblib.dump(bundle, "models/recommender_bundle.pkl")
print("\nSaved trained model bundle to models/recommender_bundle.pkl")