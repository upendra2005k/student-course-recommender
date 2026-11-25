import streamlit as st
import pandas as pd
import joblib
import subprocess
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="MSR Course Recommender", layout="wide")

@st.cache_resource
def load_bundle():
    return joblib.load("models/recommender_bundle.pkl")

bundle = load_bundle()
model = bundle["model"]
ohe = bundle["ohe"]
mlb = bundle["mlb"]
tfidf = bundle["tfidf"]
course_tfidf_matrix = bundle["course_tfidf_matrix"]
courses_df = bundle["courses_df"]

students_ref = pd.read_csv("data/student_data.csv")
branch_options = sorted(students_ref["branch"].dropna().unique().tolist())
interest_options = sorted(students_ref["interest_area"].dropna().unique().tolist())
skill_options = sorted(list(mlb.classes_))

st.sidebar.title("MSR Edusoft Portal")
page = st.sidebar.radio("Navigation", ["Student Course Recommender", "Coordinator Dashboard", "Data Upload & Retrain"])

if page == "Student Course Recommender":
    st.title("Student Course Recommendation System")
    st.write("Enter your branch, year, skills, and interest area to get your top 3 recommended courses.")

    with st.form("student_form"):
        col1, col2 = st.columns(2)
        with col1:
            branch = st.selectbox("Branch", branch_options)
            year = st.selectbox("Year of Study", [1, 2, 3, 4])
        with col2:
            interest = st.selectbox("Area of Interest", interest_options)
            skills = st.multiselect("Current Skills", options=skill_options, default=skill_options[:2])
        submitted = st.form_submit_button("Get Top 3 Recommendations")

    if submitted:
        input_df = pd.DataFrame([{"branch": branch, "interest_area": interest}])
        cat_vec = ohe.transform(input_df)
        known_skills = [s for s in skills if s in mlb.classes_]
        skill_vec = csr_matrix(mlb.transform([known_skills]))
        X_student = hstack([cat_vec, skill_vec])

        ml_probs = model.predict_proba(X_student)[0]
        ml_score_map = dict(zip(model.classes_, ml_probs))

        student_text = f"{interest} {' '.join(skills)}"
        student_tfidf = tfidf.transform([student_text])
        cos_scores = cosine_similarity(student_tfidf, course_tfidf_matrix)[0]

        results = []
        for idx, row in courses_df.iterrows():
            cname = row["course_name"]
            p_ml = ml_score_map.get(cname, 0.0)
            s_cos = cos_scores[idx]
            hybrid_score = (0.6 * p_ml) + (0.4 * s_cos)

            matched_skills = [s for s in skills if s.lower() in row["description"].lower()]
            reason_parts = []
            if matched_skills:
                reason_parts.append(f"matches your **{', '.join(matched_skills)}** skill(s)")
            if interest.lower() in row["domain"].lower():
                reason_parts.append(f"aligns with your **{interest}** interest")
            reason = "Recommended because it " + " and ".join(reason_parts) + "." if reason_parts else f"Strong fit for **{branch}** students."

            results.append({
                "course_name": cname,
                "domain": row["domain"],
                "level": row["level"],
                "duration": row["duration_weeks"],
                "match_pct": min(99, max(45, int(round((hybrid_score * 100) + 30)))),
                "raw_score": hybrid_score,
                "reason": reason
            })

        top3 = sorted(results, key=lambda x: x["raw_score"], reverse=True)[:3]

        st.subheader("Top 3 Recommended Courses for You")
        cols = st.columns(3)
        for i, rec in enumerate(top3):
            with cols[i]:
                st.metric(label=f"#{i+1} {rec['course_name']}", value=f"{rec['match_pct']}% Match")
                st.caption(f"Domain: {rec['domain']} | Level: {rec['level']} | {rec['duration']} Weeks")
                st.info(rec["reason"])

elif page == "Coordinator Dashboard":
    st.title("Coordinator Analytics Dashboard")
    df = pd.read_csv("data/student_data.csv")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Students", len(df))
    c2.metric("Total Courses Offered", len(courses_df))
    c3.metric("Course Completion Rate", f"{(df['completed'] == 'Yes').mean()*100:.1f}%")

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Most Popular Courses")
        st.bar_chart(df["course_taken"].value_counts().head(10))
    with col_right:
        st.subheader("Students by Branch")
        st.bar_chart(df["branch"].value_counts())

elif page == "Data Upload & Retrain":
    st.title("Upload New Student Data")
    uploaded_file = st.file_uploader("Upload CSV (columns: student_id, branch, year, skills, interest_area, course_taken, completed)", type=["csv"])

    if uploaded_file is not None:
        new_df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:", new_df.head())
        if st.button("Append Data & Retrain Model"):
            existing_df = pd.read_csv("data/student_data.csv")
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined.to_csv("data/student_data.csv", index=False)
            subprocess.run(["python", "train_model.py"], check=True)
            st.cache_resource.clear()
            st.success("Data appended and ML model retrained successfully!")