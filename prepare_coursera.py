import pandas as pd
import numpy as np
import re
import random

# 1. Load both Kaggle datasets
coursera = pd.read_csv("data/Coursera.csv").dropna(subset=["Course Name", "Course Description", "Skills"]).drop_duplicates(subset=["Course Name"])
skills_raw = pd.read_csv("data/skills_dataset.csv").dropna(subset=["skill_name", "related_careers"])

# Clean punctuation noise (e.g. '###', ') html') from skills_dataset.csv
skills_raw["clean_skill"] = skills_raw["skill_name"].apply(
    lambda s: re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9+#.]+$", "", str(s)).strip()
)
hard_skills = skills_raw[
    (skills_raw["category"] == "Hard Skill") &
    (skills_raw["clean_skill"].str.len().between(2, 22)) &
    (~skills_raw["clean_skill"].str.contains(","))
].copy()

# 2. Filter Coursera to 25 core engineering/tech courses
keywords = "Python|Machine Learning|Data Science|Web|Cloud|Cybersecurity|SQL|Java|Deep Learning|Linux|Network|IoT|AI"
tech_courses = coursera[coursera["Course Name"].str.contains(keywords, case=False, na=False)].head(25).reset_index(drop=True)

def map_domain(title):
    t = title.lower()
    if any(k in t for k in ["machine learning", "deep learning", "ai", "neural"]):
        return "AI/ML"
    if any(k in t for k in ["data", "sql", "analytics"]):
        return "Data Science"
    if any(k in t for k in ["web", "html", "javascript", "full stack", "react"]):
        return "Web Development"
    if any(k in t for k in ["cloud", "aws", "devops", "docker"]):
        return "Cloud Computing"
    if any(k in t for k in ["security", "cyber", "network", "linux"]):
        return "Cybersecurity"
    return "Python & Software"

courses_master = pd.DataFrame({
    "course_id": [f"C{i+1:03d}" for i in range(len(tech_courses))],
    "course_name": tech_courses["Course Name"].str.strip(),
    "domain": tech_courses["Course Name"].apply(map_domain),
    "level": tech_courses["Difficulty Level"].replace("Not Calibrated", "Intermediate").fillna("Intermediate"),
    "duration_weeks": np.random.choice([4, 6, 8], size=len(tech_courses)),
    "description": (tech_courses["Course Description"] + " " + tech_courses["Skills"]).str.strip()
})
courses_master.to_csv("data/courses_master.csv", index=False)

# 3. Build student_data.csv combining Coursera tags + Kaggle Hard Skills
domain_branch_map = {
    "AI/ML": ["CSE", "IT", "ECE"],
    "Data Science": ["CSE", "IT"],
    "Web Development": ["CSE", "IT"],
    "Cloud Computing": ["CSE", "IT", "ECE"],
    "Cybersecurity": ["CSE", "IT", "ECE"],
    "Python & Software": ["CSE", "ECE", "EEE", "IT"]
}

domain_career_keywords = {
    "AI/ML": "Machine Learning|Data Scientist|AI",
    "Data Science": "Data Analyst|Data Engineer|Business Intelligence",
    "Web Development": "Web|Full Stack|Frontend|Backend",
    "Cloud Computing": "Cloud|DevOps|Infrastructure",
    "Cybersecurity": "Security|Cyber|Network",
    "Python & Software": "Software|Python|Developer"
}

random.seed(42)
student_rows = []
for i in range(1, 351):
    course_row = courses_master.iloc[i % len(courses_master)]
    domain = course_row["domain"]

    c_skills = [re.sub(r"[^a-zA-Z0-9+#.]", "", s) for s in tech_courses.iloc[i % len(tech_courses)]["Skills"].split()]
    c_skills = [s for s in c_skills if len(s) > 2][:4]

    career_kw = domain_career_keywords[domain]
    matched_k_skills = hard_skills[hard_skills["related_careers"].str.contains(career_kw, case=False, na=False)]["clean_skill"].unique().tolist()
    extra_skills = random.sample(matched_k_skills, k=min(2, len(matched_k_skills))) if matched_k_skills else []

    combined_skills = list(dict.fromkeys(c_skills + extra_skills))
    if len(combined_skills) < 2:
        combined_skills += ["Python", "SQL"]
    chosen_skills = random.sample(combined_skills, k=min(len(combined_skills), random.randint(2, 4)))

    student_rows.append({
        "student_id": f"STU{i:04d}",
        "branch": random.choice(domain_branch_map[domain]),
        "year": random.choice([2, 3, 4]),
        "skills": ", ".join(chosen_skills),
        "interest_area": domain,
        "course_taken": course_row["course_name"],
        "completed": random.choice(["Yes", "Yes", "Yes", "No"])
    })

pd.DataFrame(student_rows).to_csv("data/student_data.csv", index=False)
print(f"Success! Created courses_master.csv ({len(courses_master)} courses) and student_data.csv ({len(student_rows)} records).")