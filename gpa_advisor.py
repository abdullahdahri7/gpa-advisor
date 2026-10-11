"""
AI-Based GPA Calculator & Academic Advisor
A rule-based expert system (knowledge representation + inference) built with Streamlit.

Run locally:
    pip install streamlit pandas
    streamlit run gpa_advisor.py
"""

import pandas as pd
import streamlit as st

# Shown in the footer of the app. Add your name here if you like.
DEVELOPER = "AI Course Project | Roll No: 25AI-87"

# ---------------------------------------------------------------
# 1. KNOWLEDGE BASE: grade scale (facts)
#    Scale taken from the university grading table.
# ---------------------------------------------------------------
GRADE_POINTS = {
    "A+": 4.00,
    "A": 3.50,
    "B+": 3.00,
    "B": 2.50,
    "C+": 2.00,
    "C": 1.50,
    "C-": 1.00,
    "F": 0.00,
}


# ---------------------------------------------------------------
# 2. GPA CALCULATION
# ---------------------------------------------------------------
def calculate_gpa(courses: pd.DataFrame):
    """Return (gpa, total_credits, total_quality_points)."""
    total_credits = 0.0
    total_points = 0.0
    for _, row in courses.iterrows():
        grade = row["Grade"]
        credits = row["Credit Hours"]
        if grade in GRADE_POINTS and pd.notna(credits) and credits > 0:
            total_credits += credits
            total_points += GRADE_POINTS[grade] * credits
    gpa = total_points / total_credits if total_credits else 0.0
    return gpa, total_credits, total_points


def cgpa_from_semesters(sem: pd.DataFrame):
    """CGPA from semester GPAs, weighted by credit hours.
    Return (cgpa, total_credits, total_quality_points)."""
    total_credits = 0.0
    total_points = 0.0
    for _, row in sem.iterrows():
        gpa = row["GPA"]
        credits = row["Credit Hours"]
        if pd.notna(gpa) and pd.notna(credits) and credits > 0 and 0 <= gpa <= 4:
            total_credits += credits
            total_points += gpa * credits
    cgpa = total_points / total_credits if total_credits else 0.0
    return cgpa, total_credits, total_points


# ---------------------------------------------------------------
# 3. KNOWLEDGE BASE: if-then rules
#    Each rule = (name, condition on facts, advice text)
# ---------------------------------------------------------------
RULES = [
    ("R1: Merit list",
     lambda f: f["gpa"] >= 3.50 and not f["has_fail"],
     "Excellent! You are eligible for the merit/dean's list. Keep it up."),

    ("R2: Good standing",
     lambda f: 3.00 <= f["gpa"] < 3.50 and not f["has_fail"],
     "Good standing. A little more effort can push you into the merit list."),

    ("R3: Average standing",
     lambda f: 2.00 <= f["gpa"] < 3.00 and not f["has_fail"],
     "Average standing. Focus on your weaker courses to improve your GPA."),

    ("R4: Probation warning",
     lambda f: f["gpa"] < 2.00,
     "WARNING: Your GPA is below 2.0. You are at risk of academic probation. "
     "Meet your advisor and make an improvement plan."),

    ("R5: Failed course",
     lambda f: f["has_fail"],
     "You have at least one F. Plan to repeat the failed course(s) to recover your GPA."),

    ("R6: Weak courses",
     lambda f: f["low_grade_count"] >= 2,
     "You have 2 or more courses with C- or below. Consider extra practice or tutoring."),

    ("R7: Light load recommended",
     lambda f: f["is_warning"],
     "Because you are in a risky standing, take a lighter course load next semester."),

    ("R8: Strong start",
     lambda f: f["gpa"] >= 3.0 and f["total_credits"] <= 30,
     "You have a strong start in your degree. Maintain consistency."),
]


# ---------------------------------------------------------------
# 4. INFERENCE ENGINE (forward chaining)
#    Step 1: derive facts from the data.
#    Step 2: fire every rule whose condition is true.
# ---------------------------------------------------------------
def derive_facts(courses: pd.DataFrame, gpa: float, total_credits: float) -> dict:
    graded = courses[courses["Grade"].isin(GRADE_POINTS.keys())]
    low_grades = ["C-", "F"]
    facts = {
        "gpa": gpa,
        "total_credits": total_credits,
        "has_fail": bool((graded["Grade"] == "F").any()),
        "low_grade_count": int(graded["Grade"].isin(low_grades).sum()),
    }
    # Derived (inferred) fact used by another rule -> chaining
    facts["is_warning"] = facts["gpa"] < 2.50 or facts["has_fail"]
    return facts


def run_inference(facts: dict):
    """Return the list of (rule name, advice) that fired."""
    return [(name, advice) for name, cond, advice in RULES if cond(facts)]


# ---------------------------------------------------------------
# 5. TARGET GPA PLANNER
# ---------------------------------------------------------------
def required_gpa(current_points, current_credits, target_gpa, remaining_credits):
    """GPA needed in the remaining credit hours to reach the target overall GPA."""
    total = current_credits + remaining_credits
    needed_points = target_gpa * total - current_points
    return needed_points / remaining_credits


def grade_for_gpa(value):
    """Lowest letter grade that meets a required GPA."""
    for grade, points in sorted(GRADE_POINTS.items(), key=lambda x: x[1]):
        if points >= value:
            return grade
    return None


# ---------------------------------------------------------------
# 6. STREAMLIT USER INTERFACE (premium look)
# ---------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [data-testid="stAppViewContainer"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 920px; }

.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
    border-radius: 20px; padding: 2.2rem 2rem; margin-bottom: 1.5rem;
    box-shadow: 0 12px 30px rgba(79, 70, 229, 0.35);
}
.hero h1 { color: #ffffff !important; font-size: 2rem; font-weight: 800;
           margin: 0 0 0.4rem 0; padding: 0; line-height: 1.2; }
.hero p { color: rgba(255, 255, 255, 0.92) !important; margin: 0 0 1rem 0; font-size: 1rem; }
.badge { display: inline-block; background: rgba(255, 255, 255, 0.18);
         border: 1px solid rgba(255, 255, 255, 0.4); color: #ffffff;
         padding: 0.25rem 0.8rem; border-radius: 999px; font-size: 0.78rem;
         font-weight: 600; margin: 0 0.4rem 0.4rem 0; }

.section-title { display: flex; align-items: center; gap: 0.7rem;
                 font-size: 1.2rem; font-weight: 700; margin: 0.2rem 0 0.8rem 0; }
.section-num { display: inline-flex; align-items: center; justify-content: center;
               width: 2rem; height: 2rem; border-radius: 50%; color: #ffffff;
               background: linear-gradient(135deg, #4f46e5, #7c3aed);
               font-size: 0.9rem; font-weight: 700; }

[data-testid="stMetric"] { background: rgba(124, 58, 237, 0.08);
    border: 1px solid rgba(124, 58, 237, 0.28); border-radius: 14px;
    padding: 1rem 1.1rem; box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06); }
[data-testid="stMetricValue"] { font-weight: 800; }

.stButton > button { background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: #ffffff; border: none; border-radius: 12px; padding: 0.6rem 1.4rem;
    font-weight: 600; box-shadow: 0 6px 16px rgba(79, 70, 229, 0.35);
    transition: transform 0.15s ease, box-shadow 0.15s ease; }
.stButton > button p { color: #ffffff; }
.stButton > button:hover { transform: translateY(-2px); color: #ffffff; border: none;
    box-shadow: 0 10px 22px rgba(79, 70, 229, 0.45); }

[data-testid="stAlert"] { border-radius: 12px; }
.standing { display: inline-block; padding: 0.35rem 0.95rem; border-radius: 999px;
            font-weight: 700; font-size: 0.9rem; color: #ffffff; margin: 0.4rem 0 0.6rem 0; }
.footer { text-align: center; opacity: 0.65; font-size: 0.82rem;
          margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(128, 128, 128, 0.3); }
</style>
"""


def section(num, title):
    """Numbered section heading."""
    st.markdown(
        f'<div class="section-title"><span class="section-num">{num}</span>{title}</div>',
        unsafe_allow_html=True,
    )


def standing(gpa, has_fail=False):
    """Label and color for the standing badge (display only)."""
    if has_fail or gpa < 2.0:
        return "At Risk", "#dc2626"
    if gpa >= 3.5:
        return "Merit List", "#16a34a"
    if gpa >= 3.0:
        return "Good Standing", "#2563eb"
    return "Average Standing", "#d97706"


def show_standing(gpa, has_fail=False):
    """Colored badge and a progress bar out of 4.0."""
    label, color = standing(gpa, has_fail)
    st.markdown(
        f'<span class="standing" style="background:{color}">{label}</span>',
        unsafe_allow_html=True,
    )
    st.progress(min(max(float(gpa) / 4.0, 0.0), 1.0), text=f"{gpa:.2f} out of 4.00")


st.set_page_config(page_title="AI GPA Advisor", page_icon="🎓")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
<div class="hero">
<h1>🎓 AI-Based GPA &amp; CGPA Advisor</h1>
<p>Calculate your GPA and CGPA, get smart academic advice, and plan the grades you need for your target.</p>
<span class="badge">Rule-Based Expert System</span><span class="badge">Forward Chaining</span><span class="badge">Knowledge Base</span>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------- GPA from courses ----------------
with st.container(border=True):
    section(1, "Enter your courses")
    default = pd.DataFrame(
        {
            "Course": ["Data Structure", "Database Systems", "Artificial Intelligence", "Statistics and Probability", "Software Engineering"],
            "Credit Hours": [3.0, 3.0, 3.0, 3.0, 3.0],
            "Grade": ["B+", "B", "A", "B+", "B"],
        }
    )
    courses = st.data_editor(
        default,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Grade": st.column_config.SelectboxColumn(
                "Grade", options=list(GRADE_POINTS.keys()), required=True
            ),
            "Credit Hours": st.column_config.NumberColumn(
                "Credit Hours", min_value=1.0, max_value=6.0, step=1.0
            ),
        },
    )

    if st.button("Calculate GPA & Get Advice", type="primary"):
        gpa, credits, points = calculate_gpa(courses)

        if credits == 0:
            st.error("Please enter at least one course with a grade and credit hours.")
        else:
            facts = derive_facts(courses, gpa, credits)

            section(2, "Your result")
            c1, c2 = st.columns(2)
            c1.metric("GPA", f"{gpa:.2f}")
            c2.metric("Total Credit Hours", f"{credits:g}")
            show_standing(gpa, facts["has_fail"])

            section(3, "Advisor recommendations")
            fired = run_inference(facts)
            for name, advice in fired:
                if "WARNING" in advice or "F." in advice:
                    st.error(f"{advice}")
                else:
                    st.success(f"{advice}")
                st.caption(f"Rule fired: {name}")

            with st.expander("Show facts used by the inference engine"):
                st.json(facts)

            # Save for the planner below
            st.session_state["points"] = points
            st.session_state["credits"] = credits

# ---------------- CGPA from semester GPAs ----------------
with st.container(border=True):
    section(4, "CGPA from semester GPAs")
    st.write(
        "Already know your GPA for each semester? Enter them here. For example, "
        "enter Semester 1 and Semester 2 to get your first-year CGPA. The CGPA is "
        "weighted by credit hours (if every semester has the same credit hours, "
        "it is just the average of the GPAs)."
    )
    sem_default = pd.DataFrame(
        {"Semester": [1, 2], "GPA": [2.5, 3.0], "Credit Hours": [18.0, 18.0]}
    )
    sem_df = st.data_editor(
        sem_default,
        num_rows="dynamic",
        use_container_width=True,
        key="semester_editor",
        column_config={
            "Semester": st.column_config.NumberColumn(
                "Semester", min_value=1, max_value=8, step=1, format="%d"
            ),
            "GPA": st.column_config.NumberColumn(
                "GPA", min_value=0.0, max_value=4.0, step=0.01
            ),
            "Credit Hours": st.column_config.NumberColumn(
                "Credit Hours", min_value=1.0, max_value=30.0, step=0.5
            ),
        },
    )

    if st.button("Calculate CGPA", key="calc_semester_cgpa"):
        cgpa, cgpa_credits, cgpa_points = cgpa_from_semesters(sem_df)

        if cgpa_credits == 0:
            st.error("Please enter at least one semester with a GPA and credit hours.")
        else:
            c1, c2 = st.columns(2)
            c1.metric("CGPA", f"{cgpa:.2f}")
            c2.metric("Total Credit Hours", f"{cgpa_credits:g}")
            show_standing(cgpa)

            st.write("**Advisor recommendations**")
            st.caption("Based on your CGPA only, because course grades were not entered.")
            cgpa_facts = {
                "gpa": cgpa,
                "total_credits": cgpa_credits,
                "has_fail": False,
                "low_grade_count": 0,
            }
            cgpa_facts["is_warning"] = cgpa_facts["gpa"] < 2.50
            for name, advice in run_inference(cgpa_facts):
                if "WARNING" in advice:
                    st.error(f"{advice}")
                else:
                    st.success(f"{advice}")
                st.caption(f"Rule fired: {name}")

            with st.expander("Show facts used by the inference engine"):
                st.json(cgpa_facts)

            # Save for the planner below
            st.session_state["points"] = cgpa_points
            st.session_state["credits"] = cgpa_credits

# ---------------- Target planner ----------------
with st.container(border=True):
    section(5, "Target GPA planner")
    target = st.number_input("Target overall GPA", 0.0, 4.0, 3.5, 0.05)
    remaining = st.number_input("Remaining credit hours in your degree", 1, 150, 60)

    if st.button("Plan my target"):
        if "points" not in st.session_state:
            st.warning("First calculate your GPA or CGPA in the sections above.")
        else:
            need = required_gpa(
                st.session_state["points"], st.session_state["credits"], target, remaining
            )
            if need > 4.0:
                st.error(
                    f"You would need a {need:.2f} GPA in the remaining courses, "
                    "which is above 4.0. This target is not reachable."
                )
            elif need <= 0:
                st.success("You have already secured this target GPA.")
            else:
                grade = grade_for_gpa(need)
                st.info(
                    f"You need an average GPA of **{need:.2f}** in the remaining "
                    f"{remaining} credit hours (about a **{grade}** average)."
                )

st.markdown(
    f'<div class="footer">Built with Python &amp; Streamlit · {DEVELOPER}</div>',
    unsafe_allow_html=True,
)
