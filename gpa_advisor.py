"""
AI-Based GPA Calculator & Academic Advisor
A rule-based expert system (knowledge representation + inference) built with Streamlit.

Run locally:
    pip install streamlit pandas
    streamlit run gpa_advisor.py
"""

import pandas as pd
import streamlit as st

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
# 6. STREAMLIT USER INTERFACE
# ---------------------------------------------------------------
st.set_page_config(page_title="AI GPA Advisor", page_icon="🎓")
st.title("🎓 AI-Based GPA Calculator & Academic Advisor")
st.caption("A rule-based expert system: facts + if-then rules + inference.")

st.subheader("1. Enter your courses")
default = pd.DataFrame(
    {
        "Course": ["Data Structure", "Database Systems", "Artificial Intelligence", "Statistics and Probability", "Software Engineering"],
        "Credit Hours": [3.0, 3.0, 3.0, 3.0, 3.0],
        "Grade": ["B+", "B", "A"],
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
        st.subheader("2. Your result")
        c1, c2 = st.columns(2)
        c1.metric("GPA", f"{gpa:.2f}")
        c2.metric("Total Credit Hours", f"{credits:g}")

        st.subheader("3. Advisor recommendations")
        facts = derive_facts(courses, gpa, credits)
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

st.divider()
st.subheader("4. Target GPA planner")
target = st.number_input("Target overall GPA", 0.0, 4.0, 3.5, 0.05)
remaining = st.number_input("Remaining credit hours in your degree", 1, 150, 60)

if st.button("Plan my target"):
    if "points" not in st.session_state:
        st.warning("First calculate your GPA in the section above.")
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
