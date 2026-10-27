"""
Employee Productivity vs. Workload Intelligence Dashboard
A lightweight, transparent Streamlit application analyzing employee productivity,
workload bottlenecks, and supportive intervention recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np


# -----------------------------------------------------------------------------
# 1. Synthetic Data Generation (Deterministic)
# -----------------------------------------------------------------------------
@st.cache_data
def generate_sample_data(seed: int = 42) -> pd.DataFrame:
    """
    Generates a deterministic dataset of 30 employees across 5 departments.
    Includes intentional patterns:
    - High overtime with declining productivity (fatigue pattern)
    - High assigned workload with low completion (bottleneck pattern)
    - High absenteeism with missed deadlines (coverage/health constraints)
    - Low engagement with declining performance (morale friction pattern)
    - High productivity with healthy, reasonable hours (sustainable benchmark)
    """
    np.random.seed(seed)

    departments = ["IT", "Marketing", "Sales", "Operations", "HR"]
    records = []

    for i in range(1, 31):
        emp_id = f"EMP{i:03d}"
        dept = departments[(i - 1) % len(departments)]

        # Base balanced parameters
        weekly_hours = round(float(np.random.uniform(38.0, 42.0)), 1)
        overtime = round(float(np.random.uniform(0.0, 4.0)), 1)
        absenteeism = int(np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1]))
        tasks_assigned = int(np.random.randint(15, 25))
        tasks_completed = int(tasks_assigned - np.random.randint(0, 3))
        avg_task_hours = round(float(np.random.uniform(2.0, 3.5)), 1)
        engagement = int(np.random.randint(6, 10))
        performance = int(np.random.randint(3, 5))
        prev_prod = round(float(np.random.uniform(75.0, 90.0)), 1)
        curr_prod = round(float(prev_prod + np.random.uniform(-3.0, 5.0)), 1)

        # Inject realistic archetypes & operational patterns:
        if i in [2, 7]:
            # Archetype 1: Severe fatigue/burnout (High Risk: >=3 flags)
            weekly_hours = round(float(np.random.uniform(50.0, 56.0)), 1)
            overtime = round(float(np.random.uniform(10.0, 15.0)), 1)
            prev_prod = round(float(np.random.uniform(82.0, 90.0)), 1)
            curr_prod = round(float(prev_prod - np.random.uniform(14.0, 20.0)), 1)
            tasks_completed = max(1, int(tasks_assigned * 0.70))
            engagement = int(np.random.choice([3, 4]))

        elif i in [5, 12]:
            # Archetype 2: Workload bottleneck (High Risk: >=3 flags)
            tasks_assigned = int(np.random.randint(32, 42))
            tasks_completed = int(tasks_assigned * np.random.uniform(0.55, 0.70))
            overtime = round(float(np.random.uniform(9.0, 12.0)), 1)
            curr_prod = round(float(prev_prod - np.random.uniform(12.0, 18.0)), 1)

        elif i in [9]:
            # Archetype 3: Absenteeism & coverage constraint (High Risk: >=3 flags)
            absenteeism = int(np.random.randint(4, 6))
            tasks_completed = int(tasks_assigned * 0.68)
            curr_prod = round(float(prev_prod - 12.5), 1)

        elif i in [15, 22]:
            # Archetype 4: Medium Risk (2 flags: e.g. High Overtime + Low Completion, but steady productivity)
            overtime = round(float(np.random.uniform(9.0, 11.5)), 1)
            tasks_assigned = int(np.random.randint(28, 35))
            tasks_completed = int(tasks_assigned * 0.72)
            curr_prod = round(float(prev_prod - 3.0), 1)  # Drop is < 10%

        elif i in [17, 28]:
            # Archetype 5: Medium Risk (2 flags: e.g. Low Engagement + Falling Productivity, but normal hours)
            engagement = 3
            curr_prod = round(float(prev_prod - 13.0), 1)
            tasks_completed = int(tasks_assigned * 0.88)
            overtime = 2.0

        elif i in [14, 26]:
            # Archetype 6: Low engagement and morale friction (High Risk: >=3 flags)
            engagement = int(np.random.choice([2, 3]))
            performance = int(np.random.choice([1, 2]))
            prev_prod = round(float(np.random.uniform(70.0, 80.0)), 1)
            curr_prod = round(float(prev_prod - np.random.uniform(12.0, 18.0)), 1)
            tasks_completed = int(tasks_assigned * 0.72)

        elif i in [1, 10, 18, 25]:
            # Archetype 7: High sustainable productivity, healthy hours (Low Risk: 0 flags)
            weekly_hours = 40.0
            overtime = round(float(np.random.uniform(0.5, 2.0)), 1)
            absenteeism = 0
            tasks_assigned = 22
            tasks_completed = 22
            engagement = int(np.random.choice([9, 10]))
            performance = 5
            prev_prod = 88.0
            curr_prod = 94.0

        # Boundary checks
        tasks_completed = min(tasks_assigned, max(0, tasks_completed))
        curr_prod = max(10.0, min(100.0, curr_prod))
        prev_prod = max(10.0, min(100.0, prev_prod))
        engagement = max(1, min(10, engagement))
        performance = max(1, min(5, performance))

        records.append({
            "employee_id": emp_id,
            "department": dept,
            "weekly_hours": weekly_hours,
            "weekly_overtime_hours": overtime,
            "absenteeism_days": absenteeism,
            "tasks_assigned": tasks_assigned,
            "tasks_completed": tasks_completed,
            "avg_task_completion_hours": avg_task_hours,
            "engagement_score": engagement,
            "performance_rating": performance,
            "prev_productivity_score": prev_prod,
            "curr_productivity_score": curr_prod,
        })

    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# 2. Metric Calculations & Bottleneck Detection
# -----------------------------------------------------------------------------
def analyze_workload(
    df: pd.DataFrame,
    ot_threshold: float = 8.0,
    completion_threshold: float = 80.0,
    prod_drop_threshold: float = 10.0,
    absentee_threshold: int = 3,
    engagement_threshold: int = 4,
) -> pd.DataFrame:
    """
    Computes completion rates, productivity variance, evaluates prototype warning
    thresholds, and assigns risk categories and explanations.
    """
    df = df.copy()

    # Safe division for task completion rate (%)
    df["task_completion_rate"] = np.where(
        df["tasks_assigned"] > 0,
        (df["tasks_completed"] / df["tasks_assigned"]) * 100.0,
        0.0,
    ).round(1)

    # Safe division for productivity percentage change
    df["productivity_change"] = np.where(
        df["prev_productivity_score"] > 0,
        ((df["curr_productivity_score"] - df["prev_productivity_score"]) / df["prev_productivity_score"]) * 100.0,
        0.0,
    ).round(1)

    # Evaluate warning conditions
    def evaluate_warnings(row):
        reasons = []
        if row["weekly_overtime_hours"] > ot_threshold:
            reasons.append(f"High Overtime ({row['weekly_overtime_hours']} hrs > {ot_threshold} hrs)")
        if row["task_completion_rate"] < completion_threshold:
            reasons.append(f"Low Task Completion ({row['task_completion_rate']}% < {completion_threshold}%)")
        if row["productivity_change"] < -prod_drop_threshold:
            reasons.append(f"Productivity Decline ({row['productivity_change']}% < -{prod_drop_threshold}%)")
        if row["absenteeism_days"] > absentee_threshold:
            reasons.append(f"Elevated Absenteeism ({row['absenteeism_days']} days > {absentee_threshold} days)")
        if row["engagement_score"] <= engagement_threshold:
            reasons.append(f"Low Engagement Score ({row['engagement_score']} ≤ {engagement_threshold})")

        num_warnings = len(reasons)
        if num_warnings >= 3:
            risk = "High Risk"
        elif num_warnings == 2:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        return pd.Series({
            "warning_count": num_warnings,
            "risk_category": risk,
            "warning_reasons": reasons,
            "needs_review": num_warnings >= 1,
        })

    eval_df = df.apply(evaluate_warnings, axis=1)
    return pd.concat([df, eval_df], axis=1)


# -----------------------------------------------------------------------------
# 3. Recommendation Engine
# -----------------------------------------------------------------------------
def generate_recommendations(row: pd.Series) -> list[dict]:
    """
    Rule-based recommendation generation mapped directly to observed operational flags.
    Non-punitive, supportive interventions only.
    """
    recs = []
    emp_id = row["employee_id"]
    dept = row["department"]

    # Rule 1: High overtime + falling productivity
    if row["weekly_overtime_hours"] > 8.0 and row["productivity_change"] < -10.0:
        recs.append({
            "employee_id": emp_id,
            "department": dept,
            "type": "Overtime & Fatigue Mitigation",
            "trigger": f"Overtime ({row['weekly_overtime_hours']} hrs) with productivity drop ({row['productivity_change']}%)",
            "action": "Audit workload distribution, cap non-essential overtime, and reassign secondary tasks to avoid burnout.",
        })

    # Rule 2: Low task completion + high assigned workload
    if row["task_completion_rate"] < 80.0 and row["tasks_assigned"] >= 25:
        recs.append({
            "employee_id": emp_id,
            "department": dept,
            "type": "Scope & Bottleneck Review",
            "trigger": f"High task volume ({row['tasks_assigned']} assigned) with low completion ({row['task_completion_rate']}%)",
            "action": "Review cross-functional dependencies, refine delivery sprint scope, and provide unblocking support.",
        })

    # Rule 3: High absenteeism + missed tasks
    if row["absenteeism_days"] > 3 and row["task_completion_rate"] < 80.0:
        recs.append({
            "employee_id": emp_id,
            "department": dept,
            "type": "Coverage & Capacity Planning",
            "trigger": f"Absenteeism ({row['absenteeism_days']} days) alongside low completion rate ({row['task_completion_rate']}%)",
            "action": "Evaluate team backup coverage and operational schedule constraints without attributing personal blame.",
        })

    # Rule 4: Low engagement + declining productivity
    if row["engagement_score"] <= 4 and row["productivity_change"] < -10.0:
        recs.append({
            "employee_id": emp_id,
            "department": dept,
            "type": "Supportive Check-in",
            "trigger": f"Low engagement ({row['engagement_score']}/10) alongside declining productivity ({row['productivity_change']}%)",
            "action": "Conduct a supportive 1-on-1 check-in to identify workflow friction, clarity blockers, and well-being needs.",
        })

    # Rule 5: Multiple high-risk indicators
    if row["warning_count"] >= 3:
        recs.append({
            "employee_id": emp_id,
            "department": dept,
            "type": "Comprehensive Workload Action Plan",
            "trigger": f"{row['warning_count']} cumulative warning flags detected ({', '.join(row['warning_reasons'])})",
            "action": "Initiate a collaborative workload assessment with department lead to co-create a realistic workload rebalancing plan.",
        })

    return recs


# -----------------------------------------------------------------------------
# 4. Streamlit UI Layout
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Employee Productivity vs. Workload Intelligence Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Employee Productivity vs. Workload Intelligence Dashboard")
    st.markdown(
        "This dashboard analyzes the relationship between workload, overtime, absenteeism, "
        "engagement, and productivity across departments. It identifies operational bottlenecks "
        "and generates supportive, data-backed workload interventions."
    )

    # --- Sidebar Controls ---
    st.sidebar.header("🔍 Filters & Controls")

    # Refresh data button
    if st.sidebar.button("🔄 Refresh Analysis / Data"):
        st.cache_data.clear()
        st.rerun()

    raw_data = generate_sample_data()

    # Department filter
    dept_options = ["All"] + sorted(list(raw_data["department"].unique()))
    selected_dept = st.sidebar.selectbox("Filter by Department", dept_options)

    # Prototype threshold adjustments
    st.sidebar.subheader("⚙️ Prototype Review Thresholds")
    st.sidebar.caption("Thresholds are operational indicators, not rigid HR rules.")
    th_ot = st.sidebar.slider("Weekly Overtime Alert (hrs)", 4.0, 16.0, 8.0, 1.0)
    th_comp = st.sidebar.slider("Min Task Completion Alert (%)", 50.0, 95.0, 80.0, 5.0)
    th_prod_drop = st.sidebar.slider("Productivity Drop Alert (%)", 5.0, 25.0, 10.0, 1.0)
    th_absentee = st.sidebar.slider("Monthly Absenteeism Alert (days)", 1, 6, 3, 1)
    th_eng = st.sidebar.slider("Low Engagement Alert (≤ score)", 1, 6, 4, 1)

    # Process metrics with thresholds
    processed_df = analyze_workload(
        raw_data,
        ot_threshold=th_ot,
        completion_threshold=th_comp,
        prod_drop_threshold=th_prod_drop,
        absentee_threshold=th_absentee,
        engagement_threshold=th_eng,
    )

    # Risk category filter
    risk_options = ["All", "High Risk", "Medium Risk", "Low Risk"]
    selected_risk = st.sidebar.selectbox("Filter by Risk Category", risk_options)

    # Apply filters
    filtered_df = processed_df.copy()
    if selected_dept != "All":
        filtered_df = filtered_df[filtered_df["department"] == selected_dept]
    if selected_risk != "All":
        filtered_df = filtered_df[filtered_df["risk_category"] == selected_risk]

    # --- 1. KPI Cards ---
    st.subheader("📌 Key Operational Indicators")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    avg_prod = filtered_df["curr_productivity_score"].mean() if not filtered_df.empty else 0.0
    avg_ot = filtered_df["weekly_overtime_hours"].mean() if not filtered_df.empty else 0.0
    total_assigned = filtered_df["tasks_assigned"].sum()
    total_completed = filtered_df["tasks_completed"].sum()
    overall_completion = (total_completed / total_assigned * 100.0) if total_assigned > 0 else 0.0
    review_needed_count = int(filtered_df["needs_review"].sum())

    kpi_col1.metric("Avg Productivity Score", f"{avg_prod:.1f} / 100")
    kpi_col2.metric("Avg Weekly Overtime", f"{avg_ot:.1f} hrs")
    kpi_col3.metric("Overall Task Completion", f"{overall_completion:.1f}%")
    kpi_col4.metric("Employees Flagged for Review", f"{review_needed_count} / {len(filtered_df)}")

    st.divider()

    # --- 2. Productivity & Workload Charts ---
    st.subheader("📈 Productivity & Workload Visualizations")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Department Average Productivity Comparison**")
        dept_prod = (
            processed_df.groupby("department")["curr_productivity_score"]
            .mean()
            .round(1)
            .reset_index()
        )
        dept_prod.columns = ["Department", "Avg Productivity Score"]
        st.bar_chart(data=dept_prod.set_index("Department"), y="Avg Productivity Score", color="#2b8cbe")

    with chart_col2:
        st.markdown("**Overtime Hours vs. Current Productivity**")
        scatter_data = filtered_df[["weekly_overtime_hours", "curr_productivity_score", "department"]].copy()
        scatter_data.columns = ["Overtime Hours", "Productivity Score", "Department"]
        st.scatter_chart(data=scatter_data, x="Overtime Hours", y="Productivity Score", color="Department")

    # Additional Workload vs Task Completion chart
    st.markdown("**Assigned Workload vs. Task Completion Rate (%)**")
    workload_scatter = filtered_df[["tasks_assigned", "task_completion_rate", "department"]].copy()
    workload_scatter.columns = ["Tasks Assigned", "Task Completion Rate (%)", "Department"]
    st.scatter_chart(data=workload_scatter, x="Tasks Assigned", y="Task Completion Rate (%)", color="Department")

    st.divider()

    # --- 3. Declining Productivity Table ---
    st.subheader("📉 Employees with Declining Productivity")
    declining_df = filtered_df[filtered_df["productivity_change"] < 0].sort_values(
        by="productivity_change", ascending=True
    )

    if not declining_df.empty:
        declining_display = declining_df[[
            "employee_id",
            "department",
            "prev_productivity_score",
            "curr_productivity_score",
            "productivity_change",
            "weekly_overtime_hours",
            "engagement_score",
            "risk_category",
        ]].rename(columns={
            "employee_id": "Employee ID",
            "department": "Department",
            "prev_productivity_score": "Prev Score",
            "curr_productivity_score": "Current Score",
            "productivity_change": "Change (%)",
            "weekly_overtime_hours": "Overtime (hrs)",
            "engagement_score": "Engagement",
            "risk_category": "Risk Level",
        })
        st.dataframe(declining_display, use_container_width=True, hide_index=True)
    else:
        st.info("No employees in the current filter selection show declining productivity.")

    st.divider()

    # --- 4. Workload Review & Bottleneck Detection Table ---
    st.subheader("⚠️ Employee Workload Review & Bottleneck Analysis")
    st.caption("Displays employees matching one or more operational review criteria.")

    flagged_df = filtered_df[filtered_df["needs_review"]].sort_values(by="warning_count", ascending=False)

    if not flagged_df.empty:
        flagged_display = flagged_df[[
            "employee_id",
            "department",
            "risk_category",
            "warning_count",
            "weekly_overtime_hours",
            "task_completion_rate",
            "productivity_change",
            "absenteeism_days",
            "engagement_score",
            "warning_reasons",
        ]].copy()

        flagged_display["warning_reasons"] = flagged_display["warning_reasons"].apply(lambda r: "; ".join(r))
        flagged_display = flagged_display.rename(columns={
            "employee_id": "Employee ID",
            "department": "Department",
            "risk_category": "Risk Level",
            "warning_count": "Warnings Count",
            "weekly_overtime_hours": "Overtime (hrs)",
            "task_completion_rate": "Completion (%)",
            "productivity_change": "Prod Change (%)",
            "absenteeism_days": "Absenteeism (days)",
            "engagement_score": "Engagement",
            "warning_reasons": "Triggered Reasons",
        })
        st.dataframe(flagged_display, use_container_width=True, hide_index=True)
    else:
        st.success("No employees currently trigger workload review conditions under the chosen filters.")

    st.divider()

    # --- 5. Actionable Recommendation Engine ---
    st.subheader("💡 Targeted Workload Interventions & Recommendations")
    st.caption(
        "Rule-based recommendations based on identified operational bottlenecks. "
        "Intended for managerial workload balancing and employee support."
    )

    all_recommendations = []
    for _, row in filtered_df.iterrows():
        recs = generate_recommendations(row)
        all_recommendations.extend(recs)

    if all_recommendations:
        rec_df = pd.DataFrame(all_recommendations)
        rec_display = rec_df.rename(columns={
            "employee_id": "Employee ID",
            "department": "Department",
            "type": "Intervention Category",
            "trigger": "Triggered Evidence",
            "action": "Recommended Supportive Action",
        })
        st.dataframe(rec_display, use_container_width=True, hide_index=True)
    else:
        st.info("No specific operational recommendations triggered for the current selection.")

    st.divider()

    # --- 6. Complete Employee Dataset Table ---
    st.subheader("📋 Complete Employee Records (Filtered View)")
    full_table_display = filtered_df[[
        "employee_id",
        "department",
        "weekly_hours",
        "weekly_overtime_hours",
        "tasks_assigned",
        "tasks_completed",
        "task_completion_rate",
        "avg_task_completion_hours",
        "absenteeism_days",
        "engagement_score",
        "performance_rating",
        "prev_productivity_score",
        "curr_productivity_score",
        "productivity_change",
        "risk_category",
    ]].rename(columns={
        "employee_id": "ID",
        "department": "Dept",
        "weekly_hours": "Hours/Wk",
        "weekly_overtime_hours": "Overtime (h)",
        "tasks_assigned": "Tasks Assigned",
        "tasks_completed": "Tasks Done",
        "task_completion_rate": "Completion %",
        "avg_task_completion_hours": "Avg Task (h)",
        "absenteeism_days": "Absent Days",
        "engagement_score": "Engagement",
        "performance_rating": "Rating",
        "prev_productivity_score": "Prev Prod",
        "curr_productivity_score": "Curr Prod",
        "productivity_change": "Change %",
        "risk_category": "Risk Level",
    })
    st.dataframe(full_table_display, use_container_width=True, hide_index=True)

    # --- 7. Ethical Notice & Disclaimer ---
    st.markdown("---")
    st.caption(
        "🛡️ **Ethical & Operational Disclaimer**: This tool is an operational workload-balancing prototype. "
        "Risk categories and warning flags do not represent psychological, medical, or clinical diagnoses. "
        "These indicators must never be used to penalize, rank, or take punitive personnel actions against employees. "
        "All recommendations are intended solely to guide supportive managerial check-ins and equitable capacity planning."
    )


if __name__ == "__main__":
    main()
