import streamlit as st
import pdfplumber
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Tapasya SFG Rank Analyzer", 
    layout="wide", 
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .metric-card {
        background-color: #0E1117;
        border: 1px solid #262730;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #0E1117;
        border-radius: 4px;
        color: #FAFAFA;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
DATA_FOLDER = 'data'
DEFAULT_TOTAL_QUESTIONS = 60
MARKS_CORRECT = 2.0
MARKS_INCORRECT = 0.66

# --- HELPER FUNCTIONS ---
def clean_batch_name(text):
    if not isinstance(text, str): return "Unknown"
    match = re.search(r'FRC[\s-]*(\d+)', text, re.IGNORECASE)
    if match: return f"FRC-{match.group(1)}"
    return "Others"

def extract_test_number(filename):
    match = re.search(r'Test\s*(\d+)', filename, re.IGNORECASE)
    if match: return int(match.group(1))
    match_generic = re.search(r'(\d+)', filename)
    if match_generic: return int(match_generic.group(1))
    return 0

def clean_student_name(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'\d+', '', text)
    text = text.replace('.', '').replace('\n', ' ')
    return " ".join(text.split()).upper()

# --- DATA ENGINE ---
@st.cache_data
def load_data():
    all_records = []
    logs = []
    
    if not os.path.exists(DATA_FOLDER):
        return pd.DataFrame(), [f"❌ Error: Folder '{DATA_FOLDER}' not found."]

    files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith('.pdf')]
    if not files:
        return pd.DataFrame(), ["⚠️ No PDF files found in 'data/' folder."]

    for filename in files:
        file_path = os.path.join(DATA_FOLDER, filename)
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    table = page.extract_table()
                    if not table: continue
                    for row in table:
                        clean_row = [str(x).replace('\n', ' ').strip() if x else "" for x in row]
                        if not clean_row or "Name" in clean_row[0] or "Sl No" in clean_row[0]: continue

                        try:
                            # Smart Parsing Logic
                            if clean_row[0].isdigit():
                                raw_name = clean_row[1]
                                raw_batch_roll = clean_row[2]
                                data_start_idx = 3
                                batch = clean_batch_name(raw_batch_roll)
                                if batch == "Others" and len(clean_row) > 3:
                                    batch = clean_batch_name(clean_row[3])
                            else:
                                raw_name = clean_row[0]
                                raw_batch_roll = clean_row[1]
                                data_start_idx = 2
                                batch = clean_batch_name(raw_batch_roll)

                            numerics = []
                            for cell in clean_row[data_start_idx:]:
                                try:
                                    numerics.append(float(cell))
                                except: continue
                            
                            if len(numerics) >= 3:
                                correct, incorrect, score = numerics[0], numerics[1], numerics[2]
                                rank = int(numerics[3]) if len(numerics) >= 4 else 0
                                name = clean_student_name(raw_name)
                                
                                if len(name) < 2: continue
                                if correct > DEFAULT_TOTAL_QUESTIONS + 5: continue

                                test_num = extract_test_number(filename)
                                test_clean_name = f"Test {test_num}" if test_num > 0 else filename

                                all_records.append({
                                    "Test_ID": test_clean_name,
                                    "Test_Number": test_num,
                                    "Name": name,
                                    "Batch": batch,
                                    "Correct": correct,
                                    "Incorrect": incorrect,
                                    "Score": score,
                                    "Rank": rank,
                                    "Attempts": correct + incorrect,
                                    "Accuracy": (correct / (correct + incorrect) * 100) if (correct + incorrect) > 0 else 0
                                })
                        except: continue
            logs.append(f"✅ Loaded {filename}")
        except Exception as e:
            logs.append(f"❌ Error reading {filename}: {str(e)}")

    df = pd.DataFrame(all_records)
    if not df.empty:
        valid_batches = ['FRC-8', 'FRC-9', 'FRC-10']
        df = df[df['Batch'].isin(valid_batches)]
        df = df.sort_values('Test_Number')
        df = df.drop_duplicates(subset=['Test_ID', 'Name'], keep='first')
        
        # Add a helper column: Is_Active_Attempt (1 if Attempts > 0, else 0)
        df['Is_Active_Attempt'] = df['Attempts'].apply(lambda x: 1 if x > 0 else 0)

    return df, logs

# --- CHARTS & HELPERS ---
def get_topper_name(full_df):
    """Returns the name of the student with the highest performance average."""
    if full_df.empty: return None
    # Filter for active attempts only for finding topper
    active_df = full_df[full_df['Attempts'] > 0]
    avg_scores = active_df.groupby('Name')['Score'].mean().sort_values(ascending=False)
    if not avg_scores.empty:
        return avg_scores.index[0]
    return full_df['Name'].iloc[0]

def render_growth_charts(student_df, full_df):
    student_df = student_df.sort_values('Test_Number')
    
    fig_att = px.line(student_df, x='Test_ID', y='Attempts', markers=True, title="📈 Trend: Number of Attempts", line_shape='spline')
    fig_att.update_traces(line_color='#FFA500')
    st.plotly_chart(fig_att, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_corr = px.line(student_df, x='Test_ID', y='Correct', markers=True, title="✅ Trend: Correct Answers")
        fig_corr.update_traces(line_color='#00CC96')
        st.plotly_chart(fig_corr, use_container_width=True)
    with c2:
        fig_inc = px.line(student_df, x='Test_ID', y='Incorrect', markers=True, title="❌ Trend: Incorrect Answers")
        fig_inc.update_traces(line_color='#EF553B')
        st.plotly_chart(fig_inc, use_container_width=True)

    fig_score = px.line(student_df, x='Test_ID', y='Score', markers=True, title="🏆 Trend: Total Score", line_shape='spline')
    fig_score.add_hline(y=student_df['Score'].mean(), line_dash="dash", annotation_text="Your Avg")
    st.plotly_chart(fig_score, use_container_width=True)

    fig_rank = px.line(student_df, x='Test_ID', y='Rank', markers=True, title="🏅 Trend: Rank (Lower is Better)")
    fig_rank.update_yaxes(autorange="reversed") 
    st.plotly_chart(fig_rank, use_container_width=True)

    # Comparison Line Chart
    comparison_data = []
    for test_id in student_df['Test_ID'].unique():
        test_data = full_df[full_df['Test_ID'] == test_id]
        if test_data.empty: continue
        comparison_data.append({
            'Test_ID': test_id,
            'Test_Number': student_df[student_df['Test_ID'] == test_id]['Test_Number'].values[0],
            'My Score': student_df[student_df['Test_ID'] == test_id]['Score'].values[0],
            'Batch Avg': test_data['Score'].mean(),
            'Topper Score': test_data['Score'].max()
        })
    comp_df = pd.DataFrame(comparison_data).sort_values('Test_Number')
    
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=comp_df['Test_ID'], y=comp_df['My Score'], name='My Score', line=dict(color='#636EFA', width=4), mode='lines+markers'))
    fig_comp.add_trace(go.Scatter(x=comp_df['Test_ID'], y=comp_df['Batch Avg'], name='Batch Avg', line=dict(color='orange', width=2, dash='dash')))
    fig_comp.add_trace(go.Scatter(x=comp_df['Test_ID'], y=comp_df['Topper Score'], name='Topper Score', line=dict(color='green', width=2, dash='dot')))
    fig_comp.update_layout(title="📊 Performance vs Others", hovermode="x unified")
    st.plotly_chart(fig_comp, use_container_width=True)

def render_predictor(avg_attempts, avg_acc, avg_score):
    st.markdown("#### 🤖 Scenario Planner")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        target_attempts = st.slider("🎯 Target Attempts", 1, DEFAULT_TOTAL_QUESTIONS, int(avg_attempts), key=f"att_{avg_score}")
    with p_col2:
        target_accuracy = st.slider("🎯 Target Accuracy (%)", 1, 100, int(avg_acc), key=f"acc_{avg_score}")
    pred_correct = target_attempts * (target_accuracy / 100)
    pred_incorrect = target_attempts - pred_correct
    pred_score = (pred_correct * MARKS_CORRECT) - (pred_incorrect * MARKS_INCORRECT)
    
    pr_c1, pr_c2, pr_c3 = st.columns(3)
    pr_c1.metric("Predicted Score", f"{round(pred_score, 2)}", delta=f"{round(pred_score - avg_score, 2)}")
    pr_c2.metric("Predicted Correct", f"{int(pred_correct)}")
    pr_c3.metric("Predicted Incorrect", f"{int(pred_incorrect)}")

# --- APP ---
df, logs = load_data()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("Select a specific test in the 'Test Drill-Down' tab to view batch details.")
    with st.expander("System Logs"):
        for log in logs: st.caption(log)

# --- MAIN LAYOUT ---
st.title("📊 Tapasya SFG Rank Analyzer")

tab_dashboard, tab_student, tab_compare, tab_test_drill = st.tabs([
    "🏆 Overall Dashboard", 
    "🔍 Analyze my Performance", 
    "⚔️ Compare Students",
    "📅 Test Drill-Down"
])

# --- TAB 1: OVERALL DASHBOARD ---
with tab_dashboard:
    if df.empty:
        st.warning("No data available.")
    else:
        # Calculate Aggregates
        total_tests_conducted = df['Test_ID'].nunique()
        active_students = df['Name'].nunique()
        avg_batch_score = df['Score'].mean()
        
        # --- TOP KPI METRICS ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Tests Conducted", total_tests_conducted)
        m2.metric("Active Students", active_students)
        m3.metric("Batch Avg Score", f"{avg_batch_score:.2f}")

        # ==============================================================================
        # 1. CONSISTENCY CHAMPIONS (Avg of Active Attempts) - IGNORES 0 ATTEMPTS
        #    (Formerly High Performers, now swapped to Consistency as per request)
        # ==============================================================================
        # Filter: Only consider rows where student actually attempted questions
        active_attempts_df = df[df['Attempts'] > 0].copy()
        
        # This DataFrame calculates avg only when present
        consistency_leaderboard = active_attempts_df.groupby(['Name']).agg({
            'Score': 'mean',          # Average of ACTIVE tests only
            'Accuracy': 'mean',
            'Is_Active_Attempt': 'count' # Count of ACTIVE tests
        }).reset_index()
        
        consistency_leaderboard = consistency_leaderboard.rename(columns={'Is_Active_Attempt': 'Tests Taken', 'Score': 'Avg Score'})
        consistency_leaderboard = consistency_leaderboard.sort_values('Avg Score', ascending=False)
        consistency_leaderboard['Rank'] = range(1, len(consistency_leaderboard) + 1)
        
        # Metric for Top Performer
        if not consistency_leaderboard.empty:
            m4.metric("Top Consistency Avg", f"{consistency_leaderboard['Avg Score'].iloc[0]:.2f}")
        
        st.divider()

        st.subheader("Ranks Based on Tests Appeared")
        st.caption("This ranking is based on the average score of **only the tests the student appeared for**. (Ignores missed tests - measures consistency when present).")
        
        st.dataframe(
            consistency_leaderboard[['Rank', 'Name', 'Avg Score', 'Accuracy', 'Tests Taken']]
            .style.format({'Avg Score': '{:.2f}', 'Accuracy': '{:.1f}%'})
            .background_gradient(subset=['Avg Score'], cmap='Greens'),
            use_container_width=True, hide_index=True
        )

        st.divider()

        # ==============================================================================
        # 2. Absolute Rank (Factoring in Tests Missed) - PENALIZES MISSED TESTS
        #    (Formerly Consistency Champions, now swapped to High Performers)
        # ==============================================================================
        st.subheader("🎯 Absolute Rank (Factoring in Tests Missed)")
        st.caption(f"This ranking calculates average based on **ALL {total_tests_conducted} TESTS** conducted so far. Missing a test counts as 0 (measures absolute performance).")

        # Logic: Sum total score from FULL dataframe (including 0s)
        perf_abs_df = df.groupby(['Name']).agg({
            'Score': 'sum',
            'Is_Active_Attempt': 'sum'
        }).reset_index()
        
        # Divide by TOTAL TESTS CONDUCTED (Constant)
        perf_abs_df['Performance Avg'] = perf_abs_df['Score'] / total_tests_conducted
        perf_abs_df['Attendance %'] = (perf_abs_df['Is_Active_Attempt'] / total_tests_conducted) * 100
        perf_abs_df['Tests Missed'] = total_tests_conducted - perf_abs_df['Is_Active_Attempt']
        
        perf_abs_df = perf_abs_df.sort_values('Performance Avg', ascending=False)
        perf_abs_df['Absolute Rank'] = range(1, len(perf_abs_df) + 1)
        
        st.dataframe(
            perf_abs_df[['Absolute Rank', 'Name', 'Performance Avg', 'Attendance %', 'Tests Missed']]
            .style.format({'Performance Avg': '{:.2f}', 'Attendance %': '{:.1f}%'})
            .background_gradient(subset=['Performance Avg'], cmap='Oranges'),
            use_container_width=True, hide_index=True
        )

# --- TAB 2: Analyse my Performance ---
with tab_student:
    if df.empty:
        st.warning("No data.")
    else:
        st.markdown("### 🔍 Analyze a Student")
        
        topper_name = get_topper_name(df)
        all_students = sorted(df['Name'].unique().tolist())
        
        try:
            default_index = all_students.index(topper_name) if topper_name in all_students else 0
        except:
            default_index = 0
            
        selected_student = st.selectbox("Select Name", all_students, index=default_index)
        
        stu_df = df[df['Name'] == selected_student].copy()
        
        if not stu_df.empty:
            # --- 1. Calculate CONSISTENCY RANK (Based on Attempts > 0) ---
            # (Formerly Performance Rank)
            active_attempts_all = df[df['Attempts'] > 0].copy()
            cons_lb = active_attempts_all.groupby('Name')['Score'].mean().reset_index().sort_values('Score', ascending=False)
            cons_lb['Rank'] = range(1, len(cons_lb) + 1)
            
            if selected_student in cons_lb['Name'].values:
                consistency_rank = cons_lb[cons_lb['Name'] == selected_student]['Rank'].values[0]
            else:
                consistency_rank = "N/A (No Attempts)"

            # --- 2. Calculate PERFORMANCE RANK (Based on Total Tests) ---
            # (Formerly Consistency Rank)
            total_tests_conducted = df['Test_ID'].nunique()
            perf_temp = df.groupby('Name')['Score'].sum().reset_index()
            perf_temp['Real_Avg'] = perf_temp['Score'] / total_tests_conducted
            perf_temp = perf_temp.sort_values('Real_Avg', ascending=False)
            perf_temp['Rank'] = range(1, len(perf_temp) + 1)
            
            if selected_student in perf_temp['Name'].values:
                performance_rank = perf_temp[perf_temp['Name'] == selected_student]['Rank'].values[0]
            else:
                performance_rank = "N/A"

            tests_taken_count = stu_df['Is_Active_Attempt'].sum()
            
            # Use Active Average for display (Score when present)
            active_avg_display = stu_df[stu_df['Attempts'] > 0]['Score'].mean()
            if pd.isna(active_avg_display): active_avg_display = 0.0

            # KPI Row - LABELS SWAPPED HERE
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Tests Taken", f"{tests_taken_count}/{total_tests_conducted}")
            k2.metric("Performance Rank", f"#{consistency_rank}", help="Rank considering only tests appeared (Active attempts).")
            k3.metric("Absolute Rank", f"#{performance_rank}", help="Rank considering ALL tests (Absolute Performance).")
            k4.metric("Avg Score (When Present)", f"{active_avg_display:.2f}")
            
            st.divider()

            # ==============================================================================
            # NEW SECTION: HALL OF RECORDS (Extremes)
            # ==============================================================================
            # Filter specifically for Active Attempts to avoid "Lowest Score: 0" from missed tests
            stu_active = stu_df[stu_df['Attempts'] > 0]
            
            if not stu_active.empty:
                st.subheader("🌟 Hall of Records (Extremes)")
                st.caption("Statistics based on active attempts only.")

                # ROW 1: Rank & Score Extremes
                r1c1, r1c2, r1c3, r1c4 = st.columns(4)
                r1c1.metric("🏆 Best Rank", f"#{int(stu_active['Rank'].min())}")
                r1c2.metric("📉 Worst Rank", f"#{int(stu_active['Rank'].max())}")
                r1c3.metric("🚀 Highest Score", f"{stu_active['Score'].max()}")
                r1c4.metric("🐢 Lowest Score", f"{stu_active['Score'].min()}")
                
                # ROW 2: Attempts & Corrections
                r2c1, r2c2, r2c3, r2c4 = st.columns(4)
                r2c1.metric("📝 Highest Attempts", f"{int(stu_active['Attempts'].max())}")
                r2c2.metric("💤 Lowest Attempts", f"{int(stu_active['Attempts'].min())}")
                r2c3.metric("✅ Highest Correct", f"{int(stu_active['Correct'].max())}")
                r2c4.metric("⚠️ Lowest Correct", f"{int(stu_active['Correct'].min())}")

                # ROW 3: Negatives
                r3c1, r3c2, r3c3, r3c4 = st.columns(4)
                r3c1.metric("❌ Highest Incorrect", f"{int(stu_active['Incorrect'].max())}")
                r3c2.metric("🛡️ Lowest Incorrect", f"{int(stu_active['Incorrect'].min())}")
                r3c3.metric("🎯 Best Accuracy", f"{stu_active['Accuracy'].max():.1f}%")
                r3c4.metric("📉 Worst Accuracy", f"{stu_active['Accuracy'].min():.1f}%")

            st.divider()
            render_predictor(stu_df['Attempts'].mean(), stu_df['Accuracy'].mean(), stu_df['Score'].mean())
            st.divider()
            render_growth_charts(stu_df, df)

# --- TAB 3: COMPARE ---
with tab_compare:
    st.markdown("### ⚔️ Compare Performance")
    if df.empty:
        st.warning("No data.")
    else:
        all_students = sorted(df['Name'].unique().tolist())
        selected_students_compare = st.multiselect("Select Students to Compare", all_students)
        
        if selected_students_compare:
            comp_df = df[df['Name'].isin(selected_students_compare)].copy().sort_values('Test_Number')
            contrast_colors = ['#FF0000', '#32CD32', '#1E90FF', '#FF8C00', '#9400D3', '#FF1493', '#00CED1']
            
            st.subheader(f"⚔️ Comparison Charts")
            
            fig_score = px.line(comp_df, x='Test_ID', y='Score', color='Name', markers=True, title="🏆 Score Comparison", color_discrete_sequence=contrast_colors)
            st.plotly_chart(fig_score, use_container_width=True)
            
            fig_acc = px.line(comp_df, x='Test_ID', y='Accuracy', color='Name', markers=True, title="🎯 Accuracy Comparison", color_discrete_sequence=contrast_colors)
            st.plotly_chart(fig_acc, use_container_width=True)

            fig_att = px.line(comp_df, x='Test_ID', y='Attempts', color='Name', markers=True, title="📝 Attempts Comparison", color_discrete_sequence=contrast_colors)
            st.plotly_chart(fig_att, use_container_width=True)

            fig_rank = px.line(comp_df, x='Test_ID', y='Rank', color='Name', markers=True, title="🏅 Rank Comparison (Lower is Better)", color_discrete_sequence=contrast_colors)
            fig_rank.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_rank, use_container_width=True)

# --- TAB 4: TEST DRILL-DOWN ---
with tab_test_drill:
    if df.empty:
        st.warning("No data.")
    else:
        st.markdown("### 📅 Drill Down by Test")
        
        unique_tests = df[['Test_ID', 'Test_Number']].drop_duplicates().sort_values('Test_Number')
        test_list = unique_tests['Test_ID'].tolist()
        selected_test = st.selectbox("Select Test", test_list, index=len(test_list)-1)
        
        test_df = df[df['Test_ID'] == selected_test].copy()
        
        if not test_df.empty:
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Students Appeared", len(test_df))
            t2.metric("Test Avg Score", f"{test_df['Score'].mean():.2f}")
            t3.metric("Highest Score", test_df['Score'].max())
            t4.metric("Best Rank", 1) 
            
            st.divider()
            
            b1, b2 = st.columns(2)
            with b1:
                fig_hist = px.histogram(test_df, x="Score", nbins=30, title=f"Score Distribution - {selected_test}", color_discrete_sequence=['#4CAF50'])
                st.plotly_chart(fig_hist, use_container_width=True)
            with b2:
                batch_perf = test_df.groupby("Batch")["Score"].mean().reset_index()
                fig_bar = px.bar(batch_perf, x="Batch", y="Score", color="Batch", title="Avg Score by Batch", text_auto='.2f')
                st.plotly_chart(fig_bar, use_container_width=True)
            
            st.subheader(f"🏅 Leaderboard: {selected_test}")
            st.dataframe(
                test_df.sort_values("Rank")[['Rank', 'Name', 'Batch', 'Score', 'Correct', 'Incorrect', 'Accuracy']], 
                use_container_width=True, hide_index=True
            )