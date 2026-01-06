import streamlit as st
import pdfplumber
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="UPSC Test Analyzer Pro", 
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
DEFAULT_TOTAL_QUESTIONS = 60  # Fixed to 60 as requested
MARKS_CORRECT = 2.0
MARKS_INCORRECT = 0.66

# --- HELPER FUNCTIONS ---
def clean_batch_name(text):
    if not isinstance(text, str):
        return "Unknown"
    match = re.search(r'FRC[\s-]*(\d+)', text, re.IGNORECASE)
    if match:
        return f"FRC-{match.group(1)}"
    return "Others"

def extract_test_number(filename):
    match = re.search(r'Test\s*(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match_generic = re.search(r'(\d+)', filename)
    if match_generic:
        return int(match_generic.group(1))
    return 0

def clean_student_name(text):
    """
    Aggressively cleans name to avoid duplicates.
    Removes digits (like merged ranks), special chars, and extra whitespace.
    """
    if not isinstance(text, str): return ""
    # Remove digits (often rank gets merged into name)
    text = re.sub(r'\d+', '', text)
    # Remove common artifacts
    text = text.replace('.', '').replace('\n', ' ')
    # Normalize spaces
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
                        # Basic cleaning of the row
                        clean_row = [str(x).replace('\n', ' ').strip() if x else "" for x in row]
                        
                        # Skip empty or header rows
                        if not clean_row or "Name" in clean_row[0] or "Sl No" in clean_row[0]:
                            continue

                        try:
                            # --- SMART PARSING LOGIC ---
                            
                            # CASE 1: STANDARD FORMAT (Most Tests)
                            # Row[0] is strictly digits (Rank/Sl No)
                            if clean_row[0].isdigit():
                                raw_name = clean_row[1]
                                raw_batch_roll = clean_row[2]
                                # Data starts from col 3 usually
                                data_start_idx = 3
                                
                                # Sometimes Batch is merged into Col 2, sometimes distinct
                                batch = clean_batch_name(raw_batch_roll)
                                if batch == "Others" and len(clean_row) > 3:
                                    batch = clean_batch_name(clean_row[3])
                            
                            # CASE 2: MERGED FORMAT (Test 1 style)
                            # Row[0] contains text (Name merged with Rank)
                            else:
                                raw_name = clean_row[0]
                                raw_batch_roll = clean_row[1]
                                # Data starts from col 2 usually
                                data_start_idx = 2
                                batch = clean_batch_name(raw_batch_roll)

                            # --- Extract Numbers (Correct, Incorrect, Score) ---
                            numerics = []
                            # Scan only from the data_start_idx to avoid picking up numbers in Name/Batch
                            for cell in clean_row[data_start_idx:]:
                                try:
                                    val = float(cell)
                                    numerics.append(val)
                                except:
                                    continue
                            
                            # We need at least 3 numbers: Correct, Incorrect, Score
                            if len(numerics) >= 3:
                                correct = numerics[0]
                                incorrect = numerics[1]
                                score = numerics[2]
                                
                                # Rank is usually the 4th number, or we ignore it if missing
                                rank = int(numerics[3]) if len(numerics) >= 4 else 0

                                # --- FINAL VALIDATION & CLEANING ---
                                name = clean_student_name(raw_name)
                                
                                # Ignore invalid rows (e.g. headers read as text)
                                if len(name) < 2: continue
                                if correct > DEFAULT_TOTAL_QUESTIONS + 5: continue # Sanity check

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
                        except Exception:
                            continue

            logs.append(f"✅ Loaded {filename}")
        except Exception as e:
            logs.append(f"❌ Error reading {filename}: {str(e)}")

    df = pd.DataFrame(all_records)
    
    if not df.empty:
        valid_batches = ['FRC-8', 'FRC-9', 'FRC-10']
        df = df[df['Batch'].isin(valid_batches)]
        df = df.sort_values('Test_Number')
        
        # --- DEDUPLICATION SAFETY NET ---
        # If the same student appears twice for the SAME Test_ID, keep the one with the higher score (or first)
        df = df.drop_duplicates(subset=['Test_ID', 'Name'], keep='first')

    return df, logs

# --- CHARTS ---
def render_growth_charts(student_df, full_df):
    student_df = student_df.sort_values('Test_Number')
    
    # 1. Attempts
    fig_att = px.line(student_df, x='Test_ID', y='Attempts', markers=True, 
                      title="📈 Trend: Number of Attempts", line_shape='spline')
    fig_att.update_traces(line_color='#FFA500')
    st.plotly_chart(fig_att, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        # 2. Correct
        fig_corr = px.line(student_df, x='Test_ID', y='Correct', markers=True, 
                           title="✅ Trend: Correct Answers")
        fig_corr.update_traces(line_color='#00CC96')
        st.plotly_chart(fig_corr, use_container_width=True)
    with c2:
        # 3. Incorrect
        fig_inc = px.line(student_df, x='Test_ID', y='Incorrect', markers=True, 
                          title="❌ Trend: Incorrect Answers")
        fig_inc.update_traces(line_color='#EF553B')
        st.plotly_chart(fig_inc, use_container_width=True)

    # 4. Score
    fig_score = px.line(student_df, x='Test_ID', y='Score', markers=True, 
                        title="🏆 Trend: Total Score", line_shape='spline')
    fig_score.add_hline(y=student_df['Score'].mean(), line_dash="dash", annotation_text="Your Avg")
    st.plotly_chart(fig_score, use_container_width=True)

    # 5. Rank
    fig_rank = px.line(student_df, x='Test_ID', y='Rank', markers=True, 
                       title="🏅 Trend: Rank (Lower is Better)")
    fig_rank.update_yaxes(autorange="reversed") 
    st.plotly_chart(fig_rank, use_container_width=True)

    # 6. Comparison (All Lines)
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
    fig_comp.add_trace(go.Scatter(x=comp_df['Test_ID'], y=comp_df['My Score'], name='My Score', 
                                  line=dict(color='#636EFA', width=4), mode='lines+markers'))
    fig_comp.add_trace(go.Scatter(x=comp_df['Test_ID'], y=comp_df['Batch Avg'], name='Batch Avg', 
                                  line=dict(color='orange', width=2, dash='dash')))
    fig_comp.add_trace(go.Scatter(x=comp_df['Test_ID'], y=comp_df['Topper Score'], name='Topper Score', 
                                  line=dict(color='green', width=2, dash='dot')))
    
    fig_comp.update_layout(title="📊 Performance vs Others", hovermode="x unified")
    st.plotly_chart(fig_comp, use_container_width=True)

def render_multi_student_comparison(student_names, full_df):
    if not student_names:
        st.warning("Select students to compare.")
        return

    comp_df = full_df[full_df['Name'].isin(student_names)].copy().sort_values('Test_Number')

    # HIGH CONTRAST COLORS (Red, Green, Blue, Orange, Purple)
    # This ensures "blue and light blue" confusion doesn't happen
    contrast_colors = ['#FF0000', '#32CD32', '#1E90FF', '#FF8C00', '#9400D3', '#FF1493', '#00CED1']
    
    st.subheader(f"⚔️ Comparison")

    # Score
    fig_score = px.line(comp_df, x='Test_ID', y='Score', color='Name', markers=True,
                        title="🏆 Score Comparison", color_discrete_sequence=contrast_colors)
    st.plotly_chart(fig_score, use_container_width=True)

    # Accuracy
    fig_acc = px.line(comp_df, x='Test_ID', y='Accuracy', color='Name', markers=True,
                      title="🎯 Accuracy Comparison", color_discrete_sequence=contrast_colors)
    st.plotly_chart(fig_acc, use_container_width=True)
    
    # Attempts
    fig_att = px.line(comp_df, x='Test_ID', y='Attempts', color='Name', markers=True,
                      title="📝 Attempts Comparison", color_discrete_sequence=contrast_colors)
    st.plotly_chart(fig_att, use_container_width=True)

    # Rank
    fig_rank = px.line(comp_df, x='Test_ID', y='Rank', color='Name', markers=True,
                       title="🏅 Rank Comparison (Lower is Better)", color_discrete_sequence=contrast_colors)
    fig_rank.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_rank, use_container_width=True)

def render_predictor(avg_attempts, avg_acc, avg_score):
    st.markdown("#### 🤖 Scenario Planner: Adjust Attempts & Accuracy")
    st.caption("Adjust sliders to see predicted score. (Based on 60 Questions)")
    
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
    default_user = "SAURAV SINGH"
    user_name = st.text_input("👤 Your Name", value=default_user).upper()
    
    if not df.empty:
        unique_tests = df[['Test_ID', 'Test_Number']].drop_duplicates().sort_values('Test_Number')
        test_options = ["All Tests (Aggregate)"] + unique_tests['Test_ID'].tolist()
        selected_option = st.selectbox("📂 Select Data View", test_options)
        
        is_aggregate = (selected_option == "All Tests (Aggregate)")
        current_df = df.copy() if is_aggregate else df[df['Test_ID'] == selected_option].copy()
    else:
        st.warning("No Data Found")
        st.stop()
    
    with st.expander("System Logs"):
        for log in logs: st.caption(log)

st.title(f"📊 UPSC Test Analytics: {selected_option}")

# UPDATED TABS ORDER (Personal First)
tab_personal, tab_compare, tab_batch, tab_student = st.tabs([
    "🚀 My Analysis", 
    "⚔️ Compare Students",
    "🏢 Batch Analysis", 
    "🔍 Student Deep Dive"
])

# --- TAB 1: MY ANALYSIS (Priority) ---
with tab_personal:
    me_df = df[df['Name'].str.contains(user_name, na=False)].copy()
    
    if me_df.empty:
        st.error(f"User '{user_name}' not found.")
    else:
        # TOP STATS ROW
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tests Taken", len(me_df))
        m2.metric("Best Rank", me_df['Rank'].min())
        m3.metric("Best Score", me_df['Score'].max())
        m4.metric("Avg Score", round(me_df['Score'].mean(), 2))
        
        st.divider()
        
        # PREDICTOR
        render_predictor(me_df['Attempts'].mean(), me_df['Accuracy'].mean(), me_df['Score'].mean())
        
        st.divider()
        st.subheader("🚀 Your Growth Journey")
        render_growth_charts(me_df, df)

# --- TAB 2: COMPARE STUDENTS ---
with tab_compare:
    st.markdown("### ⚔️ Compare Performance")
    all_students = sorted(df['Name'].unique().tolist())
    
    # Default to User
    default_selections = []
    match = next((s for s in all_students if user_name in s), None)
    if match: default_selections.append(match)
        
    selected_students_compare = st.multiselect("Select Students", all_students, default=default_selections)
    render_multi_student_comparison(selected_students_compare, df)

# --- TAB 3: BATCH ANALYSIS ---
with tab_batch:
    # KPI
    col1, col2, col3, col4 = st.columns(4)
    if is_aggregate:
        col1.metric("Unique Students", current_df['Name'].nunique())
        col2.metric("Overall Avg Score", round(current_df['Score'].mean(), 2))
        col3.metric("Avg Accuracy", f"{round(current_df['Accuracy'].mean(), 1)}%")
        col4.metric("Highest Score", current_df['Score'].max())
    else:
        col1.metric("Total Students", len(current_df))
        col2.metric("Avg Score", round(current_df['Score'].mean(), 2))
        col3.metric("Avg Accuracy", f"{round(current_df['Accuracy'].mean(), 1)}%")
        col4.metric("Highest Score", current_df['Score'].max())

    st.divider()
    
    b1, b2 = st.columns(2)
    with b1:
        st.subheader("Score Distribution")
        fig_hist = px.histogram(current_df, x="Score", nbins=30, color_discrete_sequence=['#4CAF50'])
        st.plotly_chart(fig_hist, use_container_width=True)
    with b2:
        st.subheader("Batch Performance")
        batch_perf = current_df.groupby("Batch")["Score"].mean().reset_index()
        fig_bar = px.bar(batch_perf, x="Batch", y="Score", color="Batch", text_auto='.2f')
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("🏆 Leaderboard")
    if is_aggregate:
        lb = current_df.groupby(['Name', 'Batch']).agg({'Score':'mean', 'Accuracy':'mean', 'Test_ID':'count'}).reset_index()
        lb = lb.sort_values('Score', ascending=False).head(15)
        st.dataframe(lb.style.format({'Score':'{:.2f}', 'Accuracy':'{:.1f}%'}).background_gradient(subset=['Score'], cmap='Greens'), use_container_width=True)
    else:
        st.dataframe(current_df.sort_values("Rank").head(15)[['Rank','Name','Batch','Score','Correct','Incorrect']], use_container_width=True)

# --- TAB 4: STUDENT DEEP DIVE ---
with tab_student:
    st.markdown("### Find a Student")
    s_list = sorted(df['Name'].unique())
    sel_stu = st.selectbox("Select Name", s_list)
    stu_df = df[df['Name'] == sel_stu].copy()
    
    if not stu_df.empty:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Tests Taken", len(stu_df))
        k2.metric("Avg Score", round(stu_df['Score'].mean(), 2))
        k3.metric("Avg Accuracy", f"{round(stu_df['Accuracy'].mean(), 1)}%")
        k4.metric("Best Rank", stu_df['Rank'].min())
        
        st.divider()
        # Add Predictor here too
        render_predictor(stu_df['Attempts'].mean(), stu_df['Accuracy'].mean(), stu_df['Score'].mean())
        
        st.divider()
        render_growth_charts(stu_df, df)