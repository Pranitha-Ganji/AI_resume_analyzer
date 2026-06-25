import streamlit as st
import pdfplumber
from google import genai
from google.genai import types

# ------------------------------------------------------------------
# 1. Page Configuration & Setup
# ------------------------------------------------------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("📄 AI-Powered Resume Analyzer & ATS Optimizer")
st.markdown("""
Upload your resume (PDF) and paste the target job description. 
Our advanced AI will analyze your profile, calculate an ATS match score, and provide tailored optimization feedback.
""")

# Sidebar configuration for secure API key injection
st.sidebar.header("🔑 Setup")
api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password", 
                               help="Get a free key from Google AI Studio")
st.sidebar.markdown("[Get a Free Gemini API Key](https://aistudio.google.com/)")

# ------------------------------------------------------------------
# 2. Robust Text Extraction Helper (Fixes Stream/EOF issues)
# ------------------------------------------------------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    # Open the file-like object securely using pdfplumber's layout engine
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# ------------------------------------------------------------------
# 3. Main Dashboard UI Layout
# ------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Step 1: Upload your Resume")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
with col2:
    st.subheader("🎯 Step 2: Target Job Description")
    job_description = st.text_area("Paste the job details here:", height=200, 
                                   placeholder="Paste the duties, requirements, and responsibilities...")

st.markdown("---")

# ------------------------------------------------------------------
# 4. Core Core Processing & AI Inference Logic
# ------------------------------------------------------------------
if st.button("🚀 Analyze My Resume", type="primary"):
    # Guardrails: Validate all user inputs are provided before invoking API
    if not api_key:
        st.error("Please enter your Gemini API Key in the sidebar to run the analysis.")
    elif not uploaded_file:
        st.error("Please upload your resume PDF.")
    elif not job_description.strip():
        st.error("Please paste a target job description.")
    else:
        with st.spinner("Analyzing your resume layout and content against the job requirements..."):
            try:
                # 1. Extract the text safely via pdfplumber
                resume_text = extract_text_from_pdf(uploaded_file)
                
                # Check if the extracted text is empty (e.g., scanned image PDF)
                if not resume_text.strip():
                    st.error("Could not extract text from this PDF. Please ensure it is not a scanned image/photo.")
                    st.stop()
                
                # 2. Instantiate the modern Google GenAI SDK Client
                client = genai.Client(api_key=api_key)
                
                # 3. Craft the evaluation system prompt
                analysis_prompt = f"""
                You are an expert technical recruiter and an advanced Applicant Tracking System (ATS) optimization engine.
                Analyze the following Resume against the provided Job Description.
                
                Provide a highly structured, objective report with the following exact markdown layout:
                
                ## 📊 ATS Match Score
                Provide a single bold percentage match rating from 0% to 100% based on skills, experience, and keywords matching.
                
                ## 🔍 Key Missing Keywords & Skills
                - Bullet points listing essential hard/soft skills or domain terminology present in the job description but absent/weak in the resume.
                
                ## 💡 Tailored Optimization Suggestions
                - Bullet points explaining exactly how the user can modify their summary, bullet points, or layout to better match this job. Be highly actionable.
                
                ---
                ### Resume Text:
                {resume_text}
                
                ### Job Description:
                {job_description}
                """
                
                # 4. Generate the structured evaluation response using Gemini 2.5 Flash
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=analysis_prompt,
                )
                
                # 5. Render results cleanly to the dashboard UI
                st.success("Analysis Complete!")
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")