import streamlit as st
from typing import Dict, Any
import json
from datetime import datetime
import pytz
from pytz import timezone
from anthropic import Anthropic
import pdfplumber

# Streamlit page configuration
st.set_page_config(page_title="Document Q&A with LaTeX", layout="wide", initial_sidebar_state="auto")

# Initialize Claude client
claude_client = Anthropic(api_key= os.getenv("claude_api"))

# UI elements
st.title("Question Paper Generator")
file = st.file_uploader("Upload a PDF file", type=["pdf"])
content = st.text_area("Content", key="content", help="Enter content if no PDF is uploaded")
prompt = st.text_area("Prompt", key="prompt", help="Enter the prompt for generating the question paper")

# JSON structure for question paper
json_structure = {
    "name": "name",
    "course": "course",
    "start_date": "start date",
    "end_date": "end date",
    "total_time": "total time",
    "total_marks": "total marks",
    "lessons": ["lessons"],
    "questions": [
        {
            "section_name": "section name",
            "description": "description",
            "questions": [
                {
                    "question_id": "question id",
                    "question_latex": "question latex",
                    "answer_latex": "answer latex",
                    "marks": "marks"
                }
            ]
        }
    ]
}

def _call_claude_api(assessment_data: Dict[str, Any]) -> str:
    """Call Claude API to generate HTML from assessment data"""
    try:
        # Format the start date
        start_date_raw = assessment_data.get("start_date")
        if start_date_raw:
            try:
                parsed_date = datetime.fromisoformat(start_date_raw.replace('Z', '+00:00'))
                ist_date = parsed_date.replace(tzinfo=pytz.utc).astimezone(timezone('Asia/Kolkata'))
                start_date = ist_date.strftime("%d %B %Y, %I:%M %p")
            except:
                start_date = start_date_raw
        else:
            start_date = datetime.now(timezone('Asia/Kolkata')).strftime("%d %B %Y, %I:%M %p")

        # Prepare the prompt for HTML generation
        html_prompt = f"""
        Convert the following assessment data into a complete, print-friendly HTML document optimized for light mode:
        {json.dumps(assessment_data, indent=2)}
        Requirements:
        - Create a complete HTML document with DOCTYPE, head, and body tags.
        - Use modern CSS for a professional, light mode appearance (white background, black text, high contrast).
        - Format all questions clearly with proper spacing and typography.
        - Use MathJax for mathematical notation.
        - Use Arial font, A4 page size, and 2cm margins.
        - Preserve line breaks from LaTeX content by converting \\n or \\ to <br> tags.
        - Include section headers with section name and total marks, followed by all questions with individual marks.
        - For multiple choice questions, list all options with letters (A, B, C, D).
        - For matching-type questions, use a table with two columns (Column I and Column II) and appropriate rows.
        - For short answer questions, display the question text clearly.
        - Ensure ALL questions in each section are included, iterating through the 'questions' array in each section.
        - Use sup and sub tags for superscript and subscript.
        - Return only the HTML document, no explanations.

        HTML Template:
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{assessment_data.get('name', 'Assessment')}</title>
            <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
            <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
            <style>
                body {{ font-family: 'Arial', sans-serif; margin: 20px; background: #ffffff; color: #000000; }}
                @page {{ size: A4; margin: 2cm; }}
                h1 {{ text-align: center; font-weight: bold; color: #000000; }}
                h2 {{ text-align: center; font-weight: bold; color: #000000; }}
                h4 {{ font-weight: bold; font-size: 12px; color: #000000; margin-bottom: 8px; }}
                .container {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 15px; }}
                .instructions {{ margin-top: 10px; font-size: 12px; color: #000000; background: #f8f9fa; padding: 10px; border-radius: 5px; }}
                .left {{ float: left; width: 48%; margin-right: 2%; }}
                .right {{ float: right; width: 48%; }}
                .question {{ margin-bottom: 20px; color: #000000; page-break-inside: avoid; }}
                .question-header {{ position: relative; width: 100%; min-height: 25px; margin-bottom: 10px; }}
                .question-number {{ position: absolute; left: 0; top: 0; font-weight: bold; font-size: 16px; color: #000000; }}
                .marks {{ position: absolute; right: 0; top: 0; font-size: 16px; font-weight: bold; color: #000000; }}
                .section-header {{ position: relative; margin: 20px 0 15px 0; border-bottom: 2px solid #000000; padding-bottom: 5px; width: 100%; height: 25px; }}
                .section-name {{ position: absolute; left: 0; top: 0; font-size: 18px; font-weight: bold; color: #000000; }}
                .section-marks {{ position: absolute; right: 0; top: 0; font-size: 16px; font-weight: bold; color: #000000; }}
                .question-content {{ margin-top: 10px; margin-bottom: 15px; color: #000000; white-space: pre-line; }}
                .options {{ margin-top: 10px; margin-left: 20px; line-height: 1.8; color: #000000; font-size: 14px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; color: #000000; }}
                td, th {{ border: 2px solid #000000; padding: 10px; text-align: left; color: #000000; }}
                th {{ background: #e6e6e6; font-weight: bold; color: #000000; }}
                p {{ color: #000000; line-height: 1.5; }}
                strong {{ color: #000000; font-weight: bold; }}
                @media print {{
                    body {{ margin: 0; background: #ffffff; }}
                    .page-break {{ page-break-before: always; }}
                    .instructions {{ background: #f8f9fa; }}
                }}
            </style>
        </head>
        <body>
            <div class="flex" style="justify-content: space-between; align-items: center; gap: 10px;">
                <div class="flex" style="flex-direction: column; align-items: center;">
                    <h1>{assessment_data.get('name', 'School Name')}</h1>
                    <h2>{assessment_data.get('course', 'Assessment')}</h2>
                </div>
            </div>
            <div class="container">
                <div class="left">
                    <h4>Time - {assessment_data.get('total_time', 'N/A')} minutes</h4>
                    <h4>Course - {assessment_data.get('course', 'Course Name')}</h4>
                </div>
                <div class="right">
                    <h4>Date - {start_date}</h4>
                    <h4>Max Marks - {assessment_data.get('total_marks', 'N/A')}</h4>
                </div>
            </div>
            <h4>Instructions</h4>
            <div class="instructions">
                <ol>
                    <li>This exam is scheduled for {start_date}</li>
                    <li>The total duration of the exam is {assessment_data.get('total_time', 'N/A')} minutes.</li>
                    <li>Read each question carefully before answering.</li>
                    <li>Attempt all questions. There is no negative marking for wrong answers.</li>
                    <li>Once the exam starts, the timer cannot be paused. Manage your time wisely.</li>
                    <li>Any form of malpractice or cheating will result in disqualification.</li>
                    <li>If you face any technical issues, contact the exam supervisor immediately.</li>
                    <li>Best of luck!</li>
                </ol>
            </div>
           <!-- Iterate over each section in assessment_data['questions'] -->
            <!-- For each section, create a div with class="section-header" containing the section name and total marks -->
            <!-- For each question in the section, create a div with class="question" containing the question number, LaTeX content, and marks -->
        </body>
        </html>
        """

        response = claude_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=4000,
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": html_prompt
                }
            ]
        )
        return response.content[0].text
    except Exception as e:
        raise Exception(f"Failed to generate HTML content: {str(e)}")
# Main application logic
if st.button("Generate Question Paper"):
    if not (file or content) or not prompt:
        st.error("Please upload a PDF file or enter content and provide a prompt.")
        st.stop()

    # Extract text from PDF if provided
    if file:
        try:
            with pdfplumber.open(file) as pdf:
                pdf_text = ""
                for page in pdf.pages:
                    pdf_text += page.extract_text() or ""
                content = pdf_text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            st.stop()

    # Prepare the user prompt for JSON generation
    user_prompt = f"""
    You are an expert question paper generator. Generate a question paper based on the provided content and prompt.
    The question paper should:
    - Be in the language and style of the prompt.
    - Follow the JSON structure: {json.dumps(json_structure, indent=2)}
    - Include LaTeX code for questions and answers in the specified JSON format.
    Content: {content}
    Prompt: {prompt}
    Return only the JSON output, no explanations.
    do not include ```json or ``` in the response.
    """

    try:
        # Generate JSON question paper
        response = claude_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=4000,
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )
        # Debug: Show the raw response
        st.markdown("#### Raw Claude Response")
        st.code(response.content[0].text)
        # Parse JSON response
        try:
            assessment_data = json.loads(response.content[0].text)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON response from Claude: {str(e)}")
            st.stop()

        # Validate JSON structure
        if not isinstance(assessment_data, dict) or "questions" not in assessment_data:
            st.error("Generated JSON data does not match the required structure.")
            st.stop()
        st.markdown(json.dumps(assessment_data, indent=2))
        html_content = _call_claude_api(assessment_data)

        # Display preview
        st.markdown("### Question Paper Preview")
        st.components.v1.html(html_content, height=600, scrolling=True)

        # Add download button
        st.download_button(
            label="Download Question Paper",
            data=html_content,
            file_name="question_paper.html",
            mime="text/html"
        )

    except Exception as e:
        st.error(f"Error generating question paper: {str(e)}")
