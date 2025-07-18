import re
import streamlit as st
import os
import requests
import json
import base64
import fitz  # PyMuPDF
from io import BytesIO
from dotenv import load_dotenv
import random
load_dotenv()
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib.pyplot as plt
import io

def latex_to_png(latex_str, dpi=300):
    # Remove $$ from start and end if present
    latex_str = latex_str.strip('$')
    
    plt.figure(figsize=(10, 0.5))
    plt.axis('off')
    plt.text(0.5, 0.5, latex_str, size=12, ha='center', va='center')
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=dpi, bbox_inches='tight', transparent=True)
    plt.close()
    
    img_buf.seek(0)
    return img_buf

def generate_pdf(mcqs, output_path):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Custom style for questions
    question_style = ParagraphStyle(
        'QuestionStyle',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=20
    )
    
    # Custom style for options
    option_style = ParagraphStyle(
        'OptionStyle',
        parent=styles['Normal'],
        fontSize=10,
        leftIndent=20,
        spaceAfter=10
    )

    for i, mcq in enumerate(mcqs, 1):
        # Question
        question_latex = mcq['question']
        question_img = latex_to_png(question_latex)
        img = Image(question_img, width=400, height=30)
        story.append(img)
        story.append(Spacer(1, 12))

        # Options
        options = mcq['options']
        labels = ['A', 'B', 'C', 'D']
        for j, (label, option) in enumerate(zip(labels, options)):
            option_latex = option
            option_img = latex_to_png(option_latex)
            opt_img = Image(option_img, width=200, height=20)
            story.append(Paragraph(f"{label})", option_style))
            story.append(opt_img)
            story.append(Spacer(1, 10))

        story.append(Spacer(1, 20))

    doc.build(story)



# Utility: extract and clean MCQs from Claude's output (no KaTeX conversion)
def extract_and_save_mcqs(raw_json_path, clean_json_path):
    with open(raw_json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    # If the file is a list with a single dict with 'text', extract from there
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], dict) and 'text' in raw[0]:
        text = raw[0]['text']
        # Try to extract the JSON array from the first [ to the last ]
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end+1]
        else:
            json_str = text
        try:
            mcqs = json.loads(json_str)
        except Exception as e:
            mcqs = []
            print(f"Error parsing MCQ JSON: {e}\nExtracted string: {json_str[:200]}...")
    else:
        mcqs = raw
    with open(clean_json_path, 'w', encoding='utf-8') as f:
        json.dump(mcqs, f, ensure_ascii=False, indent=2)
    return mcqs

st.title("JEE MCQ Generator")


# User input for total number of MCQs desired
total_mcqs = st.number_input(
    "How many MCQs do you want to generate in total?",
    min_value=1, max_value=100, value=20, step=2
)
# Batch size for each Claude call
BATCH_SIZE = 3

uploaded_file = st.file_uploader("Upload a PDF with MCQ questions", type=["pdf"])

if uploaded_file:

    # Reset final_mcqs.json at the start of a new session
    final_mcqs_path = "final_mcqs.json"
    if os.path.exists(final_mcqs_path):
        os.remove(final_mcqs_path)
    st.success("PDF uploaded.")

    pdf_bytes = uploaded_file.read()
    image_contents = []
    output_dir = "extracted_images"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for i in range(len(doc)):
            try:
                page = doc.load_page(i)
                pix = page.get_pixmap(dpi=200)
                img_bytes = pix.tobytes("png")
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                image_contents.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": img_b64}
                })

                # Save image to directory
                img_path = os.path.join(output_dir, f"page_{i+1}.png")
                pix.save(img_path)

            except Exception as e:
                st.warning(f"Skipping page {i+1} due to error: {e}")
                continue
    except Exception as e:
        st.error(f"Error converting PDF to images: {e}")
        st.stop()
    if not image_contents:
        st.error("No images could be extracted from the PDF.")
        st.stop()

    CLAUDE_API_KEY = os.getenv("claude_api")
    CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }

    # Load existing MCQs if present
    all_mcqs = []

    # Helper to deduplicate by question text
    def dedup_mcqs(mcq_list):
        seen = set()
        unique = []
        for mcq in mcq_list:
            q = mcq.get('question', '').strip()
            if q and q not in seen:
                seen.add(q)
                unique.append(mcq)
        return unique

    # Progress bar and info placeholder
    progress_bar = st.empty()
    progress_info = st.empty()

    num_batches = (total_mcqs + BATCH_SIZE - 1) // BATCH_SIZE
    for batch in range(num_batches):
        batch_size = min(BATCH_SIZE, total_mcqs - len(all_mcqs))
        # Randomly select 2 images for this batch
        if len(image_contents) > 2:
            batch_images = random.sample(image_contents, 2)
        else:
            batch_images = image_contents
        # Collect only the last 10 previous questions for prompt
        prev_questions = [mcq.get('question', '').strip() for mcq in all_mcqs if mcq.get('question')]
        prev_questions_limited = prev_questions[-10:]
        prev_questions_text = "\n".join(prev_questions_limited)
        prompt = (
            f"Extract {batch_size} unique MCQ questions from these images. "
            f"Return the result as a JSON array of {batch_size} objects. "
            "Each object must have exactly these keys: 'question', 'options', and 'answer'. "
            "'question' and each element of 'options' must be valid KaTeX LaTeX strings, wrapped in double dollar signs ($$ ... $$) for math rendering. "
            "'answer' is the correct option as a string. "
            "Do not include any explanations. Output only the JSON array. "
            "Do not repeat any question from previous batches. Generate new and different questions each time. "
            f"Here are some questions already generated so far (do NOT repeat these):\n{prev_questions_text}"
        )
        data = {
            "model": "claude-3-7-sonnet-20250219",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    *batch_images
                ]}
            ]
        }
        response = requests.post(CLAUDE_API_URL, headers=headers, json=data)
        if response.status_code != 200:
            st.error(f"Claude API error: {response.text}")
            st.stop()
        result = response.json()
        content = result.get("content", "")
        
        # Save raw Claude output for reference
        with open("claude_mcqs.json", "w", encoding="utf-8") as f:
            if isinstance(content, list):
                f.write(json.dumps(content, ensure_ascii=False, indent=2))
            else:
                f.write(json.dumps([{"type": "text", "text": content}], ensure_ascii=False, indent=2))

        # Extract, clean, and convert to KaTeX LaTeX
        mcqs_json = extract_and_save_mcqs("claude_mcqs.json", "claude_mcqs_clean.json")
        if isinstance(mcqs_json, list):
            all_mcqs.extend(mcqs_json)
            all_mcqs = dedup_mcqs(all_mcqs)
        else:
            st.warning("Claude did not return a valid MCQ list in this batch.")

        # Show progress after each batch (update in-place)
        percent = int(100 * min(len(all_mcqs), total_mcqs) / total_mcqs)
        progress_bar.progress(percent)
        progress_info.info(f"Progress: {min(len(all_mcqs), total_mcqs)} / {total_mcqs} MCQs generated ({percent}%)")

        if len(all_mcqs) >= total_mcqs:
            break

    # Save all MCQs to final file (append mode, deduped)
    with open(final_mcqs_path, "w", encoding="utf-8") as f:
        json.dump(all_mcqs[:total_mcqs], f, ensure_ascii=False, indent=2)

    st.subheader("Generated MCQs")

    if st.button("Generate PDF"):
    try:
        pdf_path = "mcqs.pdf"
        with open(final_mcqs_path, 'r') as f:
            mcqs = json.load(f)
        
        generate_pdf(mcqs, pdf_path)
        
        # Create download button for PDF
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
            st.download_button(
                label="Download MCQs PDF",
                data=pdf_bytes,
                file_name="mcqs.pdf",
                mime="application/pdf"
            )
        st.success("PDF generated successfully!")
    except Exception as e:
        st.error(f"Error generating PDF: {e}")

    # Download button for final_mcqs.json
    with open(final_mcqs_path, "r", encoding="utf-8") as f:
        st.download_button(
            label="Download Final MCQs JSON",
            data=f.read(),
            file_name="final_mcqs.json",
            mime="application/json"
        )


