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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
import numpy as np

def clean_latex(latex_str):
    """Clean and prepare LaTeX string for rendering"""
    if not latex_str:
        return ""
    
    # Remove outer $$ if present
    latex_str = latex_str.strip()
    if latex_str.startswith('$$') and latex_str.endswith('$$'):
        latex_str = latex_str[2:-2]
    elif latex_str.startswith('$') and latex_str.endswith('$'):
        latex_str = latex_str[1:-1]
    
    # Clean up common LaTeX issues
    latex_str = latex_str.replace('\\text{', '\\mathrm{')
    latex_str = latex_str.replace('\\displaystyle', '')
    
    return latex_str.strip()

def latex_to_png(latex_str, dpi=300, fontsize=14):
    """Convert LaTeX string to PNG image with proper mathematical rendering"""
    try:
        latex_str = clean_latex(latex_str)
        
        if not latex_str:
            return create_text_image("", fontsize)
        
        # Set up matplotlib for LaTeX rendering
        plt.rcParams['text.usetex'] = False  # Use mathtext instead of full LaTeX
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['mathtext.fontset'] = 'cm'
        
        fig, ax = plt.subplots(figsize=(12, 2))
        ax.axis('off')
        
        # Try to render as math text
        try:
            ax.text(0.5, 0.5, f'${latex_str}$', 
                   fontsize=fontsize, 
                   ha='center', 
                   va='center',
                   transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        except Exception:
            # If math rendering fails, try as regular text
            ax.text(0.5, 0.5, latex_str, 
                   fontsize=fontsize, 
                   ha='center', 
                   va='center',
                   transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Save to BytesIO
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', transparent=False)
        plt.close(fig)
        
        img_buf.seek(0)
        return img_buf
        
    except Exception as e:
        print(f"Error rendering LaTeX '{latex_str}': {e}")
        return create_text_image(latex_str, fontsize)

def create_text_image(text, fontsize=14):
    """Create a simple text image as fallback"""
    fig, ax = plt.subplots(figsize=(10, 1))
    ax.axis('off')
    ax.text(0.5, 0.5, text, fontsize=fontsize, ha='center', va='center',
           transform=ax.transAxes)
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close(fig)
    
    img_buf.seek(0)
    return img_buf

def generate_pdf(mcqs, output_path):
    """Generate PDF with properly rendered mathematical expressions"""
    try:
        print(f"🚀 Starting PDF generation with {len(mcqs)} MCQs")
        
        doc = SimpleDocTemplate(output_path, pagesize=letter, 
                               topMargin=1*inch, bottomMargin=1*inch,
                               leftMargin=1*inch, rightMargin=1*inch)
        styles = getSampleStyleSheet()
        story = []

        # Custom styles
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=1,  # Center alignment
            textColor=colors.darkblue
        )
        
        question_number_style = ParagraphStyle(
            'QuestionNumberStyle',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=10,
            textColor=colors.darkblue,
            fontName='Helvetica-Bold'
        )
        
        option_style = ParagraphStyle(
            'OptionStyle',
            parent=styles['Normal'],
            fontSize=10,
            leftIndent=30,
            spaceAfter=8
        )

        # Title
        story.append(Paragraph("JEE MCQ Questions", title_style))
        story.append(Spacer(1, 20))

        for i, mcq in enumerate(mcqs, 1):
            print(f"📝 Processing question {i}/{len(mcqs)}")
            
            # Question number
            story.append(Paragraph(f"Question {i}:", question_number_style))
            
            # Question with LaTeX rendering
            try:
                question_latex = mcq.get('question', '')
                print(f"  Question LaTeX: {question_latex[:50]}...")
                
                if question_latex:
                    question_img_buf = latex_to_png(question_latex, dpi=300, fontsize=14)
                    question_img = Image(question_img_buf, width=6*inch, height=0.8*inch)
                    story.append(question_img)
                else:
                    story.append(Paragraph("Question text missing", styles['Normal']))
                    print("  ⚠️ Question text missing")
                    
            except Exception as e:
                print(f"  ❌ Error rendering question: {e}")
                story.append(Paragraph(f"Error rendering question: {question_latex}", styles['Normal']))
            
            story.append(Spacer(1, 15))

            # Options
            options = mcq.get('options', [])
            labels = ['A', 'B', 'C', 'D']
            print(f"  Processing {len(options)} options")
            
            for j, option in enumerate(options[:4]):  # Ensure max 4 options
                if j < len(labels):
                    try:
                        # Option label
                        story.append(Paragraph(f"{labels[j]})", option_style))
                        
                        # Option with LaTeX rendering
                        if option:
                            print(f"    Option {labels[j]}: {option[:30]}...")
                            option_img_buf = latex_to_png(option, dpi=300, fontsize=12)
                            option_img = Image(option_img_buf, width=5*inch, height=0.6*inch)
                            story.append(option_img)
                        else:
                            story.append(Paragraph("Option text missing", styles['Normal']))
                            print(f"    ⚠️ Option {labels[j]} text missing")
                        
                        story.append(Spacer(1, 8))
                    except Exception as e:
                        print(f"    ❌ Error rendering option {labels[j]}: {e}")
                        story.append(Paragraph(f"{labels[j]}) Error rendering option: {option}", option_style))
                        story.append(Spacer(1, 8))

            # Answer (optional - you can remove this if you don't want answers in the PDF)
            answer = mcq.get('answer', '')
            if answer:
                answer_style = ParagraphStyle(
                    'AnswerStyle',
                    parent=styles['Normal'],
                    fontSize=10,
                    textColor=colors.darkgreen,
                    fontName='Helvetica-Bold'
                )
                story.append(Paragraph(f"Answer: {answer}", answer_style))
            
            story.append(Spacer(1, 25))
            
            # Add page break every 3 questions for better formatting
            if i % 3 == 0 and i < len(mcqs):
                story.append(PageBreak())

        # Build PDF
        print("🔨 Building PDF...")
        doc.build(story)
        print("✅ PDF generated successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        import traceback
        traceback.print_exc()
        return False

# Utility: extract and clean MCQs from Claude's output
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

# Streamlit App
st.title("JEE MCQ Generator with LaTeX Rendering")
st.markdown("Generate MCQs from PDF images with proper mathematical symbol rendering")

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
    st.success("PDF uploaded successfully.")

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
                
        doc.close()
        
    except Exception as e:
        st.error(f"Error converting PDF to images: {e}")
        st.stop()
    
    if not image_contents:
        st.error("No images could be extracted from the PDF.")
        st.stop()

    CLAUDE_API_KEY = os.getenv("claude_api")
    if not CLAUDE_API_KEY:
        st.error("Claude API key not found. Please set the 'claude_api' environment variable.")
        st.stop()
        
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

    # Initialize session state for MCQs
    if 'mcqs_generated' not in st.session_state:
        st.session_state.mcqs_generated = False
        st.session_state.mcqs_data = []

    # Generate MCQs button
    if st.button("Generate MCQs"):
        # Progress bar and info placeholder
        progress_bar = st.progress(0)
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
                "'question' and each element of 'options' must be valid LaTeX strings for mathematical expressions, wrapped in double dollar signs ($ ... $). "
                "'answer' is the correct option as a string (A, B, C, or D). "
                "Ensure all mathematical symbols, fractions, integrals, derivatives, etc. are properly formatted in LaTeX. "
                "Do not include any explanations. Output only the JSON array. "
                "Do not repeat any question from previous batches. Generate new and different questions each time. "
                f"Here are some questions already generated so far (do NOT repeat these):\n{prev_questions_text}"
            )
            
            data = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 2048,
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        *batch_images
                    ]}
                ]
            }
            
            try:
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

                # Extract, clean, and convert to LaTeX
                mcqs_json = extract_and_save_mcqs("claude_mcqs.json", "claude_mcqs_clean.json")
                if isinstance(mcqs_json, list):
                    all_mcqs.extend(mcqs_json)
                    all_mcqs = dedup_mcqs(all_mcqs)
                else:
                    st.warning("Claude did not return a valid MCQ list in this batch.")

                # Show progress after each batch
                percent = min(100, int(100 * len(all_mcqs) / total_mcqs))
                progress_bar.progress(percent)
                progress_info.info(f"Progress: {min(len(all_mcqs), total_mcqs)} / {total_mcqs} MCQs generated ({percent}%)")

                if len(all_mcqs) >= total_mcqs:
                    break
                    
            except Exception as e:
                st.error(f"Error in batch {batch + 1}: {e}")
                continue

        # Save all MCQs to final file
        final_mcqs = all_mcqs[:total_mcqs]
        with open(final_mcqs_path, "w", encoding="utf-8") as f:
            json.dump(final_mcqs, f, ensure_ascii=False, indent=2)
            
        # Update session state
        st.session_state.mcqs_generated = True
        st.session_state.mcqs_data = final_mcqs

        st.success(f"Generated {len(final_mcqs)} MCQs successfully!")
        st.rerun()  # Refresh the page to show the new buttons

# Check if MCQs exist (either in session state or file) and show download options
if st.session_state.mcqs_generated or os.path.exists(final_mcqs_path):
    st.subheader("Download Options")
    
    # Load MCQs if not in session state
    if not st.session_state.mcqs_generated and os.path.exists(final_mcqs_path):
        try:
            with open(final_mcqs_path, 'r', encoding='utf-8') as f:
                st.session_state.mcqs_data = json.load(f)
                st.session_state.mcqs_generated = True
        except Exception as e:
            st.error(f"Error loading MCQs: {e}")
    
    # Show current status
    if st.session_state.mcqs_data:
        st.info(f"📊 {len(st.session_state.mcqs_data)} MCQs ready for download")
    
    # Generate and download PDF
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Generate PDF with Rendered Math", help="Click to generate PDF with mathematical symbols"):
            if not st.session_state.mcqs_data:
                st.error("No MCQs found. Please generate MCQs first.")
            else:
                try:
                    pdf_path = "mcqs_with_math.pdf"
                    
                    # Show debug info
                    st.info(f"📝 Processing {len(st.session_state.mcqs_data)} MCQs...")
                    
                    # Debug: Show first MCQ
                    if st.session_state.mcqs_data:
                        with st.expander("🔍 Debug - First MCQ"):
                            st.json(st.session_state.mcqs_data[0])
                    
                    with st.spinner("Generating PDF with mathematical rendering..."):
                        success = generate_pdf(st.session_state.mcqs_data, pdf_path)
                    
                    if success and os.path.exists(pdf_path):
                        st.success("✅ PDF generated successfully!")
                        
                        # Show file size
                        file_size = os.path.getsize(pdf_path)
                        st.info(f"📄 PDF size: {file_size / 1024:.1f} KB")
                        
                        # Create download button for PDF
                        with open(pdf_path, "rb") as f:
                            pdf_bytes = f.read()
                            st.download_button(
                                label="📄 Download MCQs PDF",
                                data=pdf_bytes,
                                file_name="jee_mcqs_with_math.pdf",
                                mime="application/pdf",
                                key="download_pdf"
                            )
                    else:
                        st.error("❌ Error generating PDF. Check the logs above.")
                        
                except Exception as e:
                    st.error(f"❌ Error generating PDF: {str(e)}")
                    st.exception(e)  # Show full traceback for debugging

    with col2:
        # Download button for JSON
        if st.session_state.mcqs_data:
            json_data = json.dumps(st.session_state.mcqs_data, ensure_ascii=False, indent=2)
            st.download_button(
                label="📋 Download MCQs JSON",
                data=json_data,
                file_name="jee_mcqs.json",
                mime="application/json",
                key="download_json"
            )

    # Preview some MCQs
    st.subheader("Preview Generated MCQs")
    if st.session_state.mcqs_data:
        # Show first 2 MCQs as preview
        for i, mcq in enumerate(st.session_state.mcqs_data[:2], 1):
            with st.expander(f"Preview Question {i}"):
                st.write("**Question:**", mcq.get('question', 'N/A'))
                st.write("**Options:**")
                options = mcq.get('options', [])
                for j, option in enumerate(options):
                    st.write(f"  {chr(65+j)}) {option}")
                st.write("**Answer:**", mcq.get('answer', 'N/A'))
    else:
        st.info("No MCQs to preview. Generate MCQs first.")
