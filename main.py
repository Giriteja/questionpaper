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
from PIL import Image as PILImage

# Initialize the session state variable if it doesn't exist
if 'mcqs_generated' not in st.session_state:
    st.session_state.mcqs_generated = False

def generate_simple_pdf(mcqs, output_path, exam_type="", difficulty=""):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch

    try:
        doc = SimpleDocTemplate(output_path, pagesize=letter,
                                topMargin=1*inch, bottomMargin=1*inch,
                                leftMargin=1*inch, rightMargin=1*inch)

        styles = getSampleStyleSheet()
        story = []

        # Enhanced title with exam type and difficulty
        title_text = f"JEE {exam_type} MCQs - {difficulty} Level" if exam_type and difficulty else "JEE MCQs"
        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=16,
                                     alignment=1, spaceAfter=20, textColor=colors.darkblue)
        
        subtitle_style = ParagraphStyle('SubtitleStyle', parent=styles['Normal'], fontSize=12,
                                       alignment=1, spaceAfter=30, textColor=colors.darkgreen,
                                       fontName='Helvetica-Bold')
        
        question_style = ParagraphStyle('QuestionStyle', parent=styles['Normal'], fontSize=11, spaceAfter=10)
        option_style = ParagraphStyle('OptionStyle', parent=styles['Normal'], leftIndent=20, spaceAfter=5)
        answer_style = ParagraphStyle('AnswerStyle', parent=styles['Normal'], textColor=colors.darkgreen, spaceAfter=20)
        
        # Difficulty color coding
        difficulty_colors = {
            'Easy': colors.green,
            'Medium': colors.orange,
            'Hard': colors.red
        }
        
        difficulty_color = difficulty_colors.get(difficulty, colors.black)

        story.append(Paragraph(title_text, title_style))
        if exam_type and difficulty:
            story.append(Paragraph(f"Total Questions: {len(mcqs)}", subtitle_style))

        for i, mcq in enumerate(mcqs, 1):
            question = mcq.get("question", "No Question")
            options = mcq.get("options", [])
            answer = mcq.get("answer", "N/A")
            mcq_difficulty = mcq.get("difficulty", difficulty)

            # Question with difficulty indicator
            question_text = f"Q{i}. {question}"
            if mcq_difficulty:
                question_text += f" [{mcq_difficulty}]"
            
            story.append(Paragraph(question_text, question_style))
            for j, opt in enumerate(options):
                story.append(Paragraph(f"{chr(65 + j)}) {opt}", option_style))
            story.append(Paragraph(f"Answer: {answer}", answer_style))

            if i % 4 == 0:
                story.append(PageBreak())

        doc.build(story)
        return True
    except Exception as e:
        print(f"Error generating simple PDF: {e}")
        return False

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
    
    # Clean up common LaTeX issues for better rendering
    latex_str = latex_str.replace('\\text{', '\\mathrm{')
    latex_str = latex_str.replace('\\displaystyle', '')
    latex_str = latex_str.replace('\\,', ' ')
    latex_str = latex_str.replace('\\;', ' ')
    latex_str = latex_str.replace('\\!', '')
    
    return latex_str.strip()

def latex_to_png(latex_str, dpi=300, fontsize=14):
    """Convert LaTeX string to PNG image with proper mathematical rendering"""
    try:
        latex_str = clean_latex(latex_str)
        
        if not latex_str:
            return create_text_image("", fontsize)
        
        # Set up matplotlib for proper LaTeX rendering
        plt.rcParams['text.usetex'] = False  # Use mathtext for better compatibility
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams['font.size'] = fontsize
        
        # Create figure with appropriate size
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.axis('off')
        
        # Try to render as math text with better error handling
        try:
            # Check if the string contains math symbols
            if any(symbol in latex_str for symbol in ['\\frac', '\\sqrt', '\\sum', '\\int', '^', '_', '\\theta', '\\pi', '\\alpha', '\\beta', '\\gamma']):
                # Render as math
                ax.text(0.05, 0.5, f'${latex_str}$', 
                       fontsize=fontsize, 
                       ha='left', 
                       va='center',
                       transform=ax.transAxes)
            else:
                # Render as regular text
                ax.text(0.05, 0.5, latex_str, 
                       fontsize=fontsize, 
                       ha='left', 
                       va='center',
                       transform=ax.transAxes)
        except Exception as e:
            print(f"Math rendering failed for '{latex_str}': {e}")
            # Fallback to plain text
            ax.text(0.05, 0.5, latex_str, 
                   fontsize=fontsize, 
                   ha='left', 
                   va='center',
                   transform=ax.transAxes)
        
        # Save to BytesIO with better settings
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', transparent=False,
                   pad_inches=0.1)
        plt.close(fig)
        
        img_buf.seek(0)
        return img_buf
        
    except Exception as e:
        print(f"Error rendering LaTeX '{latex_str}': {e}")
        return create_text_image(latex_str, fontsize)

def render_latex_text(text, fontsize=12):
    """Render mixed text and LaTeX expressions"""
    try:
        # Set up matplotlib
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['mathtext.fontset'] = 'cm'
        
        fig, ax = plt.subplots(figsize=(10, 1.5))
        ax.axis('off')
        
        # Split text by $ delimiters and process each part
        parts = text.split('$')
        rendered_text = ""
        
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Regular text
                rendered_text += part
            else:  # LaTeX part
                rendered_text += f"${part}$"
        
        ax.text(0.05, 0.5, rendered_text, 
               fontsize=fontsize, 
               ha='left', 
               va='center',
               transform=ax.transAxes,
               wrap=True)
        
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', pad_inches=0.1)
        plt.close(fig)
        
        img_buf.seek(0)
        return img_buf
        
    except Exception as e:
        print(f"Error rendering text '{text}': {e}")
        return create_text_image(text, fontsize)

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

def get_image_dimensions(img_buf, width_limit=6*inch):
    """Auto-scale image based on aspect ratio"""
    try:
        img = PILImage.open(img_buf)
        orig_width, orig_height = img.size
        aspect_ratio = orig_height / orig_width
        target_width = width_limit
        target_height = target_width * aspect_ratio
        img_buf.seek(0)
        return target_width, target_height
    except Exception as e:
        print(f"Error sizing image: {e}")
        return width_limit, 0.8*inch  # fallback

def generate_pdf(mcqs, output_path, exam_type="", difficulty=""):
    """Generate PDF with properly rendered mathematical expressions"""
    try:
        print(f"🚀 Starting PDF generation with {len(mcqs)} MCQs")

        doc = SimpleDocTemplate(output_path, pagesize=letter, 
                                topMargin=1*inch, bottomMargin=1*inch,
                                leftMargin=1*inch, rightMargin=1*inch)
        styles = getSampleStyleSheet()
        story = []

        # Enhanced title with exam type and difficulty
        title_text = f"JEE {exam_type} MCQs - {difficulty} Level" if exam_type and difficulty else "JEE MCQ Questions"
        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=16,
                                     spaceAfter=30, alignment=1, textColor=colors.darkblue)
        
        subtitle_style = ParagraphStyle('SubtitleStyle', parent=styles['Normal'], fontSize=12,
                                       alignment=1, spaceAfter=20, textColor=colors.darkgreen,
                                       fontName='Helvetica-Bold')
        
        question_number_style = ParagraphStyle('QuestionNumberStyle', parent=styles['Normal'],
                                               fontSize=12, spaceAfter=10,
                                               textColor=colors.darkblue, fontName='Helvetica-Bold')
        option_style = ParagraphStyle('OptionStyle', parent=styles['Normal'],
                                      fontSize=10, leftIndent=30, spaceAfter=8)
        answer_style = ParagraphStyle('AnswerStyle', parent=styles['Normal'],
                                      fontSize=10, textColor=colors.darkgreen, fontName='Helvetica-Bold')

        # Title and subtitle
        story.append(Paragraph(title_text, title_style))
        if exam_type and difficulty:
            story.append(Paragraph(f"Total Questions: {len(mcqs)}", subtitle_style))
        story.append(Spacer(1, 20))

        for i, mcq in enumerate(mcqs, 1):
            print(f"📝 Processing question {i}/{len(mcqs)}")

            mcq_difficulty = mcq.get("difficulty", difficulty)
            question_header = f"Question {i}"
            if mcq_difficulty:
                question_header += f" [{mcq_difficulty}]"
            story.append(Paragraph(question_header, question_number_style))

            question_latex = mcq.get('question', '').strip()
            print(f"  Rendering question LaTeX: {question_latex}")

            question_text = mcq.get('question', '').strip()
            print(f"  Rendering question: {question_text}")

            if question_text:
                try:
                    # Use the new render function for mixed text/LaTeX
                    question_img_buf = render_latex_text(question_text, fontsize=14)
                    w, h = get_image_dimensions(question_img_buf, width_limit=6*inch)
                    question_img = Image(question_img_buf, width=w, height=h)
                    story.append(question_img)
                except Exception as e:
                    print(f"  ❌ Error rendering question: {e}")
                    story.append(Paragraph(f"Question: {question_text}", styles['Normal']))
            else:
                story.append(Paragraph("Question text missing", styles['Normal']))

            story.append(Spacer(1, 15))

            options = mcq.get('options', [])
            labels = ['A', 'B', 'C', 'D']
            print(f"  Processing {len(options)} options")

            for j, option in enumerate(options[:4]):
                if j < len(labels):
                    story.append(Paragraph(f"{labels[j]})", option_style))
                    if option:
                        try:
                            print(f"    Rendering option {labels[j]}: {option}")
                            option_img_buf = render_latex_text(f"{labels[j]}) {option}", fontsize=12)
                            w, h = get_image_dimensions(option_img_buf, width_limit=5*inch)
                            option_img = Image(option_img_buf, width=w, height=h)
                            story.append(option_img)
                        except Exception as e:
                            print(f"    ❌ Error rendering option {labels[j]}: {e}")
                            story.append(Paragraph(f"{labels[j]}) {option}", option_style))
                    else:
                        story.append(Paragraph("Option text missing", styles['Normal']))
                    story.append(Spacer(1, 8))

            answer = mcq.get('answer', '')
            if answer:
                story.append(Paragraph(f"Answer: {answer}", answer_style))

            story.append(Spacer(1, 25))

            if i % 3 == 0 and i < len(mcqs):
                story.append(PageBreak())

        print("🔨 Building PDF...")
        doc.build(story)
        print("✅ PDF generated successfully!")
        return True

    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_difficulty_prompt(difficulty, exam_type):
    """Generate appropriate prompts based on difficulty and exam type"""
    
    difficulty_guidelines = {
        'Easy': {
            'JEE Mains': "Focus on basic conceptual questions, direct formula applications, and straightforward problem-solving. Questions should test fundamental understanding without complex calculations.",
            'JEE Advanced': "Include conceptual questions with moderate complexity, requiring good understanding of fundamentals but not extremely challenging calculations."
        },
        'Medium': {
            'JEE Mains': "Include questions requiring moderate problem-solving skills, combination of concepts, and multi-step solutions. Suitable for average to good students.",
            'JEE Advanced': "Focus on questions requiring strong conceptual understanding, multi-concept integration, and moderate to complex problem-solving skills."
        },
        'Hard': {
            'JEE Mains': "Include challenging questions requiring deep conceptual understanding, complex problem-solving, and advanced application of concepts.",
            'JEE Advanced': "Focus on highly challenging questions requiring exceptional problem-solving skills, deep conceptual mastery, and ability to handle complex multi-step problems with novel approaches."
        }
    }
    
    return difficulty_guidelines.get(difficulty, {}).get(exam_type, "")

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
st.title("🎯 JEE MCQ Generator with Difficulty Levels")
st.markdown("Generate customized MCQs from PDF images with proper mathematical symbol rendering")

# Create two columns for exam type and difficulty selection
col1, col2 = st.columns(2)

with col1:
    exam_type = st.selectbox(
        "📚 Select Exam Type:",
        ["JEE Mains", "JEE Advanced"],
        help="Choose the exam type for appropriate question difficulty and pattern"
    )

with col2:
    difficulty = st.selectbox(
        "🎚️ Select Difficulty Level:",
        ["Easy", "Medium", "Hard"],
        index=1,  # Default to Medium
        help="Choose difficulty level based on your preparation level"
    )

# Display difficulty description
difficulty_descriptions = {
    "Easy": "💚 **Easy**: Basic conceptual questions, direct formula applications",
    "Medium": "🧡 **Medium**: Moderate problem-solving, combination of concepts", 
    "Hard": "❤️ **Hard**: Complex problem-solving, deep conceptual understanding"
}

st.info(difficulty_descriptions[difficulty])

# User input for total number of MCQs desired
total_mcqs = st.number_input(
    "🔢 How many MCQs do you want to generate?",
    min_value=1, max_value=100, value=20, step=2,
    help="Recommended: 20-30 for practice sessions"
)

# Batch size for each Claude call
BATCH_SIZE = 3

uploaded_file = st.file_uploader("📁 Upload a PDF with MCQ questions", type=["pdf"])

if uploaded_file:
    # Reset final_mcqs.json at the start of a new session
    final_mcqs_path = "final_mcqs.json"
    if os.path.exists(final_mcqs_path):
        os.remove(final_mcqs_path)
    st.success("✅ PDF uploaded successfully.")

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
        st.session_state.exam_type = None
        st.session_state.difficulty = None

    # Generate MCQs button
    if st.button("🚀 Generate MCQs", type="primary"):
        # Store current settings
        st.session_state.exam_type = exam_type
        st.session_state.difficulty = difficulty
        
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
            
            # Get difficulty-specific guidelines
            difficulty_guide = get_difficulty_prompt(difficulty, exam_type)
            
            prompt = (
                f"Extract {batch_size} unique MCQ questions from these images for {exam_type} at {difficulty} difficulty level. "
                f"DIFFICULTY GUIDELINES: {difficulty_guide} "
                f"Return the result as a JSON array of {batch_size} objects. "
                "Each object must have exactly these keys: 'question', 'options', 'answer', and 'difficulty'. "
                "'question' and each element of 'options' must be valid LaTeX strings for mathematical expressions, wrapped in double dollar signs ($ ... $). "
                "'answer' is the correct option as a string (A, B, C, or D). "
                f"'difficulty' should be '{difficulty}'. "
                "Ensure all mathematical symbols, fractions, integrals, derivatives, etc. are properly formatted in LaTeX. "
                "Make sure questions match the specified difficulty level and exam type requirements. "
                "Do not include any explanations. Output only the JSON array. "
                "Do not repeat any question from previous batches. Generate new and different questions each time. "
                f"Here are some questions already generated so far (do NOT repeat these):\n{prev_questions_text}"
            )
            
            data = {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 30000,
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
                    # Add difficulty info if missing
                    for mcq in mcqs_json:
                        if 'difficulty' not in mcq:
                            mcq['difficulty'] = difficulty
                    
                    all_mcqs.extend(mcqs_json)
                    all_mcqs = dedup_mcqs(all_mcqs)
                else:
                    st.warning("Claude did not return a valid MCQ list in this batch.")

                # Show progress after each batch
                percent = min(100, int(100 * len(all_mcqs) / total_mcqs))
                progress_bar.progress(percent)
                progress_info.info(f"🔄 Progress: {min(len(all_mcqs), total_mcqs)} / {total_mcqs} MCQs generated ({percent}%)")

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

        st.success(f"🎉 Generated {len(final_mcqs)} {exam_type} MCQs at {difficulty} level successfully!")
        st.rerun()  # Refresh the page to show the new buttons

# Check if MCQs exist (either in session state or file) and show download options
if st.session_state.mcqs_generated:
    st.subheader("📥 Download Options")
    
    # Load MCQs if not in session state
    if not st.session_state.mcqs_generated:
        try:
            with open(final_mcqs_path, 'r', encoding='utf-8') as f:
                st.session_state.mcqs_data = json.load(f)
                st.session_state.mcqs_generated = True
        except Exception as e:
            st.error(f"Error loading MCQs: {e}")
    
    # Show current status
    if st.session_state.mcqs_data:
        exam_info = f"{st.session_state.exam_type} - {st.session_state.difficulty}" if st.session_state.exam_type else ""
        st.info(f"📊 {len(st.session_state.mcqs_data)} MCQs ready for download {exam_info}")
    
    # Generate and download PDF
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📝 Generate PDF", help="Click to download MCQ PDF with difficulty levels"):
            if not st.session_state.mcqs_data:
                st.error("No MCQs found. Please generate MCQs first.")
            else:
                try:
                    exam_type_file = st.session_state.exam_type or exam_type
                    difficulty_file = st.session_state.difficulty or difficulty
                    pdf_filename = f"jee_{exam_type_file.lower().replace(' ', '_')}_{difficulty_file.lower()}_mcqs.pdf"
                    
                    with st.spinner("Creating PDF with difficulty levels..."):
                        success = generate_simple_pdf(
                            st.session_state.mcqs_data, 
                            pdf_filename,
                            exam_type_file,
                            difficulty_file
                        )
    
                    if success and os.path.exists(pdf_filename):
                        st.success(f"✅ {exam_type_file} {difficulty_file} MCQ PDF generated!")
                        with open(pdf_filename, "rb") as f:
                            st.download_button(
                                label=f"📄 Download {exam_type_file} {difficulty_file} MCQs PDF",
                                data=f.read(),
                                file_name=pdf_filename,
                                mime="application/pdf",
                                key="download_pdf"
                            )
                    else:
                        st.error("Failed to generate MCQ PDF.")
    
                except Exception as e:
                    st.error(f"❌ Error generating PDF: {str(e)}")
                    st.exception(e)

    with col2:
        # Download button for JSON
        if st.session_state.mcqs_data:
            json_data = json.dumps(st.session_state.mcqs_data, ensure_ascii=False, indent=2)
            exam_type_file = st.session_state.exam_type or exam_type
            difficulty_file = st.session_state.difficulty or difficulty
            json_filename = f"jee_{exam_type_file.lower().replace(' ', '_')}_{difficulty_file.lower()}_mcqs.json"
            
            st.download_button(
                label="📋 Download MCQs JSON",
                data=json_data,
                file_name=json_filename,
                mime="application/json",
                key="download_json"
            )

    # Statistics
    if st.session_state.mcqs_data:
        st.subheader("📈 Statistics")
        
        # Count questions by difficulty
        difficulty_counts = {}
        for mcq in st.session_state.mcqs_data:
            diff = mcq.get('difficulty', 'Unknown')
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Total Questions", len(st.session_state.mcqs_data))
        
        with col2:
            if st.session_state.exam_type:
                st.metric("📚 Exam Type", st.session_state.exam_type)
            
        with col3:
            if difficulty_counts:
                most_common_diff = max(difficulty_counts.items(), key=lambda x: x[1])
                st.metric("🎯 Primary Difficulty", f"{most_common_diff[0]} ({most_common_diff[1]})")

    # Preview some MCQs
    st.subheader("👀 Preview Generated MCQs")
    if st.session_state.mcqs_data:
        # Show first 3 MCQs as preview
        for i, mcq in enumerate(st.session_state.mcqs_data, 1):
            difficulty_emoji = {"Easy": "💚", "Medium": "🧡", "Hard": "❤️"}
            mcq_difficulty = mcq.get('difficulty', 'Unknown')
            difficulty_display = f"{difficulty_emoji.get(mcq_difficulty, '⚪')} {mcq_difficulty}"
            
            with st.expander(f"Preview Question {i} - {difficulty_display}"):
                st.write("**Question:**", mcq.get('question', 'N/A'))
                st.write("**Options:**")
                options = mcq.get('options', [])
                for j, option in enumerate(options):
                    st.write(f"  {chr(65+j)}) {option}")
                st.write("**Answer:**", mcq.get('answer', 'N/A'))
                st.write("**Difficulty:**", mcq.get('difficulty', 'Not specified'))
    else:
        st.info("No MCQs to preview. Generate MCQs first.")

# Add footer with tips
st.markdown("---")
st.markdown("### 💡 Tips for Better Results:")
st.markdown("""
- **Easy Level**: Perfect for beginners and concept building
- **Medium Level**: Ideal for regular practice and competitive preparation  
- **Hard Level**: Suitable for advanced preparation and challenging practice
- **JEE Mains**: Focus on NCERT-based concepts and moderate problem-solving
- **JEE Advanced**: Emphasis on conceptual depth and complex problem-solving
- Upload clear, high-quality PDF images for better question extraction
- Generate 20-30 questions at a time for optimal practice sessions
""")
