import os
from datetime import datetime
from fpdf import FPDF

class PDFReport(FPDF):
    def header(self):
        # Title
        self.set_font("helvetica", "B", 18)
        self.set_text_color(40, 40, 100)
        self.cell(0, 10, "Handwriting Personality Analysis Report", ln=True, align="C")
        self.ln(5)

    def footer(self):
        # Page numbers
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def generate_pdf_report(
    path: str,
    image_path: str,
    features: dict,
    prediction: dict,
    summary: str,
    analysis_image_path: str = None
):
    pdf = PDFReport(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    
    # ── Meta info ──
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 6, f"Analyzed Image : {os.path.basename(image_path)}", ln=True)
    pdf.cell(0, 6, f"Generated On   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 6, f"Methodology    : {prediction['method']}", ln=True)
    pdf.ln(10)

    # ── Summary ──
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "1. Personality Profile", ln=True)
    pdf.set_font("helvetica", "", 11)
    # Using multi_cell for text wrapping
    pdf.multi_cell(0, 6, summary.strip())
    pdf.ln(10)

    # ── Scores ──
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "2. Big-Five Trait Scores", ln=True)
    
    pdf.set_font("helvetica", "B", 11)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(60, 8, "Trait", border=1, fill=True)
    pdf.cell(40, 8, "Score", border=1, fill=True, align="C")
    pdf.cell(90, 8, "Diagnosis", border=1, fill=True)
    pdf.ln()

    pdf.set_font("helvetica", "", 10)
    scores = prediction["scores"]
    labels = prediction["labels"]
    
    for trait, score in scores.items():
        pdf.cell(60, 8, trait, border=1)
        pdf.cell(40, 8, f"{score:.2f} / 1.0", border=1, align="C")
        pdf.cell(90, 8, labels[trait], border=1)
        pdf.ln()
    pdf.ln(10)

    # ── Adding Visuals ──
    if analysis_image_path and os.path.exists(analysis_image_path):
        pdf.add_page()
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 10, "3. Visual Feature Analysis", ln=True)
        pdf.ln(5)
        # FPDF takes image width. 190 fills mostly an A4 portrait width (210mm wide)
        pdf.image(analysis_image_path, x=10, w=190)

    # ── Rule Engine Explanation ──
    rules = prediction.get("rules", [])
    if rules:
        pdf.add_page()
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 10, "4. Graphological Reasoning", ln=True)
        pdf.set_font("helvetica", "", 10)
        pdf.multi_cell(0, 6, "The following graphological features were detected and influenced the final scores:")
        pdf.ln(5)
        
        pdf.set_font("helvetica", "", 10)
        for r in rules:
            text = f"• [{r['trait']}] {r['feature']} → {r['effect']} ({r['reasoning']})"
            pdf.multi_cell(0, 6, text)
            pdf.ln(2)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    pdf.output(path)
