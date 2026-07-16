"""
create_sample_pdf.py — generates a realistic academic calendar PDF for Docling ingestion demo.
Run once: python data/source_pdfs/create_sample_pdf.py
Requires: reportlab  (pip install reportlab)
"""

from pathlib import Path

OUTPUT = Path(__file__).parent / "academic_calendar_2025_26.pdf"


def create_pdf():
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4,
                                rightMargin=2*cm, leftMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=16, spaceAfter=6, alignment=TA_CENTER)
        subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=11, spaceAfter=4, alignment=TA_CENTER, textColor=colors.grey)
        heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=12, spaceBefore=12, spaceAfter=4, textColor=colors.HexColor('#1a3a6b'))
        body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=9, spaceAfter=3)

        story = []

        # Header
        story.append(Paragraph("CampusAI University", title_style))
        story.append(Paragraph("Academic Calendar — 2025–2026", subtitle_style))
        story.append(Paragraph("Issued by: Office of the Registrar | Ref: REG/AC/2025-26/001", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1a3a6b'), spaceAfter=12))

        # Odd Semester
        story.append(Paragraph("ODD SEMESTER (July 2025 – November 2025)", heading_style))

        odd_data = [
            ["Event", "Date", "Notes"],
            ["Semester Begins", "01-Jul-2025", "Orientation for first-year students: 01-02 Jul"],
            ["Last date for course registration", "07-Jul-2025", "Late registration fee: INR 200/day"],
            ["Mid-Semester Examination", "15-Sep-2025 to 22-Sep-2025", "In respective classrooms; 1.5 hrs duration"],
            ["Mid-semester result declaration", "29-Sep-2025", "Accessible on ERP portal"],
            ["Last teaching day", "31-Oct-2025", "Study leave: 01-Nov to 09-Nov"],
            ["End-Semester Examination begins", "10-Nov-2025", "As per detailed schedule issued by Exam Cell"],
            ["End-Semester Examination ends", "28-Nov-2025", "—"],
            ["Practical Examinations", "28-Nov-2025 to 05-Dec-2025", "Schedule by department coordinators"],
            ["Result Declaration (Odd Sem)", "27-Dec-2025", "Revaluation: within 7 days"],
            ["Winter Break", "06-Dec-2025 to 02-Jan-2026", "Campus closed 25-Dec to 01-Jan"],
        ]

        odd_table = Table(odd_data, colWidths=[6*cm, 5*cm, 7*cm])
        odd_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a3a6b')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f4f8')]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(odd_table)
        story.append(Spacer(1, 12))

        # Even Semester
        story.append(Paragraph("EVEN SEMESTER (January 2026 – May 2026)", heading_style))

        even_data = [
            ["Event", "Date", "Notes"],
            ["Semester Begins", "03-Jan-2026", "—"],
            ["Last date for course registration", "09-Jan-2026", "Late registration fee: INR 200/day"],
            ["Mid-Semester Examination", "02-Mar-2026 to 09-Mar-2026", "In respective classrooms; 1.5 hrs duration"],
            ["Mid-semester result declaration", "16-Mar-2026", "Accessible on ERP portal"],
            ["Last teaching day", "24-Apr-2026", "Study leave: 25-Apr to 04-May"],
            ["End-Semester Examination begins", "05-May-2026", "As per detailed schedule issued by Exam Cell"],
            ["End-Semester Examination ends", "22-May-2026", "—"],
            ["Practical Examinations", "22-May-2026 to 29-May-2026", "Schedule by department coordinators"],
            ["Result Declaration (Even Sem)", "20-Jun-2026", "Revaluation: within 7 days"],
            ["Summer Break", "30-May-2026 to 30-Jun-2026", "Summer term (optional) runs in June"],
        ]

        even_table = Table(even_data, colWidths=[6*cm, 5*cm, 7*cm])
        even_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a3a6b')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f4f8')]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(even_table)
        story.append(Spacer(1, 12))

        # Fee Payment Schedule
        story.append(Paragraph("FEE PAYMENT SCHEDULE 2025–2026", heading_style))
        fee_data = [
            ["Semester", "Deadline", "Late Fee"],
            ["Odd Semester 2025", "31-Jul-2025", "INR 500/day after deadline"],
            ["Even Semester 2026", "20-Jan-2026", "INR 500/day after deadline"],
            ["Supplementary Exam Registration (Odd Sem)", "20-Jun-2025", "INR 500 per subject"],
            ["Supplementary Exam Registration (Even Sem)", "20-Dec-2025", "INR 500 per subject"],
        ]
        fee_table = Table(fee_data, colWidths=[7*cm, 5*cm, 6*cm])
        fee_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d6a4f')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#d8f3dc')]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(fee_table)
        story.append(Spacer(1, 12))

        # Public Holidays
        story.append(Paragraph("PUBLIC HOLIDAYS & CAMPUS CLOSED DAYS 2025–2026", heading_style))
        holiday_data = [
            ["Date", "Day", "Holiday"],
            ["15-Aug-2025", "Friday", "Independence Day"],
            ["02-Oct-2025", "Thursday", "Gandhi Jayanti"],
            ["02-Nov-2025", "Sunday", "Diwali (campus closed Saturday also)"],
            ["25-Dec-2025", "Thursday", "Christmas"],
            ["01-Jan-2026", "Thursday", "New Year"],
            ["26-Jan-2026", "Monday", "Republic Day"],
            ["14-Mar-2026", "Saturday", "Holi"],
            ["14-Apr-2026", "Tuesday", "Dr. Ambedkar Jayanti"],
            ["01-May-2026", "Friday", "Maharashtra Day / Labour Day"],
        ]
        holiday_table = Table(holiday_data, colWidths=[4*cm, 3*cm, 11*cm])
        holiday_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#7b2d8b')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3e5f5')]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(holiday_table)
        story.append(Spacer(1, 16))

        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey, spaceAfter=6))
        story.append(Paragraph(
            "This academic calendar is subject to revision. Any changes will be notified via the university website "
            "(www.campusai.edu.in) and the student ERP portal. For queries: registrar@campusai.edu.in | ext. 1000.",
            body_style
        ))

        doc.build(story)
        print(f"[create_sample_pdf] PDF created: {OUTPUT}")

    except ImportError:
        _create_text_fallback()


def _create_text_fallback():
    """Fallback: create a plain-text representation that Docling can also parse."""
    text = """CampusAI University
Academic Calendar 2025-2026
Issued by: Office of the Registrar | Ref: REG/AC/2025-26/001

ODD SEMESTER (July 2025 - November 2025)

Event | Date | Notes
Semester Begins | 01-Jul-2025 | Orientation for first-year students: 01-02 Jul
Last date for course registration | 07-Jul-2025 | Late registration fee: INR 200/day
Mid-Semester Examination | 15-Sep-2025 to 22-Sep-2025 | In respective classrooms; 1.5 hrs duration
Mid-semester result declaration | 29-Sep-2025 | Accessible on ERP portal
Last teaching day | 31-Oct-2025 | Study leave: 01-Nov to 09-Nov
End-Semester Examination begins | 10-Nov-2025 | As per detailed schedule issued by Exam Cell
End-Semester Examination ends | 28-Nov-2025 |
Practical Examinations | 28-Nov-2025 to 05-Dec-2025 | Schedule by department coordinators
Result Declaration (Odd Sem) | 27-Dec-2025 | Revaluation: within 7 days
Winter Break | 06-Dec-2025 to 02-Jan-2026 | Campus closed 25-Dec to 01-Jan

EVEN SEMESTER (January 2026 - May 2026)

Event | Date | Notes
Semester Begins | 03-Jan-2026 |
Last date for course registration | 09-Jan-2026 | Late registration fee: INR 200/day
Mid-Semester Examination | 02-Mar-2026 to 09-Mar-2026 | In respective classrooms; 1.5 hrs duration
Mid-semester result declaration | 16-Mar-2026 | Accessible on ERP portal
Last teaching day | 24-Apr-2026 | Study leave: 25-Apr to 04-May
End-Semester Examination begins | 05-May-2026 | As per detailed schedule issued by Exam Cell
End-Semester Examination ends | 22-May-2026 |
Practical Examinations | 22-May-2026 to 29-May-2026 | Schedule by department coordinators
Result Declaration (Even Sem) | 20-Jun-2026 | Revaluation: within 7 days
Summer Break | 30-May-2026 to 30-Jun-2026 | Summer term (optional) runs in June

FEE PAYMENT SCHEDULE 2025-2026

Semester | Deadline | Late Fee
Odd Semester 2025 | 31-Jul-2025 | INR 500/day after deadline
Even Semester 2026 | 20-Jan-2026 | INR 500/day after deadline
Supplementary Exam Registration (Odd Sem) | 20-Jun-2025 | INR 500 per subject
Supplementary Exam Registration (Even Sem) | 20-Dec-2025 | INR 500 per subject

PUBLIC HOLIDAYS & CAMPUS CLOSED DAYS 2025-2026

Date | Day | Holiday
15-Aug-2025 | Friday | Independence Day
02-Oct-2025 | Thursday | Gandhi Jayanti
02-Nov-2025 | Sunday | Diwali
25-Dec-2025 | Thursday | Christmas
01-Jan-2026 | Thursday | New Year
26-Jan-2026 | Monday | Republic Day
14-Mar-2026 | Saturday | Holi
14-Apr-2026 | Tuesday | Dr. Ambedkar Jayanti
01-May-2026 | Friday | Maharashtra Day / Labour Day
"""
    txt_path = OUTPUT.with_suffix(".txt")
    txt_path.write_text(text, encoding="utf-8")
    print(f"[create_sample_pdf] reportlab not installed; text fallback created: {txt_path}")


if __name__ == "__main__":
    create_pdf()
