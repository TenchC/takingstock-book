import sys
import pandas as pd
import os
import glob

#using reportlab to create and style the pdf
from reportlab.platypus import Table, TableStyle, Paragraph, Spacer, PageTemplate, Frame, BaseDocTemplate, PageBreak, Flowable
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

#need pdfmetrics to register the font
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

#global vars
input_path = "input/"

#CSV Vars
inputCSV = 'topic_13_counts.csv'
number_of_topics = 100
max_count = 1
min_count = 1

# PDF Vars
outputPDF = 'output.pdf'
font_size = 12
font_file = 'fonts/CrimsonText-Regular.ttf'
footer_font_file = 'fonts/CrimsonText-SemiBold.ttf'
page_size = [432, 648] #in points, 72 points = 1 inch
#margin settings (in points) (72 points = 1 inch)
inner_margin = 90  # 1.25 inch
outter_margin = 54  # 0.75 inch
top_margin = 54
bottom_margin = 54

df = pd.read_csv(inputCSV)

# joined_files = os.path.join(input_path, "topic*.csv")

# # A list of all joined files is returned
# joined_list = glob.glob(joined_files)

# # Finally, the files are joined
# df = pd.concat(map(pd.read_csv, joined_list), ignore_index=True)

# max_count



#take the top number_of_topics and put them in a new dataframe, then sort them by description alphabetically
filtered_data = df.head(number_of_topics)
#filtered_data = filtered_data.sort_values(by="description", ascending=True)



#PDF Work Starts
class BlankPage(Flowable):
    def __init__(self):
        Flowable.__init__(self)
        self.width = 0
        self.height = 528

    def draw(self):
        pass

# Register the custom font
pdfmetrics.registerFont(TTFont('CrimsonText', font_file))
pdfmetrics.registerFont(TTFont('CrimsonText-SemiBold', footer_font_file))
INNER_MARGIN = inner_margin  
OUTER_MARGIN = outter_margin 
TOP_MARGIN = top_margin
BOTTOM_MARGIN = bottom_margin

# Create style for the table
styles = getSampleStyleSheet()
table_style = ParagraphStyle(
    'TableStyle',
    parent=styles['Normal'],
    fontSize=font_size,
    leading=14,
    spaceBefore=0,
    spaceAfter=0,
    fontName='CrimsonText' 
)

# Create style for the footer text
footer_style = ParagraphStyle(
    'FooterStyle',
    parent=styles['Normal'],
    fontSize=font_size,
    leading=14,
    spaceBefore=0,
    spaceAfter=0,
    fontName='CrimsonText-SemiBold',
    alignment=1  # Center alignment
)

# Convert the data to paragraphs for proper wrapping
data_for_pdf = []

# Add data rows with only descriptions
for _, row in filtered_data.iterrows():
    data_row = [Paragraph(str(row[1]), table_style)]  # Only the description column
    data_for_pdf.append(data_row)

# Create the footer text
footer_text = [int(s) for s in inputCSV.split('_') if s.isdigit()]
footer_text = "TOPIC: " + str(footer_text[0])

#create a custom doc template
class MyDocTemplate(BaseDocTemplate):
    def __init__(self, filename, **kw):
        super().__init__(filename, **kw)
        
        # Initialize page counter to 1 (first page)
        self.page = 1

        # Create frame for odd pages (outer margin on left)
        odd_frame = Frame(
            OUTER_MARGIN,
            self.bottomMargin +1,
            self.width - OUTER_MARGIN - INNER_MARGIN,  # Adjust width to account for margins
            self.height,
            id='odd'
        )

        # Create frame for even pages (inner margin on left)
        even_frame = Frame(
            INNER_MARGIN,
            self.bottomMargin+1,
            self.width - OUTER_MARGIN - INNER_MARGIN,  # Adjust width to account for margins
            self.height,
            id='even'
        )

        # Page templates
        odd_template = PageTemplate(
            id='odd',
            frames=[odd_frame],
            onPage=self.add_footer
        )

        even_template = PageTemplate(
            id='even',
            frames=[even_frame],
            onPage=self.add_footer
        )

        # Add both templates and set the first page template
        self.addPageTemplates([odd_template, even_template])
        self.pageTemplate = odd_template  # Set initial template

    def handle_pageEnd(self):
        """Switch templates between odd and even pages"""
        if self.page % 2 == 0:  # Even page
            self.pageTemplate = self.pageTemplates[1]  # Use even template
        else:  # Odd page
            self.pageTemplate = self.pageTemplates[0]  # Use odd template
        super().handle_pageEnd()

    def add_footer(self, canvas, doc):
        #skip first page
        if self.page != 1:
            canvas.saveState()
            canvas.setFont('CrimsonText-SemiBold', font_size)

            # Choose footer alignment based on even/odd page
            is_even = doc.page % 2 == 0
            x = OUTER_MARGIN if is_even else (page_size[0] - (OUTER_MARGIN*2))

            canvas.drawString(x, BOTTOM_MARGIN - 20, footer_text)
            canvas.restoreState()

#creating pdf
pdf = MyDocTemplate(
    outputPDF,
    pagesize=page_size,
    topMargin=TOP_MARGIN,
    bottomMargin=BOTTOM_MARGIN,
    leftMargin=0,  # Set left margin to 0 since we handle margins in frames
    rightMargin=0  # Set right margin to 0 since we handle margins in frames
)

# Set column width for single column - will be adjusted per page
column_width = [page_size[0] - INNER_MARGIN - OUTER_MARGIN]  # Full width minus margins

table = Table(data_for_pdf, colWidths=column_width)

#simple styling for the table
table.setStyle(TableStyle([
    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('LEFTPADDING', (0,0), (-1,-1), 6),
    ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ('TOPPADDING', (0,0), (-1,-1), 0),
    ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ('FONTNAME', (0,0), (-1,-1), 'CrimsonText'), 
]))

# Build the PDF with just the table
pdf.build([BlankPage(), table])

#export the pdf
#pdf.save()
print(f"PDF saved to {outputPDF}")
