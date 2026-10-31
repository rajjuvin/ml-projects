"""The downloadable sample workbook: the sample encounters plus a "Field guide" sheet explaining every column.

CSV is plain text and cannot hold a second sheet, so the workbook is offered next to the plain CSV file. To
upload, save its "Sample data" sheet as CSV."""

import csv
import io

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

from readmission import config

DATA_SHEET = "Sample data"
GUIDE_SHEET = "Field guide"
TITLE = "Hospital Readmission Risk Prediction: sample encounters"
INSTRUCTIONS = [
    "Edit the 'Sample data' sheet: keep the header row, use made-up values only, and keep at most 10 rows.",
    "Medication columns are optional. A medication you leave out counts as No.",
    "Save the 'Sample data' sheet as CSV (File > Save As > CSV, with that sheet selected) and upload the CSV file "
    "on the page. The workbook itself cannot be uploaded.",
    "Do not enter real patient information.",
]
GUIDE_HEADER = ["Column", "Required", "Meaning", "Allowed values", "Example"]
GUIDE_WIDTHS = [30, 10, 60, 70, 22]
HEADER_FILL = PatternFill("solid", fgColor="DCE6F5")


def _style_header(row):
    for cell in row:
        cell.font = Font(bold=True)
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(vertical="center", wrap_text=True)


def build_sample_workbook(guide_rows, sample_csv):
    """The workbook as bytes. `guide_rows` come from InputSchema.guide(); `sample_csv` is the CSV text."""
    workbook = Workbook()
    workbook.properties.title = TITLE
    workbook.properties.creator = "Hospital Readmission Risk Prediction"

    data = workbook.active
    data.title = DATA_SHEET
    header, *rows = csv.reader(io.StringIO(sample_csv))
    data.append(header)
    for row in rows:
        data.append([int(value) if column in config.NUMERIC_COLUMNS else value
                     for column, value in zip(header, row, strict=True)])
    _style_header(data[1])
    data.freeze_panes = "A2"
    for index, column in enumerate(header, start=1):
        width = max(len(column), *(len(str(row[index - 1])) for row in rows)) + 3
        data.column_dimensions[data.cell(row=1, column=index).column_letter].width = width

    guide = workbook.create_sheet(GUIDE_SHEET)
    guide.append([TITLE])
    guide["A1"].font = Font(bold=True, size=14)
    for number, line in enumerate(INSTRUCTIONS, start=1):
        guide.append([f"{number}. {line}"])
        guide.merge_cells(start_row=guide.max_row, start_column=1, end_row=guide.max_row, end_column=5)
        guide.cell(row=guide.max_row, column=1).alignment = Alignment(wrap_text=True, vertical="top")
        guide.row_dimensions[guide.max_row].height = 32
    guide.append([])
    guide.append(GUIDE_HEADER)
    _style_header(guide[guide.max_row])
    first_table_row = guide.max_row
    for row in guide_rows:
        guide.append([row["column"], "yes" if row["required"] else "no", row["meaning"], row["allowed"],
                      str(row["example"])])
    for row in guide.iter_rows(min_row=first_table_row + 1):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    for index, width in enumerate(GUIDE_WIDTHS):
        guide.column_dimensions[guide.cell(row=1, column=index + 1).column_letter].width = width
    guide.freeze_panes = guide.cell(row=first_table_row + 1, column=1)

    workbook.active = 0  # opens on the sample data, the sheet to export as CSV
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()
