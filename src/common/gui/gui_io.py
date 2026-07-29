"""
gui_io.py
---------
Reusable PyQt6 helpers for copying/exporting QTableWidget contents and
Matplotlib figures. Deliberately independent of any specific window
class -- callers pass in the table/figure/parent widget they want
acted on, rather than these functions reaching into a particular
GUI's ``self``.
"""

import csv
import io

from PyQt6.QtWidgets import QApplication, QFileDialog


def table_to_csv(table, row_labels=None, row_label_header="Row"):
    """
    Serialize a QTableWidget to a CSV string.

    Parameters
    ----------
    table : PyQt6.QtWidgets.QTableWidget
        The table to serialize.
    row_labels : sequence, optional
        One label per row (e.g. depth values), inserted as a leading
        column. Omitted entirely if not given.
    row_label_header : str
        Header text for the leading column, if `row_labels` is given.

    Returns
    -------
    str
    """
    n_rows = table.rowCount()
    n_cols = table.columnCount()

    headers = []
    for col in range(n_cols):
        item = table.horizontalHeaderItem(col)
        text = item.text() if item is not None else ""
        headers.append(text.replace("\n", " "))
    if row_labels is not None:
        headers = [row_label_header] + headers

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)

    for row in range(n_rows):
        values = [
            table.item(row, col).text() if table.item(row, col) is not None else ""
            for col in range(n_cols)
        ]
        if row_labels is not None:
            label = row_labels[row]
            label = f"{label:g}" if isinstance(label, (int, float)) else str(label)
            values = [label] + values
        writer.writerow(values)

    return buf.getvalue()


def copy_table_csv(table, row_labels=None, row_label_header="Row"):
    """Copy a QTableWidget's contents to the system clipboard as CSV,
    ready to paste into a spreadsheet or document."""
    QApplication.clipboard().setText(
        table_to_csv(table, row_labels=row_labels, row_label_header=row_label_header)
    )


def export_table_csv(table, parent=None, default_name="table.csv",
                      row_labels=None, row_label_header="Row"):
    """
    Prompt for a file (via a save dialog parented to `parent`) and
    write a QTableWidget's contents there as CSV.

    Returns
    -------
    str or None
        The chosen path, or None if the dialog was cancelled.
    """
    path, _ = QFileDialog.getSaveFileName(
        parent, "Export table as CSV", default_name, "CSV files (*.csv)"
    )
    if not path:
        return None
    with open(path, "w", newline="") as f:
        f.write(table_to_csv(table, row_labels=row_labels, row_label_header=row_label_header))
    return path


def export_figure_pdf(figure, parent=None, default_name="figure.pdf"):
    """
    Prompt for a file (via a save dialog parented to `parent`) and save
    a Matplotlib figure there as a PDF.

    Returns
    -------
    str or None
        The chosen path, or None if the dialog was cancelled.
    """
    path, _ = QFileDialog.getSaveFileName(
        parent, "Export figure as PDF", default_name, "PDF files (*.pdf)"
    )
    if not path:
        return None
    figure.savefig(path, bbox_inches="tight")
    return path
