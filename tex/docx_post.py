"""Fix pandoc's docx tables: content-based column widths, 9pt table text.

Usage: python docx_post.py <in.docx> <out.docx>
"""
import re
import sys
import zipfile

src, dst = sys.argv[1], sys.argv[2]
WIDTH = 9000  # twips, about the text width


def table(m):
    # Column widths proportional to the longest cell text in each column
    # (cells spanning several columns are ignored).
    tbl = m.group(0)
    n = len(re.findall(r"<w:gridCol ", tbl))
    longest = [3] * n
    for row in re.findall(r"<w:tr>.*?</w:tr>", tbl, re.S):
        cells = re.findall(r"<w:tc>.*?</w:tc>", row, re.S)
        if len(cells) != n:
            continue
        for j, cell in enumerate(cells):
            text = "".join(re.findall(r"<(?:w|m):t[^>]*>([^<]*)<", cell))
            longest[j] = max(longest[j], len(text))
    longest = [k + 4 for k in longest]  # room for the cell margins
    total = sum(longest)
    cols = "".join(f'<w:gridCol w:w="{WIDTH * k // total}" />' for k in longest)
    return re.sub(r"<w:tblGrid>.*?</w:tblGrid>", f"<w:tblGrid>{cols}</w:tblGrid>",
                  tbl, flags=re.S)


with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == "word/document.xml":
            xml = data.decode()
            xml = re.sub(r"<w:tbl>.*?</w:tbl>", table, xml, flags=re.S)
            xml = xml.replace('<w:tblW w:type="auto" w:w="0" />',
                              f'<w:tblW w:type="dxa" w:w="{WIDTH}" />')
            data = xml.encode()
        elif item.filename == "word/styles.xml":
            xml = data.decode()
            # Table text at 9pt (the "Table" style pandoc assigns).
            # rPr must come before tblPr in a style.
            xml = re.sub(r'(<w:style [^>]*w:styleId="Table"[^>]*>.*?)(<w:tblPr>)',
                         r'\1<w:rPr><w:sz w:val="18" /><w:szCs w:val="18" /></w:rPr>\2',
                         xml, count=1, flags=re.S)
            data = xml.encode()
        zout.writestr(item, data)
