"""Render every page, audit visible word bounds, and make review contact sheets."""
import argparse
import json
import re
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageOps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    bbox = args.output/"bounds.xhtml"
    subprocess.run(["pdftotext","-bbox",str(args.pdf),str(bbox)],check=True)
    # Preserve boxes for legacy math delimiters whose extracted C0 text is
    # illegal in XML. This does not edit the PDF or discard the word's geometry.
    xml, controls = re.subn(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "\ufffd", bbox.read_text())
    root = ET.fromstring(xml)
    pages = list(root.iter("{http://www.w3.org/1999/xhtml}page"))
    violations = []
    for i,page in enumerate(pages,1):
        width,height = float(page.attrib["width"]),float(page.attrib["height"])
        for word in page.iter("{http://www.w3.org/1999/xhtml}word"):
            box = [float(word.attrib[k]) for k in ("xMin","yMin","xMax","yMax")]
            if box[0]<12 or box[1]<12 or box[2]>width-12 or box[3]>height-12:
                violations.append({"page":i,"word":word.text,"box":box})
    assert not violations, violations[:20]
    subprocess.run(["pdftoppm","-r","100","-png",str(args.pdf),str(args.output/"page")],check=True)
    rendered = sorted(args.output.glob("page-*.png"))
    assert len(rendered)==len(pages)
    for start in range(0,len(rendered),12):
        sheet=Image.new("RGB",(1320,2600),"#d7dfe5")
        draw=ImageDraw.Draw(sheet)
        for j,path in enumerate(rendered[start:start+12]):
            pic=ImageOps.contain(Image.open(path).convert("RGB"),(420,615))
            x=10+(j%3)*440; y=30+(j//3)*650
            sheet.paste(pic,(x,y))
            draw.text((x,y-22),f"PDF page {start+j+1}",fill="black")
        sheet.save(args.output/f"contact-{start//12+1:02}.png")
        ImageOps.grayscale(sheet).save(args.output/f"gray-{start//12+1:02}.png")
    record={"pages":len(pages),"word_bounds":"passed","rendered_pages":len(rendered),
            "legacy_math_extraction_controls":controls,
            "contact_sheets":(len(rendered)+11)//12,
            "visual_inspection":"not automatic; open and inspect images"}
    (args.output/"render-record.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record))


if __name__=="__main__":
    main()
