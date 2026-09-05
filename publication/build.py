#!/usr/bin/env python3
"""Build a vector atlas and the entire currently written book, offline.

Requires Pandoc/XeLaTeX for book PDF and reportlab/svglib for vector conversion.
Generated outputs are ignored; source-controlled SVGs remain previewable on GitHub.
"""
from pathlib import Path
import html
import json
import os
import re
import subprocess
import textwrap
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import String
from svglib.svglib import svg2rlg

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf'
BUILD=ROOT/'build/publication'


def font_path(name):
    override=os.environ.get('BOOK_FONT_DIR')
    if override and (Path(override)/name).is_file(): return str(Path(override)/name)
    resolved=subprocess.run(['kpsewhich',name],capture_output=True,text=True,check=True).stdout.strip()
    if not resolved: raise RuntimeError('Missing '+name+'; install DejaVu fonts or set BOOK_FONT_DIR')
    return resolved


def main():
    OUT.mkdir(parents=True,exist_ok=True); BUILD.mkdir(parents=True,exist_ok=True)
    subprocess.run(['python3',str(ROOT/'figures/build.py'),'--check'],check=True)
    fonts={key:font_path(name) for key,name in [('sans','DejaVuSans.ttf'),('mono','DejaVuSansMono.ttf'),('serif','DejaVuSerif.ttf')]}
    pdfmetrics.registerFont(TTFont('DejaVu Sans',fonts['sans']))
    def set_vector_font(node):
        # svglib may resolve a CSS font stack to Helvetica before registration.
        # Apply the verified Unicode font to every vector label explicitly.
        if isinstance(node,String): node.fontName='DejaVu Sans'
        for child in getattr(node,'contents',[]): set_vector_font(child)
    manifest=json.loads((ROOT/'figures/manifest.json').read_text())
    atlas=canvas.Canvas(str(OUT/'visual-atlas.pdf'),pagesize=(720,610),invariant=1)
    atlas.setTitle('Inside the LLM Engine - Visual Prototype Atlas')
    for entry in manifest['figures']:
        svg=ROOT/entry['generated'][0]; drawing=svg2rlg(str(svg))
        if drawing is None: raise RuntimeError('Cannot parse '+str(svg))
        set_vector_font(drawing)
        renderPDF.drawToFile(drawing,str(BUILD/(svg.stem+'.pdf')))
        scale=660/drawing.width
        drawing.scale(scale,scale); renderPDF.draw(drawing,atlas,30,112)
        atlas.setFont('DejaVu Sans',9)
        for i,line in enumerate(textwrap.wrap(entry['caption'],112)):
            atlas.drawString(30,95-13*i,line)
        atlas.drawString(30,22,'Prototype / '+entry['id']+' / source: '+entry['source'])
        atlas.showPage()
    atlas.save()
    # Resolve source links against the branch; do not leave machine paths in publications.
    host='https://github.com/hermonai/inside-the-llm-engine/blob/astra-visual-rewrite/'
    parts=['# Inside the LLM Engine\n\nFrom First Token to Production-Grade Inference\n\nWorking edition: seven completed chapters. Visual regeneration prototype milestone.\n\n']
    for path in sorted((ROOT/'manuscript').glob('part-*/chapter-*.md')):
        content=path.read_text()
        def link(m):
            raw=m.group(2)
            if raw.startswith(('https:','http:','#','mailto:')): return m.group(0)
            target=(path.parent/raw.split('#')[0]).resolve()
            try: relative=target.relative_to(ROOT)
            except ValueError: return m.group(0)
            return '['+m.group(1)+']('+host+str(relative)+')'
        content=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,content)
        parts.append(content)
    manuscript='\n\n'.join(parts)
    source=BUILD/'book.md'; source.write_text(manuscript)
    pdf_source=BUILD/'book-print.md'
    # Preserve the joined emoji grapheme as a vector glyph in print. XeTeX's
    # monochrome fallback cannot shape the modifier/ZWJ sequence correctly.
    print_text=manuscript.replace('`👩🏽‍💻🚀`',r'`\texttwemoji{1f469-1f3fd-200d-1f4bb}\texttwemoji{1f680}`{=latex}').replace('`👩🏽‍💻`',r'`\texttwemoji{1f469-1f3fd-200d-1f4bb}`{=latex}')
    print_text+='\n\n## Publication colophon\n\nComposite emoji graphics in this PDF use Twemoji (Twitter and contributors), CC BY 4.0, through the TeX Live twemojis package. The source and HTML preserve the original Unicode sequences.\n'
    pdf_source.write_text(print_text)
    # fvextra wraps long code; a small mono font preserves 100-column Unicode figures.
    header=BUILD/'header.tex'
    header.write_text(r'''\usepackage{fvextra}
\DefineVerbatimEnvironment{verbatim}{Verbatim}{breaklines=true,fontsize=\scriptsize}
\fvset{breaklines=true,fontsize=\scriptsize}
\usepackage[AutoFallBack=true]{xeCJK}
\usepackage{twemojis}
\setCJKmainfont{FandolSong-Regular.otf}
\setCJKmonofont{FandolFang-Regular.otf}
\setCJKfallbackfamilyfont{\CJKrmdefault}{NotoEmoji-Regular.ttf}
\setCJKfallbackfamilyfont{\CJKttdefault}{NotoEmoji-Regular.ttf}
\xeCJKDeclareCharClass{CJK}{"1F300 -> "1FAFF, "200D}
\xeCJKDeclareCharClass{Default}{"2018, "2019, "201C, "201D}
''')
    common=['pandoc',str(source),'--standalone','--toc','--metadata','title=Inside the LLM Engine','--syntax-highlighting=none']
    pdf_command=common.copy(); pdf_command[1]=str(pdf_source)
    result=subprocess.run(pdf_command+['--pdf-engine=xelatex','-V','mainfont=DejaVuSerif.ttf','-V','monofont=DejaVuSansMono.ttf','-V','mathfont=latinmodern-math.otf','-V','geometry:margin=18mm','-V','fontsize=10pt','-H',str(header),'-o',str(OUT/'inside-the-llm-engine.pdf')],check=True,capture_output=True,text=True)
    if 'Missing character:' in result.stderr:
        raise RuntimeError('PDF glyph coverage failure:\n'+result.stderr)
    if result.stderr: print(result.stderr)
    # MathML is local/offline and accessible; no CDN renderer is required.
    subprocess.run(common+['--mathml','--embed-resources','-o',str(BUILD/'book.html')],check=True)
    gallery=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Visual atlas</title><style>body{font:18px system-ui;max-width:1000px;margin:2rem auto;padding:1rem;color:#152b3c}svg{width:100%;height:auto}figure{margin:2rem 0}figcaption{line-height:1.5}</style><h1>Inside the LLM Engine: visual atlas</h1><p>Ten prototypes. Educational mechanisms beyond Chapter 7 are specifications, not implemented mini-engine features.</p>']
    for entry in manifest['figures']:
        gallery+=['<figure>',(ROOT/entry['generated'][0]).read_text(),'<figcaption>'+html.escape(entry['caption'])+'</figcaption>']
        if entry['animation']:
            import shutil
            # Package a self-contained HTML atlas directory with playable counterparts.
            for rel in [entry['animation'],entry['generated'][0],'figures/generated/player.js']:
                shutil.copyfile(ROOT/rel,BUILD/Path(rel).name)
            gallery.append('<p><a href="'+Path(entry['animation']).name+'">Play the step sequence</a></p>')
        gallery.append('</figure>')
    (BUILD/'atlas.html').write_text('\n'.join(gallery)+'</html>')
    print('Built full seven-chapter PDF/HTML and ten-plate vector PDF/HTML atlas')


if __name__=='__main__': main()
