#!/usr/bin/env python3
"""Emit a portable, source-controlled LaTeX edition with native TikZ figures.

The restricted SVG dialect is fail-closed. Legacy character diagrams become
geometric TikZ strokes and separately typeset labels, not verbatim graphs.
Run publication/build.py first to prepare the resolved prose and TeX header.
"""
from pathlib import Path
import argparse
import json
import re
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'publication/latex'


def escape(text):
    mapping = {'\\': r'\textbackslash{}', '{': r'\{', '}': r'\}', '$': r'\$',
               '&': r'\&', '#': r'\#', '%': r'\%', '_': r'\_',
               '^': r'\textasciicircum{}', '~': r'\textasciitilde{}'}
    return ''.join(mapping.get(c, c) for c in text)


def color(value):
    if value.startswith('#'):
        return '{rgb,255:red,%d;green,%d;blue,%d}' % tuple(int(value[i:i+2], 16) for i in (1, 3, 5))
    if value not in ('white', 'black', 'none'):
        raise ValueError('Unsupported color ' + value)
    return value


def svg_tikz(path):
    root = ET.parse(path).getroot()
    if set(root.attrib) - {'viewBox','role','aria-labelledby'}:
        raise ValueError('Unsupported SVG root attributes')
    if root.attrib['viewBox'] != '0 0 1000 720':
        raise ValueError('Unexpected figure coordinate contract')
    out = [r'\begin{tikzpicture}[x=.5pt,y=-.5pt]', r'\path[use as bounding box] (0,0) rectangle (1000,720);']
    for node in root:
        tag = node.tag.split('}')[-1]
        a = node.attrib
        if tag in ('title', 'desc'):
            continue
        allowed = {'rect','text','line','polygon','circle'}
        if tag not in allowed:
            raise ValueError('Unsupported SVG node ' + tag)
        attributes = {
            'rect': {'x','y','width','height','fill','stroke','stroke-width'},
            'line': {'x1','x2','y1','y2','stroke','stroke-width','stroke-dasharray'},
            'text': {'x','y','fill','font-family','font-size','text-anchor'},
            'circle': {'cx','cy','r','fill','stroke'},
            'polygon': {'points','fill'},
        }
        if set(a) - attributes[tag] or list(node):
            raise ValueError('Unsupported SVG attributes or nested text in '+tag)
        fill = color(a.get('fill', 'none'))
        stroke = color(a.get('stroke', 'none'))
        style = f'fill={fill},draw={stroke},line width={float(a.get("stroke-width",1))*.5}pt'
        if tag == 'rect':
            x,y,w,h = (float(a.get(k,0)) for k in ('x','y','width','height'))
            out.append(r'\path['+style+f'] ({x},{y}) rectangle ({x+w},{y+h});')
        elif tag == 'line':
            if 'stroke-dasharray' in a:
                dash = [float(v)*.5 for v in re.split(r'[,\s]+', a['stroke-dasharray'])]
                style += f',dash pattern=on {dash[0]}pt off {dash[-1]}pt'
            out.append(r'\draw['+style+f'] ({a["x1"]},{a["y1"]}) -- ({a["x2"]},{a["y2"]});')
        elif tag == 'polygon':
            points = re.findall(r'(-?[\d.]+),(-?[\d.]+)', a['points'])
            out.append(r'\path['+style+'] '+' -- '.join('('+x+','+y+')' for x,y in points)+' -- cycle;')
        elif tag == 'circle':
            out.append(r'\path['+style+f'] ({a["cx"]},{a["cy"]}) circle [radius={a["r"]}];')
        elif tag == 'text':
            anchor = {'start':'base west','middle':'base','end':'base east'}[a.get('text-anchor','start')]
            size = float(a['font-size'])*.5
            out.append(r'\node[inner sep=0pt,outer sep=0pt,anchor='+anchor+',text='+fill+
                       r',font=\sffamily\fontsize{'+str(size)+'}{'+str(size*1.2)+r'}\selectfont] at ('+
                       a['x']+','+a['y']+') {'+escape(node.text or '')+'};')
    return '\n'.join(out+[r'\end{tikzpicture}'])+'\n'


CONNECT = {
    '─':'lr','━':'lr','═':'lr','┄':'lr','┈':'lr','│':'ud','┃':'ud','║':'ud',
    '┌':'rd','┐':'ld','└':'ru','┘':'lu','├':'urd','┤':'uld','┬':'lrd','┴':'lru','┼':'lrud',
    '╭':'rd','╮':'ld','╰':'ru','╯':'lu',
}
ARROWS = {'▶':'r','▷':'r','►':'r','→':'r','▼':'d','▽':'d','↓':'d',
          '◀':'l','◁':'l','←':'l','▲':'u','△':'u','↑':'u'}


def legacy_tikz(text):
    """Preserve audited legacy topology; replace box characters with vector edges."""
    rows = text.rstrip().splitlines()
    width = max(map(len, rows))
    out = [r'\begin{tikzpicture}[x=6.2pt,y=-13pt,draw=black!65,line width=.45pt]',
           r'\path[use as bounding box] (-1,-1) rectangle ('+f'{width+1},{len(rows)}'+');']
    vectors = {'l':(-.5,0), 'r':(.5,0), 'u':(0,-.5), 'd':(0,.5)}
    for y,row in enumerate(rows):
        # Draw connectors independently of text glyphs. Labels are normal TeX text.
        for x,ch in enumerate(row):
            for direction in CONNECT.get(ch,''):
                dx,dy = vectors[direction]
                style = ''
                if ch in '═║':
                    style = '[double,double distance=.9pt,line width=.3pt]'
                elif ch in '━┃':
                    style = '[line width=.9pt]'
                elif ch in '┄┈':
                    style = '[dash pattern=on 1pt off 1pt]'
                out.append(r'\draw'+style+' '+f'({x},{y}) -- ({x+dx},{y+dy});')
            if ch in ARROWS:
                dx,dy = vectors[ARROWS[ch]]
                out.append(r'\draw[-{Latex[length=2pt,width=3pt]}] '+f'({x-dx},{y-dy}) -- ({x+dx},{y+dy});')
        clean = ''.join(' ' if ch in CONNECT or ch in ARROWS else ch for ch in row)
        for match in re.finditer(r'\S(?:.*?\S)?(?= {2,}|$)', clean):
            label = match.group()
            # Fit proportional labels to their audited cell spans, without scaling up.
            out.append(r'\node[anchor=west,inner sep=0pt,font=\sffamily\fontsize{9}{11}\selectfont] at ('+
                       f'{match.start()-.4},{y}'+r') {\begin{adjustbox}{max width='+str(len(label)*6.2)+
                       'pt}'+escape(label)+r'\end{adjustbox}};')
    return '\n'.join(out+[r'\end{tikzpicture}'])+'\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    files = {}
    manifest = json.loads((ROOT/'figures/manifest.json').read_text())
    for entry in manifest['figures']:
        svg = ROOT/entry['generated'][0]
        files['figures/'+svg.stem+'.tex'] = '% Generated from '+entry['generated'][0]+'; do not hand edit.\n'+svg_tikz(svg)
    content = (ROOT/'build/publication/book-print.md').read_text()
    content = content.replace('# Inside the LLM Engine\n', '# About this working edition\n', 1)
    # All manuscript vector plates become native TikZ, not embedded PDF/bitmap labels.
    def image(match):
        stem = Path(match.group(2)).stem
        assert 'figures/'+stem+'.tex' in files
        return '\n```{=latex}\n'+r'\begin{figure}[H]\centering\resizebox{\linewidth}{!}{\input{figures/'+stem+r'.tex}}\caption{'+escape(match.group(1))+r'}\end{figure}'+'\n```\n'
    content = re.sub(r'!\[([^\]]+)\]\(([^)]+)\)', image, content)
    count = 0
    def diagram(match):
        nonlocal count
        body = match.group(1)
        if not any(c in CONNECT or c in ARROWS for c in body):
            return match.group(0)
        count += 1
        name = f'legacy-{count:03}.tex'
        files['figures/'+name] = '% Legacy topology preserved as native vectors and typeset labels.\n'+legacy_tikz(body)
        return '\n```{=latex}\n'+r'\begin{center}\begin{adjustbox}{max width=\linewidth}\input{figures/'+name+r'}\end{adjustbox}\end{center}'+'\n```\n'
    content = re.sub(r'```text\n(.*?)\n```', diagram, content, flags=re.S)
    # Long inline identifiers, not only filesystem paths, must wrap in quotes/tables.
    lines = []
    fenced = False
    for line in content.splitlines():
        if line.lstrip().startswith('```'):
            fenced = not fenced
        if not fenced and not line.lstrip().startswith('```'):
            def identifier(match):
                value = match.group(1)
                if len(value) < 16 or '\\' in value or any(c.isspace() for c in value):
                    return match.group(0)
                return '`'+r'\texttt{\seqsplit{'+escape(value)+'}}`{=latex}'
            line = re.sub(r'`([^`\n]+)`', identifier, line)
        lines.append(line)
    content = '\n'.join(lines)+'\n'
    result = subprocess.run(['pandoc','--from=markdown','--to=latex','--standalone','--toc','--toc-depth=2',
             '--syntax-highlighting=none','--metadata','title=Inside the LLM Engine',
             '-V','mainfont=DejaVuSerif.ttf','-V','sansfont=DejaVuSans.ttf',
             '-V','monofont=DejaVuSansMono.ttf','-V','mathfont=latinmodern-math.otf',
             '-V','geometry:margin=18mm','-V','fontsize=10pt',
             '-H',str(ROOT/'build/publication/header.tex')],input=content,text=True,capture_output=True,check=True)
    native_header = r'''\usepackage{tikz}
\usepackage{adjustbox}
\usetikzlibrary{arrows.meta}
\setmainfont[BoldFont=DejaVuSerif-Bold.ttf,ItalicFont=DejaVuSerif-Italic.ttf,BoldItalicFont=DejaVuSerif-BoldItalic.ttf]{DejaVuSerif.ttf}
\setsansfont[BoldFont=DejaVuSans-Bold.ttf,ItalicFont=DejaVuSans-Oblique.ttf]{DejaVuSans.ttf}
\setmonofont[BoldFont=DejaVuSansMono-Bold.ttf,ItalicFont=DejaVuSansMono-Oblique.ttf]{DejaVuSansMono.ttf}
\setlength{\emergencystretch}{3em}
\makeatletter
\renewcommand{\@pnumwidth}{2em}
\renewcommand{\@tocrmarg}{3em}
\makeatother
\renewcommand{\maketitle}{\hypersetup{pageanchor=false}\begin{titlepage}\centering
\vspace*{35mm}
{\sffamily\fontsize{34}{42}\selectfont Inside the\\LLM Engine\par}
\vspace{12mm}
{\Large From model weights and KV memory\\to industrial inference serving.\par}
\vspace{18mm}\rule{.7\linewidth}{.6pt}\par\vspace{8mm}
{\sffamily Native LaTeX and vector-illustrated working edition\par}
\vfill Eight completed chapters\par September 2026
\end{titlepage}\hypersetup{pageanchor=true}}
\let\editioncontents\tableofcontents
\renewcommand{\tableofcontents}{\clearpage\editioncontents\clearpage}
\begin{document}'''
    tex = result.stdout.replace(r'\begin{document}', native_header)
    def table_widths(match):
        table = match.group(0)
        if 'SentencePiece fixture' not in table:
            return table
        widths = iter(('0.30','0.14','0.12','0.20','0.24'))
        return re.sub(r'\\real\{[\d.]+\}', lambda _: r'\real{'+next(widths)+'}', table, count=5)
    tex = re.sub(r'\\begin\{longtable\}.*?\\end\{longtable\}', table_widths, tex, flags=re.S)
    if str(ROOT) in tex:
        raise ValueError('Machine path leaked into portable TeX')
    files['main.tex'] = '% Generated by publication/build-tex.py; portable native-vector edition.\n'+tex
    files['manifest.json'] = json.dumps({'canonical_and_reference_plates':len(manifest['figures']),
        'legacy_vector_diagrams':count,'files':sorted(files)},indent=2)+'\n'
    for relative,data in files.items():
        path = DEST/relative
        if args.check:
            if not path.is_file() or path.read_text()!=data:
                raise SystemExit('Stale native TeX: '+relative)
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(data)
    print(f'Native TeX edition: {len(manifest["figures"])} semantic plates, {count} legacy vector diagrams; '+('checked' if args.check else 'generated'))


if __name__ == '__main__':
    main()
