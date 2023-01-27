import argparse

import os.path as pth


text = '''\\documentclass{article}
\\usepackage[paperwidth=1.618in, paperheight=1in]{geometry}
\\newfont{\\bongbaletter}{%s}
\\newcommand{\\%s}{{\\bongbaletter %s}}

\\begin{document}
\\thispagestyle{empty}
\\hspace{0pt}\\vfill
\\begin{center}
\\%s\\
\\end{center}
\\vfill\hspace{0pt}
\\end{document}'''

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--output-directory", type=str, default='.', help="output directory")
parser.add_argument("input_file")
opt = parser.parse_args()

chname = pth.basename(opt.input_file).replace('.600pk', '')
outfile = '%s.tex' % chname
outdir = opt.output_directory
outfile = pth.join(outdir, outfile)

if __name__ == '__main__':
    with open(outfile, 'w') as f:
        f.write(text % tuple([chname for _ in range(text.count('%s'))]))
        f.flush()
