import argparse


rules = '''rule mf
  command = mf '\mode=ljfour; mode_setup; input $in'

rule gftopk
  command = gftopk $in $out

rule texgen
  command = PYTHONPATH=../build python3 -m gen.texgen $in

rule pdflatex
  command = pdflatex $in

rule convert
  command = convert -density 300 -trim -monochrome -white-threshold 50% -black-threshold 50% $in -quality 100 -gravity center -extent 96x96 -transparent white $out

rule copy
  command = cp $in $out

'''

build = '''
build %s.600gf: mf ../metafont/%s.mf
build %s.600pk: gftopk %s.600gf
build %s.tex: texgen %s.600pk
build %s.pdf: pdflatex %s.tex
build %s.png: convert %s.pdf
build ../glyph/%s.png: copy %s.png
'''

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--build-directory", type=str, default='build', help="build directory")
opt = parser.parse_args()


if __name__ == '__main__':
    import os.path as pth
    from glob import glob

    script = rules
    for mffile in glob('metafont/*.mf'):
        chname = pth.basename(mffile).replace('.mf', '')
        params = tuple([chname for _ in list(range(build.count('%s')))])
        content = build % params
        script += content

    with open('%s/build.ninja' % opt.build_directory, 'w') as f:
        f.write(script)
        f.flush()
