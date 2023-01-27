import numpy as np
import math
import os
import random as r


tmpl = """
%% file name: %s.mf
%% mode_setup;
%% Define a random shape for the training corpus

beginchar("%s",12pt#,12pt#,0);
  %% Setup coordinates as an equation system
%s

%s

  %% Draw the character curve
  %% z1 is the same as (x1, y1)
%s
endchar;

end
    """[1:-4]


total_entropy = 7.0


def shanon_entropy(p):
    return - (p * np.log2(p)).sum()


def get_start_position(current_entropy):
    elems = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    probs = np.ones(8) / 8
    entropy = 2 * shanon_entropy(probs)
    return current_entropy - entropy, np.random.choice(elems, 2, p=probs)


def get_start_angle(current_entropy):
    elems = np.array([0, 30, 60, 90, 120, 150, 180])
    probs = np.array([0.0625, 0.0625, 0.0625, 0.625, 0.0625, 0.0625, 0.0625])
    return current_entropy - shanon_entropy(probs), np.random.choice(elems, 1, p=probs)[0]


def get_random_nradical(current_entropy):
    elems = np.array([1, 2, 3, 4])
    probs = np.ones(4) / 4
    return current_entropy - shanon_entropy(probs), np.random.choice(elems, 1, p=probs)[0]


def get_random_npart(entropy):
    elems = np.array([1, 2, 3, 4])
    probs = np.ones(4) / 4
    return entropy - shanon_entropy(probs), np.random.choice(elems, 1, p=probs)[0]


def get_random_ncontrol(current_entropy):
    base = int(current_entropy) * (current_entropy > 0) + 2
    elems = np.arange(2, base + 2)
    probs = np.ones(base) / base
    return current_entropy - shanon_entropy(probs), np.random.choice(elems, 1, p=probs)[0]


def get_random_length(current_entropy):
    base = int(np.log2(current_entropy * (current_entropy > 0) + 1) * 2 + 2) * 2
    elems = np.arange(1, base + 1)
    probs = np.ones(base) / base
    return current_entropy - shanon_entropy(probs), np.random.choice(elems, 1, p=probs)[0]


def get_random_angle(current_entropy, angl):
    elems = np.array([0, 30, 60, 90, 120, 150, 180])
    probs = np.array([0.0625, 0.0625, 0.0625, 0.625, 0.0625, 0.0625, 0.0625])
    return current_entropy - shanon_entropy(probs), (angl + np.random.choice(elems, 1, p=probs)[0]) % 360


def get_random_pen():
    x = r.random() * 0.06 + 0.02
    y = r.random() * 0.06 + 0.02
    a = r.randint(0, 360)
    return '  pickup pencircle xscaled %0.2fw yscaled %0.2fw rotated %d;' % (x, y, a)


def get_random_path(current_entropy):
    init_entropy = current_entropy
    current_entropy, (x, y) = get_start_position(current_entropy)
    current_entropy, angl = get_start_angle(current_entropy)

    xs, ys = [x], [y]
    current_entropy, ncontrol = get_random_ncontrol(current_entropy)
    n = int(ncontrol)
    for i in range(n):
        current_entropy, leng = get_random_length(current_entropy)
        x1 = int(x + leng * math.cos(math.radians(angl)))
        y1 = int(y + leng * math.sin(math.radians(angl)))
        if 0 < x1 < 9 and 0 < y1 < 9 and not (x1 == x and y1 == y):
            x, y = x1, y1
            xs.append(x)
            ys.append(y)

            current_entropy, angl = get_random_angle(current_entropy, angl)

    m = min(len(xs), len(ys))
    success = current_entropy < init_entropy * 0.7 and m == n
    if not success:
        return get_random_path(init_entropy)
    else:
        zs = list(['z%s%s' % (x, y) for x, y in zip(xs, ys)])
        ps = '..'.join(['%s' for _ in range(m)])
        ts = ps % tuple(zs)
        return current_entropy, '  draw %s;' % ts, zs


def gen_random_metafont(fname):
    points = []
    lines = []
    lines.append(get_random_pen())

    current_entropy = total_entropy
    current_entropy, nradical = get_random_nradical(current_entropy)
    for ix in range(int(nradical)):
        entropy_by_radical = current_entropy / nradical
        entropy_by_part, npart = get_random_npart(entropy_by_radical)
        for jx in range(int(npart)):
            entropy = entropy_by_part
            entropy, path, path_points = get_random_path(entropy)
            points.extend(path_points)
            lines.append(path)

    paths = '\n'.join(lines)
    points = sorted(list(set(points)))
    xpoints = [p.replace('z', 'x') for p in points]
    ypoints = [p.replace('z', 'y') for p in points]

    xdefs = '\n'.join(['  %s = %s * w / 10;' % (x, x[1]) for x in xpoints])
    ydefs = '\n'.join(['  %s = %s * w / 10;' % (y, y[2]) for y in ypoints])

    return tmpl % (fname, fname, xdefs, ydefs, paths)


def main():
    for _ in range(100):
        fname = ''.join(r.sample('abcdefghijklmnopqrstuvwxyz', 5))
        fpath = 'metafont/%s.mf' % fname
        if not os.path.exists(fpath):
            with open(fpath, 'w') as f:
                f.write(gen_random_metafont(fname))


if __name__ == '__main__':
    main()
