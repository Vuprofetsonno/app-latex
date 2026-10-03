"""Khảo sát hàm số — chạy bằng: python -m streamlit run web.py.
Giữ bốn phần khảo sát và hai chức năng xuất LaTeX của bản gốc.
"""
import ast
import io
import math

import streamlit as st
import sympy as sp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import MaxNLocator

BLUE = "#2563eb"
INK = "#172b4d"
GRAY = "#64748b"
X = sp.Symbol("x", real=True)
PLOT_STYLE = {
    "font.family": "DejaVu Sans", "mathtext.fontset": "stix",
    "font.size": 12, "axes.unicode_minus": True,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "text.color": INK, "axes.edgecolor": INK,
}


def tex(value):
    if value == sp.oo:
        return r"+\infty"
    if value == -sp.oo:
        return r"-\infty"
    return sp.latex(sp.simplify(value))


def math_label(value):
    # Matplotlib MathText dùng $...$, KHÔNG dùng \(...\).
    return "$" + tex(value) + "$"


def ox_intersections_latex(roots):
    """Chỉ đổi cách trình bày giao điểm; giữ nghiệm gốc để tính toán."""
    points, approximations = [], []
    for index, root in enumerate(roots, 1):
        shown = root
        if isinstance(root, sp.CRootOf) and root.poly.degree() <= 4:
            # Ưu tiên dạng căn thực gọn, ví dụ 1 - căn bậc ba của 2.
            candidates = sp.solve(root.poly.as_expr(), root.poly.gen)
            for candidate in candidates:
                if candidate.is_real is not True or candidate.has(sp.CRootOf):
                    continue
                if abs(sp.N(candidate - root, 30)) > sp.Float("1e-25"):
                    continue
                if root.equals(candidate) is True:
                    shown = sp.simplify(candidate)
                    break
        if shown.has(sp.CRootOf):
            label = fr"\alpha_{{{index}}}"
            decimal = format(float(sp.N(root, 20)), ".6f").replace(".", "{,}")
            approximations.append(label + r"\approx " + decimal)
        else:
            label = tex(shown)
        points.append(r"\left(" + label + r";0\right)")
    return r"Ox:\quad" + r",\quad".join(points), r",\qquad".join(approximations)


def real_roots(expr, x):
    expr = sp.cancel(expr)
    if expr == 0 or not expr.has(x):
        return []
    poly = sp.Poly(expr, x)
    try:
        roots = poly.real_roots()
    except (NotImplementedError, sp.PolynomialError):
        solved = sp.solveset(expr, x, domain=sp.S.Reals)
        if not isinstance(solved, sp.FiniteSet):
            raise ValueError("Chưa xác định được đầy đủ nghiệm thực của biểu thức này.")
        roots = list(solved)
    return sorted(set(roots), key=lambda a: float(sp.N(a, 18)))


def read_expression(source, x=X):
    """Đọc phép toán thông thường, không thực thi mã Python người dùng nhập.
    Giữ các điểm loại của mẫu gốc, kể cả khi phân thức rút gọn được.
    """
    source = source.strip().replace("^", "**")
    if not source or len(source) > 800:
        raise ValueError("Vui lòng nhập biểu thức có độ dài từ 1 đến 800 ký tự.")
    excluded = []
    names = {"x": x, "pi": sp.pi, "E": sp.E}
    funcs = {"sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos,
             "tan": sp.tan, "exp": sp.exp, "log": sp.log, "Abs": sp.Abs}

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return sp.Rational(str(node.value))
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = visit(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in funcs and len(node.args) == 1 and not node.keywords:
                return funcs[node.func.id](visit(node.args[0]))
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                if b == 0:
                    raise ValueError("Mẫu số không được đồng nhất bằng 0.")
                excluded.append(sp.fraction(sp.together(b))[0])
                return a / b
            if isinstance(node.op, ast.Pow):
                if not b.is_number or abs(float(b)) > 30:
                    raise ValueError("Vui lòng dùng số mũ hằng có trị tuyệt đối không quá 30.")
                if b.is_negative:
                    excluded.append(sp.fraction(sp.together(a))[0])
                return a ** b
        raise ValueError("Cú pháp chưa hợp lệ. Dùng x, số, +, -, *, /, ** và ngoặc tròn.")

    try:
        y = visit(ast.parse(source, mode="eval").body)
    except SyntaxError as exc:
        raise ValueError("Thiếu ngoặc hoặc sai cú pháp biểu thức.") from exc
    if y.has(sp.zoo, sp.nan, sp.oo, -sp.oo) or y.is_real is False:
        raise ValueError("Biểu thức không xác định hoặc không nhận giá trị thực.")
    # Thuật toán khảo sát gốc dành cho đa thức/phân thức trên R.
    # Không suy diễn sai D = R cho sqrt(x), log(x), ...
    if not y.is_rational_function(x):
        raise ValueError("Bản này khảo sát đa thức và phân thức hữu tỉ. Hàm chứa căn, "
                         "logarit hoặc lượng giác theo x cần thuật toán miền xác định riêng.")
    _, den = sp.fraction(sp.together(y))
    excluded.append(den)
    holes = []
    for e in excluded:
        if not e.is_polynomial(x):
            raise ValueError("Mẫu số cần là biểu thức đa thức theo x.")
        holes.extend(real_roots(e, x))
    return sp.cancel(y), sorted(set(holes), key=lambda a: float(sp.N(a, 18)))


def variation_data(y_sym, y_prime, x_sym, nghiem_thuc, nghiem_mau):
    pts = sorted(set(nghiem_thuc + nghiem_mau), key=lambda a: float(sp.N(a, 18)))
    probes = ([sp.S.Zero] if not pts else [pts[0] - 1] +
              [(a + b) / 2 for a, b in zip(pts, pts[1:])] + [pts[-1] + 1])
    signs = []
    for p in probes:
        v = sp.simplify(y_prime.subs(x_sym, p))
        s = sp.sign(v)
        if s not in (-1, 0, 1):
            s = sp.sign(sp.N(v, 40))
        if s not in (-1, 0, 1):
            raise ValueError("Chưa xác định được dấu đạo hàm trên một khoảng.")
        signs.append(int(s))
    left = [sp.limit(y_sym, x_sym, -sp.oo)]
    right = [left[0]]
    for p in pts:
        left.append(sp.limit(y_sym, x_sym, p, dir="-") if p in nghiem_mau
                    else sp.simplify(y_sym.subs(x_sym, p)))
        right.append(sp.limit(y_sym, x_sym, p, dir="+") if p in nghiem_mau
                     else left[-1])
    end = sp.limit(y_sym, x_sym, sp.oo)
    left.append(end)
    right.append(end)
    return pts, signs, left, right


def survey(source):
    y, excluded = read_expression(source)
    derivative = sp.factor(sp.diff(y, X))
    num_d, _ = sp.fraction(sp.together(derivative))
    stationary = [p for p in real_roots(num_d, X) if p not in excluded]
    num, den = sp.fraction(sp.together(y))
    asymptote, kind = None, ""
    if sp.degree(den, X) > 0:
        quotient, _ = sp.div(num, den, X)
        if quotient == 0 or sp.degree(quotient, X) <= 1:
            asymptote = sp.simplify(quotient)
            kind = "ngang" if not asymptote.has(X) else "xiên"
    poles = [p for p in excluded if any(sp.limit(y, X, p, dir=d) in (sp.oo, -sp.oo)
                                       for d in ("-", "+"))]
    return dict(y=y, derivative=derivative, stationary=stationary,
                excluded=excluded, poles=poles, asymptote=asymptote, kind=kind)


# ==========================================
# 1. BẢNG BIẾN THIÊN TRÊN WEB: HÌNH TĨNH SẮC NÉT
# ==========================================
def ve_bang_bien_thien(y_sym, y_prime, x_sym, nghiem_thuc, nghiem_mau):
    pts, signs, left, right = variation_data(y_sym, y_prime, x_sym, nghiem_thuc, nghiem_mau)
    count = len(pts) + 2
    positions = np.arange(count, dtype=float) * 2.2 + 1.8
    width = positions[-1] + 0.85
    fig, ax = plt.subplots(figsize=(max(7.5, 2.1 * count + 1), 3.1))
    ax.set(xlim=(0, width), ylim=(0, 3.55))
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), width, 3.55, fc="white", ec=INK, lw=1.1))
    ax.add_patch(Rectangle((0, 2.77), width, 0.78, fc="#eff6ff", ec="none"))
    ax.plot([0, width, width, 0, 0], [0, 0, 3.55, 3.55, 0], color=INK, lw=1.1)
    for yy in (2.1, 2.77):
        ax.plot([0, width], [yy, yy], color=INK, lw=0.9)
    ax.plot([0.9, 0.9], [0, 3.55], color=INK, lw=0.9)
    for label, yy in (("$x$", 3.16), ("$y'$", 2.43), ("$y$", 1.05)):
        ax.text(0.45, yy, label, ha="center", va="center", fontsize=18)
    for pos, p in zip(positions, [-sp.oo] + pts + [sp.oo]):
        ax.text(pos, 3.16, math_label(p), ha="center", va="center", fontsize=17)
    for j, s in enumerate(signs):
        ax.text((positions[j] + positions[j + 1]) / 2, 2.43,
                "$" + {1: "+", -1: "-", 0: "0"}[s] + "$",
                ha="center", va="center", fontsize=18, color=BLUE)
    for j, p in enumerate(pts, 1):
        if p in nghiem_mau:
            for dx in (-0.035, 0.035):
                ax.plot([positions[j] + dx] * 2, [0.08, 2.7], color=INK, lw=0.9)
        else:
            ax.text(positions[j], 2.43, "$0$", ha="center", va="center", fontsize=16)

    # Chỉ neo mũi tên tại cực trị hoặc điểm loại của TXĐ.
    # Nghiệm đạo hàm không đổi dấu nằm TRÊN cùng một mũi tên liên tục.
    anchors = [0] + [j for j, p in enumerate(pts, 1)
                     if p in nghiem_mau or signs[j - 1] != signs[j]] + [count - 1]
    bbox = dict(facecolor="white", edgecolor="none", pad=1.8)
    high, low, mid = 1.73, 0.35, 1.04
    for a, b in zip(anchors, anchors[1:]):
        s = signs[a]
        ya, yb = ((low, high) if s > 0 else (high, low) if s < 0 else (mid, mid))
        xa, xb = positions[a], positions[b]
        if 0 < a < count - 1 and pts[a - 1] in nghiem_mau:
            xa += 0.42
        if 0 < b < count - 1 and pts[b - 1] in nghiem_mau:
            xb -= 0.42
        ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="->",
                                    mutation_scale=15, lw=1.25, color=INK,
                                    shrinkA=23, shrinkB=23, zorder=2))
        ax.text(xa, ya, math_label(right[a]), ha="center", va="center",
                fontsize=16, bbox=bbox, zorder=3)
        ax.text(xb, yb, math_label(left[b]), ha="center", va="center",
                fontsize=16, bbox=bbox, zorder=3)
        for j in range(a + 1, b):
            t = (positions[j] - xa) / (xb - xa)
            yy = ya + t * (yb - ya)
            ax.plot(positions[j], yy, "o", ms=3.3, color=BLUE, zorder=4)
            ax.annotate(math_label(left[j]), (positions[j], yy), xytext=(0, 17),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=15, bbox=bbox, zorder=5)
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.045, top=0.97)
    return fig


# ==========================================
# 2. XUẤT MÃ TKZ-TAB
# ==========================================
def sinh_ma_bbt_tikz(y_sym, y_prime, x_sym, nghiem_thuc, nghiem_mau):
    pts, signs, left, right = variation_data(y_sym, y_prime, x_sym, nghiem_thuc, nghiem_mau)
    if all(s == 0 for s in signs):
        # tkz-tab không có mũi tên ngang trong tkzTabVar: vẽ bảng hằng bằng TikZ.
        xs = [r"-\infty"] + [tex(p) for p in pts] + [r"+\infty"]
        out = [r"\begin{center}", r"\begin{tikzpicture}[>=stealth]",
               r"\tkzTabInit[lgt=1,espcl=2.3,deltacl=0.6]{$x$/0.8,$y'$/0.7,$y$/1.6}"
               + "{" + ",".join("$" + p + "$" for p in xs) + "}",
               r"\tkzTabLine{," + ",d,".join("0" for _ in signs) + ",}"]
        for i in range(len(signs)):
            # N hàng tkz-tab nằm tại ranh giới hàng: y từ -1.5 đến -3.1.
            out.append(fr"\draw[->] ([xshift=8pt,yshift=-0.8cm]N{i+1}2) -- "
                       fr"node[above] {{$ {tex(y_sym)} $}} ([xshift=-8pt,yshift=-0.8cm]N{i+2}2);")
        for i in range(2, len(xs)):
            out.append(fr"\draw[double,double distance=1.5pt] (N{i}2) -- (N{i}3);")
        return "\n".join(out + [r"\end{tikzpicture}", r"\end{center}"])
    line = [""]
    for j, p in enumerate(pts):
        line.extend(["+" if signs[j] > 0 else "-", "d" if p in nghiem_mau else "z"])
    line.extend(["+" if signs[-1] > 0 else "-", ""])
    variations = [("-" if signs[0] > 0 else "+") + "/$" + tex(right[0]) + "$"]
    resting = []
    anchors = [1]
    for j, p in enumerate(pts, 1):
        before, after = signs[j - 1], signs[j]
        if p in nghiem_mau:
            a = "+" if before > 0 else "-"
            b = "-" if after > 0 else "+"
            variations.append(a + "D" + b + "/$" + tex(left[j]) + "$/ $" + tex(right[j]) + "$")
            anchors.append(j + 1)
        elif before != after:
            variations.append(("+" if before > 0 else "-") + "/$" + tex(left[j]) + "$")
            anchors.append(j + 1)
        else:
            variations.append("R/")
            resting.append(j + 1)
    variations.append(("+" if signs[-1] > 0 else "-") + "/$" + tex(left[-1]) + "$")
    anchors.append(len(pts) + 2)
    out = [r"% Preamble: \usepackage{tikz,tkz-tab}", r"\begin{center}",
           r"\begin{tikzpicture}[>=stealth]",
           r"\tkzTabInit[nocadre=false,lgt=1.1,espcl=2.6,deltacl=0.7]",
           r"  {$x$/0.85, $y'$/0.75, $y$/2.5}",
           "  {" + ", ".join("$" + tex(p) + "$" for p in [-sp.oo] + pts + [sp.oo]) + "}",
           r"\tkzTabLine{" + ", ".join(line) + "}",
           r"\tkzTabVar{" + ", ".join(variations) + "}"]
    for j in resting:
        a = max(t for t in anchors if t < j)
        b = min(t for t in anchors if t > j)
        out.append(fr"\tkzTabIma{{{a}}}{{{b}}}{{{j}}}{{$ {tex(left[j - 1])} $}}")
    return "\n".join(out + [r"\end{tikzpicture}", r"\end{center}"])


# ==========================================
# 3. ĐỒ THỊ VÀ NHÃN TRỤC / XUẤT TIKZ
# ==========================================
def tinh_toan_khung_do_thi(y_sym, x_sym, nghiem_thuc, nghiem_mau, tiem_can_y):
    xp = [0.] + [float(p) for p in nghiem_thuc + nghiem_mau]
    for p in nghiem_mau:
        xp.extend([float(p) - 3, float(p) + 3])
    xmin, xmax = math.floor(min(xp) - 1.6), math.ceil(max(xp) + 1.6)
    if xmax - xmin < 6:
        xmin, xmax = min(xmin, -3), max(xmax, 3)
    # Luôn giữ gốc tọa độ và toàn bộ điểm dừng trong khung.
    yp = [0.]
    samples = list(nghiem_thuc)
    if sp.S.Zero not in nghiem_mau:
        samples.append(sp.S.Zero)
    for p in nghiem_mau:
        samples.extend([p - 1, p + 1])
        if tiem_can_y is not None:
            yp.append(float(tiem_can_y.subs(x_sym, p)))
    for p in samples:
        if p in nghiem_mau:
            continue
        v = y_sym.subs(x_sym, p)
        if v.is_real and v.is_finite:
            yp.append(float(v))
    spread = max(yp) - min(yp)
    pad = max(2, spread * 0.22)
    ymin, ymax = math.floor(min(yp) - pad), math.ceil(max(yp) + pad)
    yrange = ymax - ymin
    step = max(1, math.ceil(yrange / 10))
    return xmin, xmax, ymin, ymax, step, yrange


def curve_segments(y_sym, x_sym, excluded, frame):
    xmin, xmax, ymin, ymax = frame[:4]
    fn = sp.lambdify(x_sym, y_sym, modules="numpy")
    cuts = [xmin] + sorted(float(p) for p in excluded if xmin < float(p) < xmax) + [xmax]
    segments = []
    for index, (a, b) in enumerate(zip(cuts, cuts[1:])):
        eps = max((b - a) * 1e-7, 1e-10)
        lo = a + eps if index else a
        hi = b - eps if index < len(cuts) - 2 else b
        xs = np.linspace(lo, hi, 1500)
        # Thêm mẫu sát hai đầu để không mất nhánh gần tiệm cận.
        near = np.geomspace(eps, max(eps * 2, (hi - lo) * 0.15), 150)
        xs = np.unique(np.concatenate([xs, lo + near, hi - near]))
        xs = xs[(xs >= lo) & (xs <= hi)]
        with np.errstate(all="ignore"):
            ys = np.asarray(fn(xs), dtype=float)
            ys = np.broadcast_to(ys, xs.shape)
        current = []
        for xx, yy in zip(xs, ys):
            if np.isfinite(yy) and ymin - 0.08 * (ymax-ymin) <= yy <= ymax + 0.08 * (ymax-ymin):
                current.append((float(xx), float(yy)))
            elif current:
                if len(current) > 1:
                    segments.append(np.asarray(current))
                current = []
        if len(current) > 1:
            segments.append(np.asarray(current))
    return segments


def graph_marks(y_sym, x_sym, stationary, excluded, asymptote):
    """Dùng chung điểm/nhãn cho Matplotlib và TikZ; khử nhãn trùng."""
    points = [(p, sp.simplify(y_sym.subs(x_sym, p))) for p in stationary]
    xmarks, ymarks = {}, {}
    for p, v in points:
        if p != 0:
            xmarks[p] = "above" if float(v) < 0 else "below"
        if v != 0:
            ymarks[v] = "right" if float(p) < 0 else "left"
    for p in excluded:
        if p != 0:
            xmarks.setdefault(p, "below")
    if asymptote is not None and not asymptote.has(x_sym) and asymptote != 0:
        ymarks.setdefault(asymptote, "left")
    return points, xmarks, ymarks


def place_axis_labels(fig, ax, marks):
    """Dời chữ bằng offset (point), không thay đổi tọa độ toán học."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied = [t.get_window_extent(renderer).expanded(1.06, 1.1) for t in ax.texts]
    for value, axis, side in marks:
        anchor = (float(value), 0) if axis == "x" else (0, float(value))
        artist = None
        direction = 1 if side in ("above", "right") else -1
        for attempt in range(9):
            offset = 15 + (attempt // 3) * 13
            shift = (0, -17, 17)[attempt % 3]
            xytext = (shift, direction * offset) if axis == "x" else (direction * offset, shift)
            ha = "center" if axis == "x" else ("left" if direction > 0 else "right")
            va = ("bottom" if direction > 0 else "top") if axis == "x" else "center"
            artist = ax.annotate(math_label(value), anchor, xytext=xytext,
                                 textcoords="offset points", ha=ha, va=va,
                                 fontsize=13, annotation_clip=False,
                                 bbox=dict(fc="white", ec="none", alpha=0.94, pad=1.1), zorder=8)
            box = artist.get_window_extent(renderer).expanded(1.08, 1.15)
            if not any(box.overlaps(old) for old in occupied) or attempt == 8:
                occupied.append(box)
                break
            artist.remove()


def ve_do_thi_sgk(y_sym, x_sym, nghiem_thuc, nghiem_mau, tiem_can_y):
    frame = tinh_toan_khung_do_thi(y_sym, x_sym, nghiem_thuc, nghiem_mau, tiem_can_y)
    xmin, xmax, ymin, ymax = frame[:4]
    fig, ax = plt.subplots(figsize=(9, 6.3))
    fig.subplots_adjust(left=0.08, right=0.94, bottom=0.09, top=0.95)
    for seg in curve_segments(y_sym, x_sym, nghiem_mau, frame):
        ax.plot(seg[:, 0], seg[:, 1], color=BLUE, lw=2.05, zorder=4)
    for p in nghiem_mau:
        limits = [sp.limit(y_sym, x_sym, p, dir=d) for d in ("-", "+")]
        if any(v in (sp.oo, -sp.oo) for v in limits):
            ax.axvline(float(p), color=GRAY, ls=(0, (5, 4)), lw=1, zorder=1)
        elif limits[0] == limits[1] and limits[0].is_finite:
            ax.plot(float(p), float(limits[0]), "o", mfc="white", mec=BLUE, ms=7, zorder=6)
    if tiem_can_y is not None:
        ax.plot([xmin, xmax], [float(tiem_can_y.subs(x_sym, v)) for v in (xmin, xmax)],
                color=GRAY, ls=(0, (5, 4)), lw=1, zorder=1)
        # Chỉ đặt I nếu giao hai tiệm cận thật sự là tâm đối xứng.
        if len(nghiem_mau) == 1:
            p = nghiem_mau[0]
            v = sp.simplify(tiem_can_y.subs(x_sym, p))
            if sp.simplify(y_sym + y_sym.subs(x_sym, 2*p-x_sym) - 2*v) == 0:
                ax.plot(float(p), float(v), "o", color=INK, ms=4, zorder=6)
                ax.annotate("$I$", (float(p), float(v)), xytext=(10, 10),
                            textcoords="offset points", fontsize=15,
                            bbox=dict(fc="white", ec="none", pad=1), zorder=8)
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    for name in ("left", "bottom"):
        ax.spines[name].set_position("zero")
        ax.spines[name].set_linewidth(1.0)
    for name in ("right", "top"):
        ax.spines[name].set_visible(False)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    ax.tick_params(axis="both", length=4, width=0.85, direction="inout",
                   labelbottom=False, labelleft=False, colors=GRAY)
    ax.plot(xmax, 0, marker=">", color=INK, ms=7, clip_on=False)
    ax.plot(0, ymax, marker="^", color=INK, ms=7, clip_on=False)
    ax.annotate("$x$", (xmax, 0), xytext=(10, -9), textcoords="offset points", fontsize=17)
    ax.annotate("$y$", (0, ymax), xytext=(-13, 9), textcoords="offset points", fontsize=17)
    ax.annotate("$O$", (0, 0), xytext=(-15, -17), textcoords="offset points", fontsize=15,
                bbox=dict(fc="white", ec="none", pad=0.6), zorder=8)
    points, xmarks, ymarks = graph_marks(y_sym, x_sym, nghiem_thuc, nghiem_mau, tiem_can_y)
    for p, v in points:
        px, py = float(p), float(v)
        if px and py:
            ax.plot([px, px, 0], [0, py, py], ls=(0, (4, 3)), color=GRAY, lw=0.9, zorder=2)
        ax.plot(px, py, "o", ms=4, color=BLUE, zorder=5)
    marks = [(p, "x", side) for p, side in xmarks.items()] + [(v, "y", side) for v, side in ymarks.items()]
    place_axis_labels(fig, ax, marks)
    return fig


def sinh_ma_tikz(y_sym, x_sym, nghiem_thuc, nghiem_mau, tiem_can_y):
    frame = tinh_toan_khung_do_thi(y_sym, x_sym, nghiem_thuc, nghiem_mau, tiem_can_y)
    xmin, xmax, ymin, ymax = frame[:4]
    # Kích thước luôn gọn trong trang A4; không cắt mất điểm cực trị.
    sx, sy = 12 / (xmax-xmin), 7 / (ymax-ymin)
    coord = lambda a, b: f"({float(a):.8f},{float(b):.8f})"
    out = [r"% Preamble: \usepackage{tikz}",
           r"% Duong cong lay mau so; nhan toa do van giu gia tri chinh xac.",
           r"\begin{center}", fr"\begin{{tikzpicture}}[>=stealth,x={sx:.6f}cm,y={sy:.6f}cm]",
           r"\begin{scope}", fr"\clip ({xmin},{ymin}) rectangle ({xmax},{ymax});"]
    for p in nghiem_mau:
        lims = [sp.limit(y_sym, x_sym, p, dir=d) for d in ("-", "+")]
        if any(v in (sp.oo, -sp.oo) for v in lims):
            out.append(r"\draw[dashed,gray] " + coord(p, ymin) + " -- " + coord(p, ymax) + ";")
    if tiem_can_y is not None:
        out.append(r"\draw[dashed,gray] " + coord(xmin, tiem_can_y.subs(x_sym, xmin)) +
                   " -- " + coord(xmax, tiem_can_y.subs(x_sym, xmax)) + ";")
    for seg in curve_segments(y_sym, x_sym, nghiem_mau, frame):
        # Bỏ bớt mẫu gần thẳng để mã gọn, giữ sai số trong hệ trục vật lý nhỏ.
        keep = [seg[0]]
        for p in seg[1:-1]:
            if np.linalg.norm((p - keep[-1]) * np.array([sx, sy])) >= 0.035:
                keep.append(p)
        keep.append(seg[-1])
        out.append(r"\draw[blue!75!black,line width=1pt,line join=round] plot coordinates {")
        out.extend("  " + " ".join(coord(*p) for p in keep[j:j+5]) for j in range(0, len(keep), 5))
        out.append("};")
    for p in nghiem_mau:
        value = sp.limit(y_sym, x_sym, p)
        if value.is_finite:
            out.append(r"\draw[blue!75!black,fill=white] " + coord(p, value) + " circle (2.2pt);")
    out += [r"\end{scope}",
            fr"\draw[->] ({xmin},0) -- ({xmax},0) node[right] {{$x$}};",
            fr"\draw[->] (0,{ymin}) -- (0,{ymax}) node[above] {{$y$}};",
            r"\node[below left,fill=white,inner sep=1pt] at (0,0) {$O$};"]
    # Vạch chia dùng dịch chuyển pt, tránh trộn đơn vị vào tọa độ.
    for value in MaxNLocator(nbins=10, integer=True).tick_values(xmin, xmax):
        if xmin < value < xmax and value != 0:
            out.append(fr"\draw ({value:g},0) ++(0,-2pt) -- ++(0,4pt);")
    for value in MaxNLocator(nbins=10, integer=True).tick_values(ymin, ymax):
        if ymin < value < ymax and value != 0:
            out.append(fr"\draw (0,{value:g}) ++(-2pt,0) -- ++(4pt,0);")
    points, xmarks, ymarks = graph_marks(y_sym, x_sym, nghiem_thuc, nghiem_mau, tiem_can_y)
    for p, v in points:
        if p != 0 and v != 0:
            out.append(r"\draw[dashed,gray,thin] " + coord(p, 0) + " -- " + coord(p, v) + " -- " + coord(0, v) + ";")
        out.append(r"\fill[blue!75!black] " + coord(p, v) + " circle (1.5pt);")
    for marks, axis in ((xmarks, "x"), (ymarks, "y")):
        last = {}
        for value, side in sorted(marks.items(), key=lambda pair: float(pair[0])):
            physical = float(value) * (sx if axis == "x" else sy)
            prev, tier = last.get(side, (-1e100, 0))
            tier = (tier + 1) % 3 if physical - prev < 0.8 else 0
            last[side] = (physical, tier)
            offset = 5 + tier * 12
            pos = coord(value, 0) if axis == "x" else coord(0, value)
            out.append(fr"\node[{side}={offset}pt,fill=white,inner sep=1.5pt] at {pos} {{$ {tex(value)} $}};")
    if tiem_can_y is not None and len(nghiem_mau) == 1:
        p = nghiem_mau[0]
        v = sp.simplify(tiem_can_y.subs(x_sym, p))
        if sp.simplify(y_sym + y_sym.subs(x_sym, 2*p-x_sym) - 2*v) == 0:
            out.append(r"\fill " + coord(p, v) + r" circle (1.4pt) node[above right=4pt,fill=white,inner sep=1pt] {$I$};")
    return "\n".join(out + [r"\end{tikzpicture}", r"\end{center}"])


def png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=230, bbox_inches="tight", facecolor="white", pad_inches=0.18)
    plt.close(fig)
    return buf.getvalue()


# ==========================================
# 4. GIAO DIỆN STREAMLIT — GIỮ NGUYÊN BỐN PHẦN
# ==========================================
CSS = """
<style>
.stApp {background:linear-gradient(180deg,#f0f6ff 0,#f8fafc 430px); color:#172b4d;}
[data-testid="stHeader"] {background:rgba(248,250,252,.9);}
[data-testid="stSidebar"] {background:#fff; border-right:1px solid #e2e8f0;}
.block-container {max-width:1100px; padding-top:6rem !important; padding-bottom:3rem;}
h1,h2,h3 {color:#16325c !important; letter-spacing:-.025em;}
h3 {font-size:1.25rem !important;}
[data-testid="stVerticalBlockBorderWrapper"] > div {border-radius:18px !important;}
[data-testid="stForm"] {background:#fff; border:1px solid #dbe7f7; border-radius:18px; padding:1.4rem;}
[data-testid="stTextInput"] input {font-size:1.05rem;}
[data-testid="stTextInput"] label {color:#172b4d;}
.stButton>button, .stDownloadButton>button {border-radius:10px;}
[data-testid="stFormSubmitButton"] button {border-radius:11px; background:#2563eb; color:white; border:0; padding:.55rem 1.6rem;}
[data-testid="stFormSubmitButton"] button:hover {background:#1d4ed8; color:white;}
[data-testid="stImage"] {background:white; border-radius:13px; overflow:hidden;}
[data-testid="stLatex"] {overflow-x:auto; padding:.5rem 0;}
.hero {padding:0 0 1.4rem;}
.eyebrow {color:#2563eb; font-size:.76rem; font-weight:750; letter-spacing:.15em; margin-bottom:12px;}
.hero h1 {font-size:2.35rem; line-height:1.2; margin:0 0 12px; padding:0; font-weight:760;}
.hero p {color:#64748b; font-size:1rem; margin:0; line-height:1.65;}
.author {padding:16px 15px; background:#eff6ff; border:1px solid #dbeafe; border-radius:14px; line-height:1.85; color:#334155; font-size:.91rem;}
.author strong {color:#1e40af;}
.section-label {font-size:.74rem; letter-spacing:.11em; color:#2563eb; font-weight:700; margin:0 0 3px;}
.footer {color:#64748b; font-size:.8rem; text-align:center; padding:26px 0 5px;}
@media(max-width:640px) {.block-container {padding:6rem 1rem 3rem;} .hero h1 {font-size:1.8rem;}}
</style>
"""


def section(k, title):
    st.markdown(f'<div class="section-label">PHẦN {k}</div>', unsafe_allow_html=True)
    st.subheader(title)


def main():
    st.set_page_config(page_title="Khảo sát hàm số | Trường Vũ", page_icon="📈", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### Thông tin tác giả")
        st.markdown('''<div class="author"><strong>Nguyễn Bùi Trường Vũ (Chính)</strong><br>
        Nguyễn Lương Lâm Sơn<br>Võ Đăng Khoa<br>Hoàng Kim Gia Bảo<br><br>
        <b>SĐT:</b> 0854085229<br><b>Đơn vị:</b> Sinh viên năm 4 Khoa Toán DHS</div>''', unsafe_allow_html=True)
        st.divider()
        st.markdown("### Hướng dẫn nhập hàm số")
        st.markdown("- **Lũy thừa:** `x**3` hoặc `x^3`\n- **Nhân, chia:** `*`, `/`\n- **Phân thức:** đặt tử và mẫu trong `()`\n- **Hệ số căn:** `sqrt(2)*x`")
        st.caption("Một số ví dụ")
        st.code("x**3 - 3*x**2 + 2\n-x**3 + 3*x**2 - 3*x - 1\n(x + 2)/(x - 1)", language="python")
        st.info("Phạm vi khảo sát: hàm đa thức và hàm phân thức hữu tỉ.")
    st.markdown('''<div class="hero"><div class="eyebrow">TOÁN THPT · KHẢO SÁT HÀM SỐ</div>
    <h1>Hệ Thống Khảo Sát<br>Sự Biến Thiên</h1>
    <p>Khảo sát từng bước · Bảng biến thiên rõ ràng · Đồ thị và mã LaTeX</p></div>''', unsafe_allow_html=True)
    with st.form("function_form"):
        source = st.text_input("Nhập hàm số y = f(x)", "-x**3 + 3*x**2 - 3*x - 1")
        st.caption("Ví dụ: (x + 2)/(x - 1). Nhấn Enter hoặc nút bên dưới để khảo sát.")
        submitted = st.form_submit_button("Bắt Đầu Khảo Sát", type="primary")
    if submitted:
        try:
            with st.spinner("Đang tính toán và dựng hình…"):
                result = survey(source)
                y, yp = result["y"], result["derivative"]
                pts, excluded, asym = result["stationary"], result["excluded"], result["asymptote"]
                with plt.rc_context(PLOT_STYLE):
                    result["bbt_png"] = png_bytes(ve_bang_bien_thien(y, yp, X, pts, excluded))
                    result["graph_png"] = png_bytes(ve_do_thi_sgk(y, X, pts, excluded, asym))
                result["bbt_tex"] = sinh_ma_bbt_tikz(y, yp, X, pts, excluded)
                result["graph_tex"] = sinh_ma_tikz(y, X, pts, excluded, asym)
                st.session_state["survey_result"] = result
        except Exception as exc:
            st.session_state.pop("survey_result", None)
            st.error(f"Chưa thể khảo sát biểu thức này. {exc}")
    result = st.session_state.get("survey_result")
    if result is None:
        st.caption("Kết quả khảo sát, bảng biến thiên, đồ thị và mã LaTeX sẽ hiển thị tại đây.")
        return
    y, yp, pts, excluded = (result[k] for k in ("y", "derivative", "stationary", "excluded"))
    st.latex("y = " + tex(y))
    with st.container(border=True):
        section("I", "Tập xác định và sự biến thiên")
        st.markdown("**1. Tập xác định**")
        st.latex(r"D = \mathbb{R}" + (r"\setminus\left\{" + ", ".join(tex(p) for p in excluded) + r"\right\}" if excluded else ""))
        st.markdown("**2. Giới hạn và tiệm cận**")
        st.latex(r"\lim_{x\to-\infty}f(x)=" + tex(sp.limit(y, X, -sp.oo)) +
                 r",\qquad \lim_{x\to+\infty}f(x)=" + tex(sp.limit(y, X, sp.oo)))
        for p in excluded:
            st.latex(r"\lim_{x\to(" + tex(p) + r")^-}f(x)=" + tex(sp.limit(y, X, p, dir="-")) +
                     r",\qquad \lim_{x\to(" + tex(p) + r")^+}f(x)=" + tex(sp.limit(y, X, p, dir="+")))
        if result["poles"]:
            st.write("Tiệm cận đứng:")
            st.latex(r"\quad;\quad".join("x=" + tex(p) for p in result["poles"]))
        if result["asymptote"] is not None:
            st.write(f'Tiệm cận {result["kind"]}:')
            st.latex("y=" + tex(result["asymptote"]))
        st.markdown("**3. Đạo hàm**")
        st.latex("y'=" + tex(yp))
        if yp == 0:
            st.latex(r"y'=0\quad\text{với mọi }x\in D.")
        elif pts:
            st.latex(r"y'=0\iff x\in\left\{" + ", ".join(tex(p) for p in pts) + r"\right\}.")
        else:
            st.latex(r"y'=0\quad\text{vô nghiệm trên }D.")
        st.markdown("**4. Giao điểm với các trục tọa độ**")
        if sp.S.Zero in excluded:
            st.write("Đồ thị không cắt trục Oy vì 0 không thuộc tập xác định.")
        else:
            st.latex(r"Oy:\quad\left(0;" + tex(y.subs(X, 0)) + r"\right).")
        num, _ = sp.fraction(sp.together(y))
        roots = [p for p in real_roots(num, X) if p not in excluded]
        if y == 0:
            st.write("Đồ thị nằm trên trục Ox tại các điểm có hoành độ thuộc D.")
        elif roots:
            intersections, approximations = ox_intersections_latex(roots)
            st.latex(intersections)
            if approximations:
                st.latex(approximations)
                st.caption("Hoành độ gần đúng được làm tròn đến 6 chữ số thập phân.")
        else:
            st.write("Đồ thị không cắt trục Ox.")
    with st.container(border=True):
        section("II", "Bảng biến thiên")
        st.image(result["bbt_png"], width="stretch")
        if any(p not in excluded and yp.subs(X, p) == 0 for p in pts):
            st.caption("Nghiệm của đạo hàm chỉ là điểm cực trị khi đạo hàm đổi dấu qua điểm đó.")
        st.download_button("Tải bảng biến thiên PNG", result["bbt_png"], "bang_bien_thien.png", "image/png", key="download_bbt")
    with st.container(border=True):
        section("III", "Đồ thị mô phỏng")
        st.image(result["graph_png"], width="stretch")
        st.download_button("Tải đồ thị PNG", result["graph_png"], "do_thi.png", "image/png", key="download_graph")
    with st.container(border=True):
        section("IV", "Xuất mã LaTeX")
        tab1, tab2 = st.tabs(["Bảng biến thiên · tkz-tab", "Đồ thị · TikZ"])
        with tab1:
            st.caption("Thêm vào phần khai báo đầu tài liệu:")
            st.code(r"\usepackage{tikz,tkz-tab}", language="latex")
            st.code(result["bbt_tex"], language="latex")
            st.download_button("Tải mã bảng biến thiên", result["bbt_tex"], "bang_bien_thien.tex", "text/plain", key="tex_bbt")
        with tab2:
            st.caption("Thêm vào phần khai báo đầu tài liệu:")
            st.code(r"\usepackage{tikz}", language="latex")
            st.code(result["graph_tex"], language="latex")
            st.download_button("Tải mã đồ thị", result["graph_tex"], "do_thi.tex", "text/plain", key="tex_graph")
    st.markdown('<div class="footer">Nguyễn Bùi Trường Vũ & cộng sự · Khảo sát hàm số THPT</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
