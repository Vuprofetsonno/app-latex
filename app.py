import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import re
import io # Thêm thư viện io để xử lý ảnh tĩnh

# ==========================================
# 1. HÀM VẼ BẢNG BIẾN THIÊN TRÊN WEB (An toàn với KaTeX - Không dùng multicolumn)
# ==========================================
def tao_bang_bien_thien_latex(y_sym, y_prime, x_sym, nghiem_thuc, nghiem_mau):
    diem = sorted(list(set(nghiem_thuc + nghiem_mau)))
    n_cols = 2 * len(diem) + 4
    test_pts = [0] if not diem else [diem[0] - 1] + [(diem[i] + diem[i+1])/2 for i in range(len(diem)-1)] + [diem[-1] + 1]

    dau_yp = []
    for tp in test_pts:
        val = y_prime.subs(x_sym, tp)
        dau_yp.append("+" if val > 0 else ("-" if val < 0 else "0"))

    def fmt_y_val(val, height):
        if val == sp.oo: s = r"+\infty"
        elif val == -sp.oo: s = r"-\infty"
        else: s = sp.latex(sp.simplify(val))

        if height == "HIGH":
            return fr"\begin{{matrix}} {s} \\ \phantom{{0}} \\ \phantom{{0}} \end{{matrix}}"
        elif height == "LOW":
            return fr"\begin{{matrix}} \phantom{{0}} \\ \phantom{{0}} \\ {s} \end{{matrix}}"
        else:
            return fr"\begin{{matrix}} \phantom{{0}} \\ {s} \\ \phantom{{0}} \end{{matrix}}"

    row_x, row_yp, row_y = [""] * n_cols, [""] * n_cols, [""] * n_cols
    row_x[0], row_yp[0], row_y[0] = "x", "y'", "y"

    # Cực trái
    row_x[1] = r"-\infty"
    ht_am = "LOW" if dau_yp[0] == "+" else "HIGH"
    lim_am = sp.limit(y_sym, x_sym, -sp.oo)
    row_y[1] = fmt_y_val(lim_am, ht_am)

    # Cực phải
    row_x[-1] = r"+\infty"
    ht_duong = "HIGH" if dau_yp[-1] == "+" else "LOW"
    lim_duong = sp.limit(y_sym, x_sym, sp.oo)
    row_y[-1] = fmt_y_val(lim_duong, ht_duong)

    # Các khoảng mũi tên
    for i in range(len(diem) + 1):
        col_idx = 2 + 2 * i
        row_yp[col_idx] = fr"\quad {dau_yp[i]} \quad"
        arrow = r"\nearrow" if dau_yp[i] == "+" else (r"\searrow" if dau_yp[i] == "-" else r"\rightarrow")
        row_y[col_idx] = fr"\begin{{matrix}} \phantom{{0}} \\ {arrow} \\ \phantom{{0}} \end{{matrix}}"

    # Các điểm tới hạn / Tiệm cận
    for i in range(len(diem)):
        col_idx = 3 + 2 * i
        pt = diem[i]
        row_x[col_idx] = sp.latex(sp.together(pt))
        prev_sign = dau_yp[i]
        next_sign = dau_yp[i+1]

        if pt in nghiem_mau:
            row_yp[col_idx] = r"\Vert"
            lim_trai = sp.limit(y_sym, x_sym, pt, dir='-')
            ht_trai = "HIGH" if prev_sign == "+" else "LOW"
            s_trai = fmt_y_val(lim_trai, ht_trai)
            
            lim_phai = sp.limit(y_sym, x_sym, pt, dir='+')
            ht_phai = "LOW" if next_sign == "+" else "HIGH"
            s_phai = fmt_y_val(lim_phai, ht_phai)

            s_vert = r"\begin{matrix} \Vert \\ \Vert \\ \Vert \end{matrix}"
            row_y[col_idx] = fr"{s_trai} \,\, {s_vert} \,\, {s_phai}"
        else:
            row_yp[col_idx] = "0"
            val_pt = y_sym.subs(x_sym, pt)
            
            # Xử lý độ cao neo (Cao, Thấp, hoặc Bậc thang ở giữa)
            if prev_sign == "+" and next_sign == "-": ht = "HIGH"
            elif prev_sign == "-" and next_sign == "+": ht = "LOW"
            else: ht = "MID"
            
            row_y[col_idx] = fmt_y_val(val_pt, ht)

    cols_format = "|c|" + "c" * (n_cols - 1) + "|"
    latex_str = (
        r"\begin{array}{" + cols_format + r"} \hline " + "\n" +
        " & ".join(row_x) + r" \\ \hline " + "\n" +
        " & ".join(row_yp) + r" \\ \hline " + "\n" +
        " & ".join(row_y) + r" \\ \hline " + "\n" +
        r"\end{array}"
    )
    return latex_str


# ==========================================
# 2. HÀM XUẤT MÃ LATEX BẢNG BIẾN THIÊN (TKZ-TAB CHUẨN OVERLEAF)
# ==========================================
def sinh_ma_bbt_tikz(y_sym, y_prime, x_sym, nghiem_thuc, nghiem_mau):
    diem = sorted(list(set(nghiem_thuc + nghiem_mau)))
    
    test_pts = []
    if not diem:
        test_pts.append(0)
    else:
        test_pts.append(float(diem[0]) - 1)
        for i in range(len(diem) - 1):
            test_pts.append(float(diem[i] + diem[i+1]) / 2)
        test_pts.append(float(diem[-1]) + 1)
        
    dau_yp = []
    for tp in test_pts:
        val = y_prime.subs(x_sym, tp)
        dau_yp.append("+" if val > 0 else "-")
        
    def fmt(val):
        if val == sp.oo: return r"\(+\infty\)"
        if val == -sp.oo: return r"\(-\infty\)"
        return f"\({sp.latex(sp.simplify(val))}\)"

    x_row_items = [r"-\infty"] + [sp.latex(sp.together(d)) for d in diem] + [r"+\infty"]
    x_str = "{" + ", ".join([f"\({item}\)" for item in x_row_items]) + "}"
    
    tkz_tab_line = []
    for i, pt in enumerate(diem):
        tkz_tab_line.append(dau_yp[i])
        if pt in nghiem_mau:
            tkz_tab_line.append("d")
        else:
            tkz_tab_line.append("0")
    tkz_tab_line.append(dau_yp[-1])
    y_prime_str = "\\tkzTabLine{ , " + ", ".join(tkz_tab_line) + ", }"
    
    tkz_tab_var = []
    val_inf_am = sp.limit(y_sym, x_sym, -sp.oo)
    tkz_tab_var.append(f"-/ {fmt(val_inf_am)}" if dau_yp[0] == '+' else f"+/ {fmt(val_inf_am)}")
        
    for i, pt in enumerate(diem):
        dau_truoc = dau_yp[i]
        dau_sau = dau_yp[i+1]
        
        if pt in nghiem_mau:
            lim_trai = sp.limit(y_sym, x_sym, pt, dir='-')
            lim_phai = sp.limit(y_sym, x_sym, pt, dir='+')
            pos_trai = "+" if dau_truoc == "+" else "-"
            pos_phai = "-" if dau_sau == "+" else "+"
            tkz_tab_var.append(f"{pos_trai}D{pos_phai}/ {fmt(lim_trai)} / {fmt(lim_phai)}")
        else:
            val_pt = y_sym.subs(x_sym, pt)
            if dau_truoc == "+" and dau_sau == "-":
                tkz_tab_var.append(f"+/ {fmt(val_pt)}")
            elif dau_truoc == "-" and dau_sau == "+":
                tkz_tab_var.append(f"-/ {fmt(val_pt)}")
            else:
                tkz_tab_var.append(f"R/ {fmt(val_pt)}")
                
    val_inf_duong = sp.limit(y_sym, x_sym, sp.oo)
    tkz_tab_var.append(f"+/ {fmt(val_inf_duong)}" if dau_yp[-1] == '+' else f"-/ {fmt(val_inf_duong)}")
        
    y_var_str = "\\tkzTabVar{" + ", ".join(tkz_tab_var) + "}"
    
    latex_code = (
        "\\begin{center}\n"
        "\\begin{tikzpicture}\n"
        "    % Tùy chỉnh mũi tên chuẩn Toán (stealth) và nới rộng khoảng cách dấu ||\n"
        "    \\tikzset{>=stealth, double distance=2pt}\n"
        "    \\tkzTabInit[nocadre=false, lgt=1.5, espcl=3.5, deltacl=0.8]\n"
        "      {\(x\) / 1.0, \(y'\) / 1.0, \(y\) / 2.5}\n"
        f"      {x_str}\n"
        f"    {y_prime_str}\n"
        f"    {y_var_str}\n"
        "\\end{tikzpicture}\n"
        "\\end{center}"
    )
    return latex_code


# ==========================================
# 3. CÁC HÀM XỬ LÝ VẼ ĐỒ THỊ VÀ XUẤT MÃ TIKZ
# ==========================================
def tinh_toan_khung_do_thi(y_sym, x_sym, nghiem_thuc, nghiem_mau, tiem_can_y):
    x_pts = [float(n) for n in nghiem_thuc if n.is_real] + [0]
    
    if nghiem_mau:
        tc_x = float(nghiem_mau[0])
        x_pts.extend([tc_x - 4, tc_x + 4])
        
    x_min_crit, x_max_crit = min(x_pts), max(x_pts)
    padding_x = 1.5
    x_min = int(np.floor(x_min_crit - padding_x))
    x_max = int(np.ceil(x_max_crit + padding_x))

    y_pts = []
    for c in nghiem_thuc:
        if c.is_real and c not in nghiem_mau:
            y_pts.append(float(y_sym.subs(x_sym, c)))

    if nghiem_mau:
        tc_x = float(nghiem_mau[0])
        for dx in [-2, -1, 1, 2]:
            val = y_sym.subs(x_sym, tc_x + dx)
            if val.is_real:
                y_pts.append(float(val))
        if tiem_can_y is not None:
            y_pts.append(float(tiem_can_y.subs(x_sym, tc_x)))

    if not y_pts:
        y_pts = [-2, 2] 
        
    y_min_crit, y_max_crit = min(y_pts), max(y_pts)
    y_range = y_max_crit - y_min_crit
    padding_y = max(3.0, y_range * 0.3)
    
    y_min = int(np.floor(y_min_crit - padding_y))
    y_max = int(np.ceil(y_max_crit + padding_y))
    
    if y_max - y_min > 20:
        if nghiem_mau and tiem_can_y is not None:
            y_i = float(tiem_can_y.subs(x_sym, float(nghiem_mau[0])))
            y_min = int(np.floor(y_i - 8))
            y_max = int(np.ceil(y_i + 8))
        else:
            y_center = (y_max + y_min) / 2
            y_min = int(np.floor(y_center - 10))
            y_max = int(np.ceil(y_center + 10))
            
    y_range_total = y_max - y_min
    if y_range_total <= 8:
        y_step = 1
    elif y_range_total <= 16:
        y_step = 2
    elif y_range_total <= 25:
        y_step = 4
    else:
        y_step = 5

    return x_min, x_max, y_min, y_max, y_step, y_range_total

def ve_do_thi_sgk(y_sym, x_sym, nghiem_thuc, nghiem_mau, tiem_can_y):
    x_min, x_max, y_min, y_max, y_step, _ = tinh_toan_khung_do_thi(y_sym, x_sym, nghiem_thuc, nghiem_mau, tiem_can_y)
    
    f = sp.lambdify(x_sym, y_sym, modules=['numpy'])
    xs = np.linspace(x_min - 1, x_max + 1, 600)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ys = f(xs)
        for tc in nghiem_mau:
            ys[np.abs(xs - float(tc)) < 0.15] = np.nan
            
        ys[ys > y_max + 3] = np.nan
        ys[ys < y_min - 3] = np.nan

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ys, 'b-', linewidth=1.8)
    
    for tc in nghiem_mau:
        ax.axvline(float(tc), color='k', linestyle='--', linewidth=1, alpha=0.6)
        
    if tiem_can_y is not None:
        f_asy = sp.lambdify(x_sym, tiem_can_y, modules=['numpy'])
        y_asy = f_asy(xs)
        if np.isscalar(y_asy):
            y_asy = np.full_like(xs, y_asy)
        ax.plot(xs, y_asy, 'k--', linewidth=1, alpha=0.6)
        
        if nghiem_mau:
            tc_val = float(nghiem_mau[0])
            y_i = float(tiem_can_y.subs(x_sym, tc_val))
            ax.plot(tc_val, y_i, 'ko', markersize=4)
            # Khôi phục nhãn \(I\) chuẩn Toán học vì đã có ảnh tĩnh bảo vệ
            ax.annotate('\(I\)', xy=(tc_val, y_i), xytext=(8, 8), textcoords='offset points', fontsize=12)

    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    
    x_ticks = np.arange(x_min, x_max + 1, 1)
    y_ticks = np.arange(y_min, y_max + 1, y_step)
    
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([]) 
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', length=4, direction='inout', colors='black')

    ax.plot((1), (0), ls="", marker=">", ms=6, color="k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot((0), (1), ls="", marker="^", ms=6, color="k", transform=ax.get_xaxis_transform(), clip_on=False)
    ax.text(x_max - 0.2, -0.4, 'x', fontsize=12, fontstyle='italic')
    ax.text(-0.3, y_max - 0.2, 'y', fontsize=12, fontstyle='italic')
    ax.text(-0.25, -0.35, 'O', fontsize=12, fontstyle='italic')

    bbox_props = dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.5)

    for xt in nghiem_thuc:
        x_val = float(xt)
        y_val = float(y_sym.subs(x_sym, xt))
        
        if abs(x_val) > 0.05 and abs(y_val) > 0.05:
            ax.plot([x_val, x_val], [0, y_val], 'k--', linewidth=0.8)
            ax.plot([0, x_val], [y_val, y_val], 'k--', linewidth=0.8)
            ax.plot(x_val, 0, 'ko', markersize=3)
            ax.plot(0, y_val, 'ko', markersize=3)

        if abs(x_val) > 0.05:
            x_latex = sp.latex(sp.together(xt))
            ax.annotate(f'\({x_latex}\)', xy=(x_val, 0), xytext=(0, -22), textcoords='offset points', 
                        ha='center', va='center', fontsize=11, bbox=bbox_props)

        if abs(y_val) > 0.05:
            y_latex = sp.latex(sp.together(sp.simplify(y_sym.subs(x_sym, xt))))
            ax.annotate(f'\({y_latex}\)', xy=(0, y_val), xytext=(-24, 0), textcoords='offset points', 
                        ha='center', va='center', fontsize=11, bbox=bbox_props)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    return fig

def sinh_ma_tikz(y_sym, x_sym, nghiem_thuc, nghiem_mau, tiem_can_y):
    x_min, x_max, y_min, y_max, y_step, y_range_total = tinh_toan_khung_do_thi(y_sym, x_sym, nghiem_thuc, nghiem_mau, tiem_can_y)
    
    y_scale = round(6.0 / max(5, y_range_total), 2)
    y_scale = max(0.3, min(1.0, y_scale))
    
    x_ticks = ",".join([str(i) for i in range(x_min, x_max + 1) if i != 0])
    y_ticks = ",".join([str(i) for i in range(y_min, y_max + 1, y_step) if i != 0])
    
    tikz = "\\begin{center}\n"
    tikz += f"\\begin{{tikzpicture}}[>=stealth, x=1.2cm, y={y_scale}cm]\n"
    tikz += f"    \\draw[->, line width=0.8pt] ({x_min - 0.5},0) -- ({x_max + 0.5},0) node[below] {{\(x\)}};\n"
    tikz += f"    \\draw[->, line width=0.8pt] (0,{y_min - 0.5}) -- (0,{y_max + 0.5}) node[left] {{\(y\)}};\n"
    tikz += f"    \\fill (0,0) circle (1.5pt);\n"
    tikz += f"    \\node[below right] at (0,0) {{\(O\)}};\n"
    
    tikz += f"    \n    % Vạch chia\n"
    tikz += f"    \\foreach \\x in {{{x_ticks}}} {{\\draw[line width=0.6pt] (\\x, -2.5pt) -- (\\x, 2.5pt);}}\n"
    tikz += f"    \\foreach \\y in {{{y_ticks}}} {{\\draw[line width=0.6pt] (-2.5pt, \\y) -- (2.5pt, \\y);}}\n"
    
    if nghiem_mau or tiem_can_y is not None:
        tikz += f"    \n    % Asymptotes (Đường tiệm cận)\n"
        for tc in nghiem_mau:
            tc_val = float(tc)
            tikz += f"    \\draw[dashed, line width=0.6pt] ({tc_val:.3f},{y_min-0.5}) -- ({tc_val:.3f},{y_max+0.5});\n"
            
        if tiem_can_y is not None:
            y1 = float(tiem_can_y.subs(x_sym, x_min - 0.5))
            y2 = float(tiem_can_y.subs(x_sym, x_max + 0.5))
            tikz += f"    \\draw[dashed, line width=0.6pt] ({x_min - 0.5:.3f},{y1:.3f}) -- ({x_max + 0.5:.3f},{y2:.3f});\n"
            
            if nghiem_mau:
                tc_val = float(nghiem_mau[0])
                y_i = float(tiem_can_y.subs(x_sym, tc_val))
                tikz += f"    \\fill ({tc_val:.3f},{y_i:.3f}) circle (1.5pt) node[above right] {{\(I\)}};\n"

    tikz += f"    \n    % Điểm cực trị\n"
    for xt in nghiem_thuc:
        x_val = float(xt)
        y_val = float(y_sym.subs(x_sym, xt))
        x_c = f"{x_val:.4f}"
        y_c = f"{y_val:.4f}"
        x_latex = f"\({sp.latex(sp.together(xt))}\)"
        y_latex = f"\({sp.latex(sp.together(sp.simplify(y_sym.subs(x_sym, xt))))}\)"
        
        if abs(x_val) > 0.05 and abs(y_val) > 0.05:
            tikz += f"    \\draw[dashed, line width=0.5pt] ({x_c},0) -- ({x_c},{y_c}) -- (0,{y_c});\n"
        if abs(x_val) > 0.05:
            tikz += f"    \\fill ({x_c},0) circle (1.5pt);\n"
            tikz += f"    \\node[{'above' if y_val < 0 else 'below'}, fill=white, inner sep=1.5pt] at ({x_c}, {'6pt' if y_val < 0 else '-6pt'}) {{{x_latex}}};\n"
        if abs(y_val) > 0.05:
            tikz += f"    \\fill (0,{y_c}) circle (1.5pt);\n"
            tikz += f"    \\node[{'right' if x_val < 0 else 'left'}, fill=white, inner sep=1.5pt] at ({'6pt' if x_val < 0 else '-6pt'}, {y_c}) {{{y_latex}}};\n"

    expr_str = str(y_sym).replace('**', '^')
    bieu_thuc_tikz = re.sub(r'\bx\b', r'(\\x)', expr_str)
    
    tikz += f"    \n    % Graph (Sử dụng scope + clip để tự động cắt gọn trong vùng trục)\n"
    tikz += f"    \\begin{{scope}}\n"
    tikz += f"        \\clip ({x_min - 0.2}, {y_min - 0.2}) rectangle ({x_max + 0.2}, {y_max + 0.2});\n"
    
    if nghiem_mau:
        tc_val = float(nghiem_mau[0])
        tikz += f"        \\draw[line width=1.2pt, smooth, samples=100, domain={x_min - 1}:{tc_val - 0.15}] plot (\\x, {{{bieu_thuc_tikz}}});\n"
        tikz += f"        \\draw[line width=1.2pt, smooth, samples=100, domain={tc_val + 0.15}:{x_max + 1}] plot (\\x, {{{bieu_thuc_tikz}}});\n"
    else:
        tikz += f"        \\draw[line width=1.2pt, smooth, samples=200, domain={x_min - 1}:{x_max + 1}] plot (\\x, {{{bieu_thuc_tikz}}});\n"
        
    tikz += f"    \\end{{scope}}\n"
    tikz += f"\\end{{tikzpicture}}\n"
    tikz += f"\\end{{center}}"
    
    return tikz


# ==========================================
# 4. CẤU HÌNH GIAO DIỆN CHÍNH STREAMLIT
# ==========================================
st.set_page_config(page_title="Khảo Sát Hàm Số Tự Động", layout="centered")

with st.sidebar:
    st.markdown("### 👨‍💻 Thông tin tác giả")
    st.markdown("**Nguyễn Bùi Trường Vũ (Chính)**")
    st.markdown("**Nguyễn Lương Lâm Sơn**")
    st.markdown("**Võ Đăng Khoa**")
    st.markdown("**Hoàng Kim Gia Bảo**")
    st.markdown("📞 **SĐT:** 0854085229")
    st.markdown("🎓 **Đơn vị:** Sinh viên năm 4 Khoa Toán DHS")
    
    st.markdown("---")
    st.markdown("### 📝 Hướng dẫn nhập hàm số")
    st.info(
        "Nhập biểu thức theo cú pháp Python (SymPy):\n\n"
        "- **Lũy thừa (\(x^n\)):** Dùng `**` (vd: `x**3`)\n"
        "- **Phân thức:** Đặt tử và mẫu trong ngoặc tròn `()`\n"
        "- **Nhân/chia:** Dùng `*`, `/`\n"
        "- **Căn bậc hai:** `sqrt(x)`\n"
    )
    st.markdown("**Một số ví dụ:**")
    st.code("x**3 - 3*x**2 + 2", language="python")
    st.code("-x**3 + 3*x**2 - 3*x - 1", language="python")
    st.code("(x + 2)/(x - 1)", language="python")
    
    st.warning("Ứng dụng tối ưu tốt nhất cho hàm đa thức và hàm phân thức.")

st.title("Hệ Thống Khảo Sát Sự Biến Thiên")
st.markdown("*Ứng dụng hỗ trợ trình bày lời giải step-by-step chuẩn SGK Toán THPT.*")

bieu_thuc = st.text_input("Nhập hàm số y = f(x):", "-x**3 + 3*x**2 - 3*x - 1")

if st.button("Bắt Đầu Khảo Sát"):
    x = sp.Symbol('x', real=True)
    try:
        y = sp.sympify(bieu_thuc, locals={'x': x})
        tu, mau = sp.fraction(sp.together(y))
        nghiem_mau = sp.solve(mau, x) if mau != 1 else []
        y_prime = sp.factor(sp.simplify(sp.diff(y, x)))
        tu_y_prime, mau_y_prime = sp.fraction(y_prime)
        nghiem_thuc = [n for n in sp.solve(tu_y_prime, x) if n.is_real and n not in nghiem_mau]
        
        tiem_can_y = None
        loai_tc = ""
        if mau != 1:
            thuong, du = sp.div(tu, mau, domain=sp.QQ)
            if sp.degree(tu, x) <= sp.degree(mau, x):
                tiem_can_y = thuong
                loai_tc = "TCN"
            elif sp.degree(tu, x) == sp.degree(mau, x) + 1:
                tiem_can_y = thuong
                loai_tc = "TCX"

        # --- PHẦN I. TẬP XÁC ĐỊNH & GIỚI HẠN ---
        st.markdown("### I. TẬP XÁC ĐỊNH VÀ SỰ BIẾN THIÊN")
        
        if not nghiem_mau:
            txđ = r"D = \mathbb{R}"
        else:
            mau_str = r", ".join([sp.latex(m) for m in nghiem_mau])
            txđ = fr"D = \mathbb{{R}} \setminus \{{{mau_str}\}}"
        st.write("**1. Tập xác định:**")
        st.latex(txđ)

        st.write("**2. Giới hạn và Tiệm cận:**")
        def fmt_lim(lim):
            if lim == sp.oo: return r"+\infty"
            if lim == -sp.oo: return r"-\infty"
            return sp.latex(lim)

        if tiem_can_y is not None:
            tc_str = f"TCĐ: x = {sp.latex(nghiem_mau[0])}; \\quad {loai_tc}: y = {sp.latex(tiem_can_y)}"
            st.latex(tc_str)
        else:
            lim_am = sp.limit(y, x, -sp.oo)
            lim_duong = sp.limit(y, x, sp.oo)
            st.latex(f"\\lim_{{x \\to -\\infty}} y = {fmt_lim(lim_am)}; \\quad \\lim_{{x \\to +\\infty}} y = {fmt_lim(lim_duong)}")

        st.write("**3. Đạo hàm:**")
        st.latex(f"y' = {sp.latex(sp.simplify(sp.diff(y, x)))}")
        
        if nghiem_thuc:
            nghiem_str = r" \quad ".join([f"x = {sp.latex(sp.together(n))}" for n in nghiem_thuc])
            st.latex(f"y' = 0 \Leftrightarrow {nghiem_str}")
        else:
            st.latex(r"y' = 0 \text{ vô nghiệm}")

        st.write("**4. Giao điểm với trục toạ độ:**")
        giao_oy = y.subs(x, 0)
        nghiem_ox = [sp.latex(n) for n in sp.solve(tu, x) if n.is_real and n not in nghiem_mau]
        
        oy_str = f"x = 0 \Rightarrow y = {sp.latex(giao_oy)}" if not sp.sympify(0) in nghiem_mau else r"x = 0 \text{ không thuộc TXĐ}"
        ox_str = f"y = 0 \Rightarrow x \in \{{{', '.join(nghiem_ox)}\}}" if nghiem_ox else r"y = 0 \text{ vô nghiệm}"
        st.latex(fr"{oy_str} \quad ; \quad {ox_str}")

        # --- PHẦN II. BẢNG BIẾN THIÊN TRỰC QUAN ---
        st.markdown("### II. BẢNG BIẾN THIÊN")
        st.info("Lưu ý: Do giới hạn kỹ thuật của Web, bảng mô phỏng buộc phải ngắt mũi tên tại điểm uốn. Tuy nhiên mã xuất Overleaf (bên dưới) vẫn sẽ tự động nối thành 1 dải liên tục chuẩn SGK.")
        st.latex(tao_bang_bien_thien_latex(y, y_prime, x, nghiem_thuc, nghiem_mau))
        
        # --- PHẦN III. ĐỒ THỊ ---
        st.markdown("### III. ĐỒ THỊ MÔ PHỎNG")
        fig = ve_do_thi_sgk(y, x, nghiem_thuc, nghiem_mau, tiem_can_y)
        
        # FIXED: Render đồ thị ra bộ nhớ đệm (buffer) ảnh tĩnh PNG.
        # Khắc phục 100% việc Streamlit dịch nhầm các điểm tọa độ số thành \(\)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
        st.image(buf, use_container_width=True)
        
        # --- PHẦN IV. XUẤT MÃ LATEX / TIKZ ---
        st.markdown("### IV. XUẤT MÃ LATEX (CHÈN OVERLEAF)")
        st.info("Đã tích hợp mã Tkz-Tab (Bảng biến thiên) và mã TikZ (Đồ thị) chuẩn SGK.")
        
        tab1, tab2 = st.tabs(["📊 Mã Bảng Biến Thiên (tkz-tab)", "📈 Mã Đồ Thị (TikZ)"])
        
        with tab1:
            st.markdown("*Lưu ý: Bạn cần khai báo thư viện `\\usepackage{tkz-tab}` ở phần preamble của file LaTeX.*")
            st.code(sinh_ma_bbt_tikz(y, y_prime, x, nghiem_thuc, nghiem_mau), language='latex')
            
        with tab2:
            st.markdown("*Lưu ý: Bạn cần khai báo thư viện `\\usepackage{tikz}` ở phần preamble của file LaTeX.*")
            st.code(sinh_ma_tikz(y, x, nghiem_thuc, nghiem_mau, tiem_can_y), language='latex')

    except Exception as e:
        st.error(f"Đã xảy ra lỗi! Vui lòng kiểm tra lại cú pháp hàm số.\n\nChi tiết lỗi: {e}")
