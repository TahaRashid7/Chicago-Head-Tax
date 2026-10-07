import re
s = open("main_updated.tex", encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:80])
    s = s.replace(a, b)

# 0. fixes to the previous pass
s = s.replace("fig3\\_revenue_range", "fig3\\_revenue\\_range")

# 1. robust figslot
i = s.index("\\newcommand{\\figslot}[4]{%")
j = s.index("\\end{figure}}", i) + len("\\end{figure}}")
new_macro = r"""\newcommand{\figslot}[4]{%
  \gdef\figpath{}%
  \IfFileExists{figures/#1.pdf}{\gdef\figpath{figures/#1.pdf}}{%
  \IfFileExists{#1.pdf}{\gdef\figpath{#1.pdf}}{%
  \IfFileExists{figures/#1.png}{\gdef\figpath{figures/#1.png}}{%
  \IfFileExists{#1.png}{\gdef\figpath{#1.png}}{}}}}%
  \begin{figure}[H]
  \centering
  \ifx\figpath\empty
    \fcolorbox{red}{white}{\parbox[c][2.3in][c]{0.86\textwidth}{\centering
      \textcolor{red}{\textbf{FIGURE PLACEHOLDER}}\\[6pt]
      Looked for \texttt{\detokenize{figures/#1.pdf}} and \texttt{\detokenize{#1.pdf}}\\[6pt]
      \footnotesize\textcolor{cmfheadergray}{#4}}}%
  \else
    \includegraphics[width=0.92\textwidth]{\figpath}%
  \fi
  \caption{#2}
  \label{#3}
  \end{figure}}"""
s = s[:i] + new_macro + s[j:]
rep("% Upload the finished figure to Overleaf as figures/<file-stem>.pdf (or .png).",
    "% Upload the finished figure to Overleaf as figures/<file-stem>.pdf (or .png). The macro also\n% looks in the project root, so a file uploaded without a folder is still found.")

# 2. Table 3 cells
rep("2003--2011 & \\$4, threshold 50 & 1.74 & \\PEND{fill} & \\PEND{fill} \\\\", "2003--2011 & \\$4, threshold 50 & 1.74 & 1.39 & 1.01 \\\\")
rep("2012--2013 & Phase-out, \\$2 & 2.89 & \\PEND{fill} & \\PEND{fill} \\\\", "2012--2013 & Phase-out, \\$2 & 2.89 & 1.73 & 1.09 \\\\")
rep("2014--2022 & Repealed & 3.60 & \\PEND{fill} & \\PEND{fill} \\\\", "2014--2022 & Repealed & 3.60 & 2.07 & 1.57 \\\\")
rep("Excess mass at 49 employees ($R_{49}$), single-location observed sample}", "Excess mass at 49 employees ($R_{49}$), single-location observed sample, size 51 left out of the baseline}")

# 3. Table 5 counts
rep("Sizes 46--49 pooled vs suburban Cook & $+0.004$ & \\PEND{count} & 1.00 \\\\", "Sizes 46--49 pooled vs suburban Cook & $+0.004$ & 12 of 12 & 1.00 \\\\")
rep("Sizes 46--49 pooled vs Illinois ex Cook & $-0.179$ & \\PEND{count} & 0.67 \\\\", "Sizes 46--49 pooled vs Illinois ex Cook & $-0.179$ & 8 of 12 & 0.67 \\\\")

# 4. government name list
rep("by industry code (NAICS 92, or SIC 9xxx) and by a named list of public bodies that the vendor codes to their operating industry. \\PEND{append the name list}",
    "by industry code (NAICS 92, or SIC 9xxx) and by a named list of public bodies that the vendor codes to their operating industry (Table~\\ref{tab:govlist}).")
gov = r"""
\begin{table}[H]
\centering
\caption{Public bodies removed by name because the vendor codes them to an operating industry, 2025}
\label{tab:govlist}
\footnotesize
\begin{tabular}{@{}L{2.9in}L{1.55in}C{0.7in}C{0.8in}@{}}
\toprule
\textbf{Record (as named by the vendor)} & \textbf{Vendor industry} & \textbf{Sites} & \textbf{Employees} \\
\midrule
UI HEALTH & Health care & 49 & 5,919 \\
UNITED STATES POSTAL SVC & Transport & 66 & 5,161 \\
CITY COLLEGES-CHICAGO-HARRY S & Education & 14 & 4,078 \\
CITY-CHICAGO STREETS DEPT & Construction & 1 & 3,500 \\
CHICAGO TRANSIT AUTHORITY & Transport & 170 & 3,162 \\
CHICAGO PUBLIC LIBRARY & Information & 76 & 2,379 \\
CHICAGO TRANSIT AUTHORITY (second record) & Transport & 1 & 1,500 \\
CHICAGO O'HARE INTL AIRPORT & Transport & 1 & 1,301 \\
CHICAGO STATE UNIVERSITY & Education & 5 & 1,121 \\
MCCORMICK PLACE & Accommodation and food & 1 & 1,001 \\
METROPOLITAN PIER \& EXPOSITION & Administrative support & 1 & 1,000 \\
BOARD-TRUSTEES-CMNTY CLG DIST & Finance and insurance & 1 & 750 \\
FANTUS HEALTH CTR-COOK COUNTY & Health care & 1 & 750 \\
CHICAGO READ MENTAL HEALTH CTR & Health care & 1 & 628 \\
PROVIDENT HOSPITAL-COOK COUNTY & Health care & 11 & 609 \\
CHICAGO TRANSIT AUTHORITY (third record) & Transport & 1 & 600 \\
MCCORMICK PLACE (second record) & Administrative support & 1 & 500 \\
DEPARTMENT OF CULTURAL AFFAIRS & Arts and recreation & 1 & 500 \\
LANE TECH HIGH SCHOOL & Education & 1 & 500 \\
\midrule
\textbf{Total, 19 records} & & & \textbf{34,959} \\
\bottomrule
\end{tabular}

\vspace{2pt}
{\footnotesize Source: authors' calculations from the Data Axle Historical Business Database. A further 14 firms with 19,196 employees are removed by industry code (NAICS 92) and are not listed. Sites are Chicago establishments rolled up to the record.}
\end{table}
"""
k = s.index("\\item Apply Rule A (central base) and Rules B, C and the review flag")
k2 = s.index("\\end{enumerate}", k) + len("\\end{enumerate}")
s = s[:k2] + "\n" + gov + s[k2:]

# 5. drop slots we cannot support, fix names and captions of the rest
s = re.sub(r"\\figslot\{bunching_asymmetry\}.*?\n\n", "", s, count=1, flags=re.S)
s = re.sub(r"\\figslot\{bunching_event_study\}.*?\n\n", "", s, count=1, flags=re.S)
assert "bunching_asymmetry" not in s and "bunching_event_study" not in s
rep("\\figslot{heaping_profile}{Heaping in Chicago employer size, 2025: count by employee size for sizes 40 to 60, verified records and all records.}{fig:heaping}{Shows that 50 is a strong attractor but not the strongest round number. Distribution of establishments by exact size from 40 to 60, two series.}",
    "\\figslot{heaping_profile_2025}{Heaping across the Chicago size distribution, 2025: the count at each round number relative to the median count at neighbouring sizes.}{fig:heaping}{Bar chart of mass at 10, 20, 25, 30, 40, 50, 60, 75, 100 and 150 relative to neighbours. File: heaping\\_profile\\_2025.pdf.}")
rep("Chicago establishments by employment source, 2025. Left: overall split. Right: share verified by size band.", "Chicago establishments by employment source, 2025. Top: overall split between verified and modeled. Bottom: share verified by size band.")
rep("\\figslot{bunching_heatmap_chicago}{Chicago employer size distribution by year, sizes 40 to 60.}", "\\figslot{bunching_heatmap_chicago}{Chicago employer size distribution by year, sizes 30 to 70, 1997 to 2025. Colour shows the count at each size relative to a smooth trend fitted to sizes that are not multiples of five; the outlined column is 49.}")
rep("\\figslot{placebo_sizes}{Excess mass by period at placebo sizes and at 49, Chicago.}", "\\figslot{placebo_thresholds}{Clustering just below round numbers in Chicago, by year (top), and the tax test at each size with 95 percent intervals (bottom). Sizes other than 49 carry no Chicago tax.}")
rep("{fig:placebo-fig}{Small multiples for sizes 30, 35, 40, 45, 49, 55, 60 to 90. Same y-axis. Shows flat placebos against the rise at 49.}", "{fig:placebo-fig}{Top: series at 49, 99 and twelve placebo sizes. Bottom: difference-in-differences at each size. File: placebo\\_thresholds.pdf.}")

open("main_final.tex", "w", encoding="utf-8").write(s)
print("PEND left:", re.findall(r"\\PEND\{([^}]{0,60})", s))
print("em dashes:", s.count("\u2014"))
