import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from analysis import (
    load_any_csv, clean_data, detect_drops,
    get_stats, get_numeric_columns, get_text_columns
)
from prediction import run_full_prediction
from ai_recommendations import (
    get_why_drops, get_recommendations,
    get_insight_summary, test_api_key
)

COLORS = {
    'bg'           : '#F0F4F8',
    'header_bg'    : '#1A3C5E',
    'header_fg'    : '#FFFFFF',
    'btn_primary'  : '#2563EB',
    'btn_primary_fg': '#FFFFFF',
    'btn_success'  : '#16A34A',
    'btn_success_fg': '#FFFFFF',
    'btn_warning'  : '#D97706',
    'btn_warning_fg': '#FFFFFF',
    'panel_bg'     : '#FFFFFF',
    'panel_border' : '#CBD5E1',
    'text_dark'    : '#1E293B',
    'text_muted'   : '#64748B',
    'drop_red'     : '#DC2626',
    'success_green': '#15803D',
    'section_head' : '#1D4ED8',
    'output_bg'    : '#F8FAFC',
    'insight_bg'   : '#EFF6FF',
    'insight_border': '#3B82F6',
}

FONT_TITLE  = ('Arial', 14, 'bold')
FONT_HEADER = ('Arial', 11, 'bold')
FONT_NORMAL = ('Arial', 10)
FONT_SMALL  = ('Arial', 9)
FONT_MONO   = ('Courier', 10)
FONT_BTN    = ('Arial', 10, 'bold')


def _none(val):
    if not val or val == 'None':
        return None
    return val



class SalesDropApp:

    def __init__(self, root):
        self.root = root
        self.root.title(
            "AI-Based Sales Drop Analysis with Prediction & Smart Recommendations"
        )
        self.root.geometry("1300x900")
        self.root.minsize(1100, 750)
        self.root.configure(bg=COLORS['bg'])

       
        self.df_raw      = None
        self.df_clean    = None
        self.drops_df    = None
        self.all_df      = None
        self.stats       = {}
        self.pred_result = {}

        self.filepath    = tk.StringVar(value="No file selected")
        self.api_key_var = tk.StringVar()
        self.value_col   = tk.StringVar()
        self.group_col   = tk.StringVar(value="None")
        self.threshold   = tk.DoubleVar(value=10.0)
        self.n_future    = tk.IntVar(value=3)
        self.status_var  = tk.StringVar(
            value="Ready — please upload a CSV file to begin."
        )

        self._chart_fig    = None
        self._chart_canvas = None

        self._build_header()
        self._build_toolbar()
        self._build_main_area()
        self._build_status_bar()

   
    def _build_header(self):
        header = tk.Frame(
            self.root, bg=COLORS['header_bg'], height=65, relief='flat'
        )
        header.pack(fill='x', side='top')
        header.pack_propagate(False)

        tk.Label(
            header,
            text="📉 AI-Based Sales Drop Analysis",
            font=('Arial', 15, 'bold'),
            bg=COLORS['header_bg'], fg=COLORS['header_fg']
        ).pack(side='left', padx=20, pady=8)

        tk.Label(
            header,
            text="with Prediction & Smart Recommendations",
            font=('Arial', 11),
            bg=COLORS['header_bg'], fg='#93C5FD'
        ).pack(side='left', padx=0, pady=8)

        tk.Label(
            header, text="AI",
            font=('Arial', 9),
            bg=COLORS['header_bg'], fg='#94A3B8'
        ).pack(side='right', padx=20)

   
    def _build_toolbar(self):
        tb_frame = tk.Frame(self.root, bg='#E2E8F0', pady=6)
        tb_frame.pack(fill='x')

        tk.Label(
            tb_frame, text="Steps:", font=FONT_HEADER,
            bg='#E2E8F0', fg=COLORS['text_dark']
        ).pack(side='left', padx=10)

        steps = [
            ("📂 1. Upload CSV",        self._upload_csv,       COLORS['btn_primary']),
            ("🧹 2. Clean Data",         self._clean_data,        '#0891B2'),
            ("🔍 3. Drop Analysis",      self._run_drop_analysis, '#7C3AED'),
            ("📈 4. Prediction",         self._run_prediction,    COLORS['btn_success']),
            ("🤖 5. AI Recommendations", self._run_ai,            COLORS['btn_warning']),
        ]
        for label, cmd, color in steps:
            btn = tk.Button(
                tb_frame, text=label, command=cmd,
                bg=color, fg='white', font=FONT_BTN,
                relief='flat', padx=10, pady=5,
                cursor='hand2',
                activebackground=color, activeforeground='white'
            )
            btn.pack(side='left', padx=4)

    
    def _build_main_area(self):
        outer = tk.Frame(self.root, bg=COLORS['bg'])
        outer.pack(fill='both', expand=True, padx=10, pady=8)

        # Settings row (fixed height, does not expand)
        top_frame = tk.Frame(outer, bg=COLORS['bg'])
        top_frame.pack(fill='x', pady=(0, 6))
        self._build_settings_panel(top_frame)

        # ── Three panels in a PanedWindow so they ALWAYS show ──
        # Using a Frame with grid so all three are always visible
        panels_frame = tk.Frame(outer, bg=COLORS['bg'])
        panels_frame.pack(fill='both', expand=True, pady=(0, 6))

        # Give each column equal weight so they share space equally
        panels_frame.columnconfigure(0, weight=1, uniform='panel')
        panels_frame.columnconfigure(1, weight=1, uniform='panel')
        panels_frame.columnconfigure(2, weight=1, uniform='panel')
        panels_frame.rowconfigure(0, weight=1)

        self._build_drop_panel(panels_frame)
        self._build_prediction_panel(panels_frame)
        self._build_recommendation_panel(panels_frame)

        # Insight strip
        self._build_insight_panel(outer)

   
    def _build_settings_panel(self, parent):
        frame = tk.LabelFrame(
            parent,
            text=" ⚙️ Settings & Configuration ",
            font=FONT_HEADER,
            bg=COLORS['panel_bg'], fg=COLORS['section_head'],
            relief='groove', bd=1, padx=8, pady=6
        )
        frame.pack(fill='x', pady=(0, 4))

        # Row 1 — file path
        r1 = tk.Frame(frame, bg=COLORS['panel_bg'])
        r1.pack(fill='x', pady=2)
        tk.Label(r1, text="File:", font=FONT_NORMAL,
                 bg=COLORS['panel_bg'], fg=COLORS['text_dark'],
                 width=14, anchor='w').pack(side='left')
        tk.Entry(r1, textvariable=self.filepath, font=FONT_SMALL,
                 state='readonly', bg='#F1F5F9', relief='flat', bd=1,
                 width=55).pack(side='left', padx=4)
        tk.Button(r1, text="Browse...", command=self._upload_csv,
                  bg=COLORS['btn_primary'], fg='white', font=FONT_BTN,
                  relief='flat', padx=8, cursor='hand2').pack(side='left', padx=4)

        # Row 2 — column + threshold
        r2 = tk.Frame(frame, bg=COLORS['panel_bg'])
        r2.pack(fill='x', pady=2)
        tk.Label(r2, text="Analyse column:", font=FONT_NORMAL,
                 bg=COLORS['panel_bg'], fg=COLORS['text_dark'],
                 width=14, anchor='w').pack(side='left')
        self.value_combo = ttk.Combobox(r2, textvariable=self.value_col,
                                        state='readonly', width=18, font=FONT_NORMAL)
        self.value_combo.pack(side='left', padx=4)

        tk.Label(r2, text=" Group by:", font=FONT_NORMAL,
                 bg=COLORS['panel_bg'], fg=COLORS['text_dark']).pack(side='left')
        self.group_combo = ttk.Combobox(r2, textvariable=self.group_col,
                                        state='readonly', width=14, font=FONT_NORMAL)
        self.group_combo.pack(side='left', padx=4)

        tk.Label(r2, text=" Drop threshold %:", font=FONT_NORMAL,
                 bg=COLORS['panel_bg'], fg=COLORS['text_dark']).pack(side='left')
        tk.Spinbox(r2, from_=5, to=50, textvariable=self.threshold,
                   width=6, font=FONT_NORMAL).pack(side='left', padx=4)

        tk.Label(r2, text=" Predict periods:", font=FONT_NORMAL,
                 bg=COLORS['panel_bg'], fg=COLORS['text_dark']).pack(side='left')
        tk.Spinbox(r2, from_=1, to=12, textvariable=self.n_future,
                   width=4, font=FONT_NORMAL).pack(side='left', padx=4)

        # Row 3 — API key
        r3 = tk.Frame(frame, bg=COLORS['panel_bg'])
        r3.pack(fill='x', pady=2)
        tk.Label(r3, text="Groq API Key:", font=FONT_NORMAL,
                 bg=COLORS['panel_bg'], fg=COLORS['text_dark'],
                 width=14, anchor='w').pack(side='left')
        tk.Entry(r3, textvariable=self.api_key_var, font=FONT_SMALL,
                 show='*', bg='#F1F5F9', relief='flat', bd=1,
                 width=45).pack(side='left', padx=4)
        tk.Button(r3, text="Test Key", command=self._test_api_key,
                  bg='#475569', fg='white', font=FONT_BTN,
                  relief='flat', padx=8, cursor='hand2').pack(side='left', padx=4)
        tk.Label(r3, text="Get FREE key → console.groq.com (no card)",
                 font=FONT_SMALL, bg=COLORS['panel_bg'],
                 fg=COLORS['text_muted']).pack(side='left', padx=10)

   
    def _build_drop_panel(self, parent):
        frame = tk.LabelFrame(
            parent,
            text=" 🔍 Drop Analysis ",
            font=FONT_HEADER,
            bg=COLORS['panel_bg'], fg=COLORS['drop_red'],
            relief='groove', bd=1
        )
        # sticky='nsew' + uniform column weight = constant equal width
        frame.grid(row=0, column=0, sticky='nsew', padx=(0, 3), pady=0)

        # ── Detected Drops Table with scrollbars ──────────
        tk.Label(frame, text="Detected Drops Table:",
                 font=FONT_SMALL, bg=COLORS['panel_bg'],
                 fg=COLORS['text_muted']).pack(anchor='w', padx=6, pady=(4, 0))

        tree_wrap = tk.Frame(frame, bg=COLORS['panel_bg'])
        tree_wrap.pack(fill='x', padx=6, pady=(0, 4))

        sy = ttk.Scrollbar(tree_wrap, orient='vertical')
        sx = ttk.Scrollbar(tree_wrap, orient='horizontal')

        self.drops_tree = ttk.Treeview(
            tree_wrap, height=6, show='headings',
            selectmode='browse',
            yscrollcommand=sy.set,
            xscrollcommand=sx.set
        )
        sy.config(command=self.drops_tree.yview)
        sx.config(command=self.drops_tree.xview)

        # Pack order: scrollbars first, then tree (so tree fills remaining space)
        sy.pack(side='right', fill='y')
        sx.pack(side='bottom', fill='x')
        self.drops_tree.pack(side='left', fill='both', expand=True)

        # ── AI Why-drop text with scrollbar ───────────────
        tk.Label(frame, text="AI Analysis — Why did it drop?",
                 font=FONT_SMALL, bg=COLORS['panel_bg'],
                 fg=COLORS['text_muted']).pack(anchor='w', padx=6)

        why_wrap = tk.Frame(frame, bg=COLORS['panel_bg'])
        why_wrap.pack(fill='both', expand=True, padx=6, pady=(0, 6))

        why_sy = tk.Scrollbar(why_wrap, orient='vertical')
        why_sx = tk.Scrollbar(why_wrap, orient='horizontal')

        self.why_text = tk.Text(
            why_wrap, font=FONT_NORMAL, wrap='none',
            bg=COLORS['output_bg'], fg=COLORS['text_dark'],
            relief='flat', bd=1, state='disabled',
            yscrollcommand=why_sy.set,
            xscrollcommand=why_sx.set
        )
        why_sy.config(command=self.why_text.yview)
        why_sx.config(command=self.why_text.xview)

        why_sy.pack(side='right', fill='y')
        why_sx.pack(side='bottom', fill='x')
        self.why_text.pack(side='left', fill='both', expand=True)

    
    def _build_prediction_panel(self, parent):
        frame = tk.LabelFrame(
            parent,
            text=" 📈 AI Prediction (Linear Regression) ",
            font=FONT_HEADER,
            bg=COLORS['panel_bg'], fg=COLORS['section_head'],
            relief='groove', bd=1
        )
        frame.grid(row=0, column=1, sticky='nsew', padx=3, pady=0)

        # Metric cards
        info_row = tk.Frame(frame, bg=COLORS['panel_bg'])
        info_row.pack(fill='x', padx=6, pady=4)
        self.slope_lbl     = self._make_info_card(info_row, "Slope",     "—")
        self.intercept_lbl = self._make_info_card(info_row, "Intercept", "—")
        self.r2_lbl        = self._make_info_card(info_row, "R² Score",  "—")
        self.mae_lbl       = self._make_info_card(info_row, "MAE",       "—")

        self.direction_lbl = tk.Label(
            frame, text="Trend Direction: —",
            font=FONT_HEADER,
            bg=COLORS['panel_bg'], fg=COLORS['text_dark']
        )
        self.direction_lbl.pack(pady=(0, 2))

        # Bar chart placeholder (fixed height, never collapses)
        self.chart_frame = tk.Frame(
            frame, bg=COLORS['panel_bg'], height=160
        )
        self.chart_frame.pack(fill='x', padx=6, pady=(0, 4))
        self.chart_frame.pack_propagate(False)

        self._chart_placeholder = tk.Label(
            self.chart_frame,
            text="Chart appears after running prediction",
            font=FONT_SMALL, bg=COLORS['panel_bg'],
            fg=COLORS['text_muted']
        )
        self._chart_placeholder.pack(expand=True)

        # Future predictions table with scrollbar
        tk.Label(frame, text="Future Predictions:",
                 font=FONT_SMALL, bg=COLORS['panel_bg'],
                 fg=COLORS['text_muted']).pack(anchor='w', padx=6)

        pred_wrap = tk.Frame(frame, bg=COLORS['panel_bg'])
        pred_wrap.pack(fill='x', padx=6, pady=(0, 4))

        pred_sy = ttk.Scrollbar(pred_wrap, orient='vertical')
        self.pred_tree = ttk.Treeview(
            pred_wrap,
            columns=('Period', 'PredictedValue'),
            show='headings', height=4,
            yscrollcommand=pred_sy.set
        )
        self.pred_tree.heading('Period',         text='Period')
        self.pred_tree.heading('PredictedValue', text='Predicted Value')
        self.pred_tree.column('Period',         width=100)
        self.pred_tree.column('PredictedValue', width=120)
        pred_sy.config(command=self.pred_tree.yview)
        pred_sy.pack(side='right', fill='y')
        self.pred_tree.pack(side='left', fill='x', expand=True)

        # Prediction interpretation with scrollbars
        tk.Label(frame, text="Prediction Interpretation:",
                 font=FONT_SMALL, bg=COLORS['panel_bg'],
                 fg=COLORS['text_muted']).pack(anchor='w', padx=6)

        pred_text_wrap = tk.Frame(frame, bg=COLORS['panel_bg'])
        pred_text_wrap.pack(fill='both', expand=True, padx=6, pady=(0, 6))

        pt_sy = tk.Scrollbar(pred_text_wrap, orient='vertical')
        pt_sx = tk.Scrollbar(pred_text_wrap, orient='horizontal')

        self.pred_text = tk.Text(
            pred_text_wrap, font=FONT_NORMAL, wrap='none',
            bg=COLORS['output_bg'], fg=COLORS['text_dark'],
            relief='flat', bd=1, state='disabled',
            yscrollcommand=pt_sy.set,
            xscrollcommand=pt_sx.set
        )
        pt_sy.config(command=self.pred_text.yview)
        pt_sx.config(command=self.pred_text.xview)

        pt_sy.pack(side='right', fill='y')
        pt_sx.pack(side='bottom', fill='x')
        self.pred_text.pack(side='left', fill='both', expand=True)

    
    def _build_recommendation_panel(self, parent):
        frame = tk.LabelFrame(
            parent,
            text=" 💡 Recommendations to Increase ",
            font=FONT_HEADER,
            bg=COLORS['panel_bg'], fg=COLORS['success_green'],
            relief='groove', bd=1
        )
        frame.grid(row=0, column=2, sticky='nsew', padx=(3, 0), pady=0)

        stats_row = tk.Frame(frame, bg=COLORS['panel_bg'])
        stats_row.pack(fill='x', padx=6, pady=4)
        self.total_lbl = self._make_info_card(stats_row, "Total", "—")
        self.mean_lbl  = self._make_info_card(stats_row, "Mean",  "—")
        self.max_lbl   = self._make_info_card(stats_row, "Max",   "—")
        self.min_lbl   = self._make_info_card(stats_row, "Min",   "—")

        tk.Label(frame, text="AI Smart Recommendations:",
                 font=FONT_SMALL, bg=COLORS['panel_bg'],
                 fg=COLORS['text_muted']).pack(anchor='w', padx=6)

        rec_wrap = tk.Frame(frame, bg=COLORS['panel_bg'])
        rec_wrap.pack(fill='both', expand=True, padx=6, pady=(0, 6))

        rec_sy = tk.Scrollbar(rec_wrap, orient='vertical')
        rec_sx = tk.Scrollbar(rec_wrap, orient='horizontal')

        self.rec_text = tk.Text(
            rec_wrap, font=FONT_NORMAL, wrap='none',
            bg=COLORS['output_bg'], fg=COLORS['text_dark'],
            relief='flat', bd=1, state='disabled',
            yscrollcommand=rec_sy.set,
            xscrollcommand=rec_sx.set
        )
        rec_sy.config(command=self.rec_text.yview)
        rec_sx.config(command=self.rec_text.xview)

        rec_sy.pack(side='right', fill='y')
        rec_sx.pack(side='bottom', fill='x')
        self.rec_text.pack(side='left', fill='both', expand=True)

   
    def _build_insight_panel(self, parent):
        frame = tk.LabelFrame(
            parent,
            text=" ✨ Insight Generated ",
            font=FONT_HEADER,
            bg=COLORS['insight_bg'], fg=COLORS['insight_border'],
            relief='groove', bd=2
        )
        frame.pack(fill='x', pady=(0, 4))

        insight_wrap = tk.Frame(frame, bg=COLORS['insight_bg'])
        insight_wrap.pack(fill='x', padx=8, pady=6)

        ins_sy = tk.Scrollbar(insight_wrap, orient='vertical')
        ins_sx = tk.Scrollbar(insight_wrap, orient='horizontal')

        self.insight_text = tk.Text(
            insight_wrap, font=FONT_NORMAL, wrap='none',
            bg=COLORS['insight_bg'], fg=COLORS['text_dark'],
            relief='flat', bd=0, height=5, state='disabled',
            yscrollcommand=ins_sy.set,
            xscrollcommand=ins_sx.set
        )
        ins_sy.config(command=self.insight_text.yview)
        ins_sx.config(command=self.insight_text.xview)

        ins_sy.pack(side='right', fill='y')
        ins_sx.pack(side='bottom', fill='x')
        self.insight_text.pack(side='left', fill='both', expand=True)

    
    def _build_status_bar(self):
        bar = tk.Frame(self.root, bg='#334155', height=24)
        bar.pack(fill='x', side='bottom')
        bar.pack_propagate(False)
        tk.Label(
            bar, textvariable=self.status_var,
            font=FONT_SMALL, bg='#334155', fg='#94A3B8', anchor='w'
        ).pack(side='left', padx=10)

   
   
    def _make_info_card(self, parent, label_text, value_text):
        card = tk.Frame(parent, bg='#EFF6FF', relief='flat', bd=1, padx=8, pady=4)
        card.pack(side='left', padx=3, fill='x', expand=True)
        tk.Label(card, text=label_text, font=FONT_SMALL,
                 bg='#EFF6FF', fg=COLORS['text_muted']).pack()
        val_lbl = tk.Label(card, text=value_text, font=('Arial', 10, 'bold'),
                           bg='#EFF6FF', fg=COLORS['section_head'])
        val_lbl.pack()
        return val_lbl

  
    def _set_text(self, widget, content):
        widget.config(state='normal')
        widget.delete('1.0', 'end')
        widget.insert('end', content)
        widget.config(state='disabled')

    
    def _status(self, msg):
        self.status_var.set(msg)
        self.root.update_idletasks()

    def _upload_csv(self):
        path = filedialog.askopenfilename(
            title="Select your CSV file",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        if not path:
            return

        self._status("Loading CSV file...")
        df, msg = load_any_csv(path)

        if df is None:
            messagebox.showerror("Load Error", msg)
            self._status("Error loading file.")
            return

        self.df_raw   = df
        self.df_clean = None
        self.drops_df = None
        self.filepath.set(path)

        num_cols  = get_numeric_columns(df)
        text_cols = get_text_columns(df)

        self.value_combo['values'] = num_cols
        if num_cols:
            self.value_col.set(num_cols[0])

        self.group_combo['values'] = ['None'] + text_cols
        self.group_col.set('None')

        self._preview_data(df)
        self._status(f"✅ {msg} | Columns: {list(df.columns)}")
        messagebox.showinfo(
            "File Loaded",
            f"{msg}\n\nColumns found:\n"
            f"{', '.join(df.columns)}\n\n"
            "Next: Click '🧹 2. Clean Data'"
        )

    # ─────────────────────────────────────────────────────
    #  PREVIEW — top rows in drops tree
    # ─────────────────────────────────────────────────────
    def _preview_data(self, df):
        cols = list(df.columns)
        self.drops_tree['columns'] = cols
        for col in cols:
            self.drops_tree.heading(col, text=col)
            self.drops_tree.column(col, width=80, minwidth=50)
        for row in self.drops_tree.get_children():
            self.drops_tree.delete(row)
        for _, row in df.head(8).iterrows():
            self.drops_tree.insert('', 'end', values=[str(v) for v in row.values])

    # ─────────────────────────────────────────────────────
    #  STEP 2 — Clean Data
    # ─────────────────────────────────────────────────────
    def _clean_data(self):
        if self.df_raw is None:
            messagebox.showwarning("No Data", "Please upload a CSV file first.")
            return

        self._status("Cleaning data...")
        df_clean, report = clean_data(self.df_raw)
        self.df_clean = df_clean

        lines = [
            "DATA CLEANING REPORT",
            "=" * 35,
            f"Original rows     : {report['original_rows']}",
            f"Cleaned rows      : {report['cleaned_rows']}",
            f"Duplicates removed: {report['duplicate_rows']}",
            f"Missing before    : {report['missing_before']}",
            f"Missing after     : {report['missing_after']}",
            "",
            "Columns fixed:",
        ]
        if report.get('columns_fixed'):
            for fix in report['columns_fixed']:
                lines.append(f"  • {fix}")
        else:
            lines.append("  • No missing values found — data is clean!")

        self._set_text(self.why_text, "\n".join(lines))
        self._status(
            f"✅ Data cleaned — "
            f"{report['missing_before']} missing values handled, "
            f"{report['duplicate_rows']} duplicates removed."
        )
        messagebox.showinfo(
            "Cleaning Complete",
            f"Data cleaned successfully!\n\n"
            f"Missing values fixed: {report['missing_before']}\n"
            f"Duplicate rows removed: {report['duplicate_rows']}\n\n"
            "Next: Click '🔍 3. Drop Analysis'"
        )

   
    def _run_drop_analysis(self):
        if self.df_clean is None:
            messagebox.showwarning("No Data", "Please clean data first (Step 2).")
            return

        vcol = self.value_col.get()
        if not vcol:
            messagebox.showwarning("No Column", "Please select a column to analyse.")
            return

        gcol      = _none(self.group_col.get())
        threshold = self.threshold.get()

        self._status(f"Running drop analysis on '{vcol}'...")

        drops_df, all_df = detect_drops(self.df_clean, vcol, gcol, threshold)
        self.drops_df = drops_df
        self.all_df   = all_df
        self.stats    = get_stats(self.df_clean, vcol)

        self.total_lbl.config(text=f"{self.stats.get('total', 0):,.0f}")
        self.mean_lbl.config( text=f"{self.stats.get('mean',  0):,.1f}")
        self.max_lbl.config(  text=f"{self.stats.get('max',   0):,.0f}")
        self.min_lbl.config(  text=f"{self.stats.get('min',   0):,.0f}")

        drop_cols = [c for c in drops_df.columns if c not in ['prev_value']]
        self.drops_tree['columns'] = drop_cols
        for c in drop_cols:
            self.drops_tree.heading(c, text=c)
            self.drops_tree.column(c, width=85, minwidth=50)

        for row in self.drops_tree.get_children():
            self.drops_tree.delete(row)

        if len(drops_df) == 0:
            self.drops_tree.insert('', 'end',
                                   values=["No drops found above threshold"])
            self._set_text(
                self.why_text,
                f"✅ No drops detected in '{vcol}' "
                f"above {threshold}% threshold.\n"
                "The data appears stable."
            )
            self._status("✅ No drops detected.")
        else:
            for _, row in drops_df.iterrows():
                self.drops_tree.insert('', 'end',
                                       values=[str(row[c]) for c in drop_cols])

            n_critical = n_high = n_medium = 0
            if 'severity' in drops_df.columns:
                n_critical = drops_df['severity'].str.contains('Critical', na=False).sum()
                n_high     = drops_df['severity'].str.contains('High',     na=False).sum()
                n_medium   = drops_df['severity'].str.contains('Medium',   na=False).sum()

            self._set_text(
                self.why_text,
                f"Found {len(drops_df)} drop(s) in '{vcol}'.\n\n"
                f"Click '🤖 5. AI Recommendations' to get AI\n"
                f"analysis of WHY these drops happened.\n\n"
                f"Drop threshold used : {threshold}%\n"
                f"Drops detected      : {len(drops_df)}\n"
                f"Critical drops      : {n_critical}\n"
                f"High drops          : {n_high}\n"
                f"Medium drops        : {n_medium}"
            )
            self._status(f"✅ {len(drops_df)} drop(s) found in '{vcol}'.")

    
    def _run_prediction(self):
        if self.df_clean is None:
            messagebox.showwarning("No Data", "Please clean data first (Step 2).")
            return

        vcol = self.value_col.get()
        if not vcol:
            messagebox.showwarning("No Column", "Please select a column to predict.")
            return

        gcol = _none(self.group_col.get())
        n    = self.n_future.get()

        self._status(f"Running Linear Regression prediction for '{vcol}'...")

        try:
            result = run_full_prediction(
                self.df_clean,
                value_col=vcol,
                group_col=gcol,
                group_name=None,
                n_future=n
            )
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            self._status(f"Prediction failed: {e}")
            return

        if not result.get('success'):
            messagebox.showerror("Prediction Error", result.get('error', 'Unknown error'))
            self._status("Prediction failed.")
            return

        self.pred_result = result

        # Update metric cards
        self.slope_lbl.config(    text=str(result['slope']))
        self.intercept_lbl.config(text=str(result['intercept']))
        self.r2_lbl.config(       text=str(result['metrics']['R2']))
        self.mae_lbl.config(      text=str(result['metrics']['MAE']))

        is_up = '↑' in str(result.get('direction', ''))
        self.direction_lbl.config(
            text=f"Trend Direction: {result.get('direction', '—')}",
            fg='#16A34A' if is_up else '#DC2626'
        )

        # Draw chart
        self._draw_chart(result, vcol)

        # Populate future predictions table
        for row in self.pred_tree.get_children():
            self.pred_tree.delete(row)
        for lbl, val in zip(result['future_labels'], result['future_preds']):
            self.pred_tree.insert('', 'end', values=(lbl, f"{val:,.2f}"))

        # Interpretation text
        interp_lines = [
            "LINEAR REGRESSION RESULTS",
            "=" * 35,
            f"Column analysed  : {vcol}",
            f"Data points used : {result['metrics']['n_samples']}",
            f"Slope  (m)       : {result['slope']}",
            f"Intercept (c)    : {result['intercept']}",
            "",
            f"Formula: y = {result['slope']} × x + {result['intercept']}",
            "",
            "MODEL ACCURACY",
            "=" * 35,
            f"R² Score : {result['metrics']['R2']} — "
            f"{result['metrics']['R2_label']}",
            f"MAE      : {result['metrics']['MAE']} (avg prediction error)",
            "",
            "FUTURE FORECAST",
            "=" * 35,
        ]
        for lbl, val in zip(result['future_labels'], result['future_preds']):
            interp_lines.append(f"  {lbl} → {val:,.2f}")

        self._set_text(self.pred_text, "\n".join(interp_lines))
        self._status(
            f"✅ Prediction complete. R²={result['metrics']['R2']}, "
            f"Direction={result.get('direction', '—')}"
        )

   
    def _draw_chart(self, result, vcol):
        # Remove placeholder and any previous chart
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        y_vals = result['y']
        x_idx  = list(range(1, len(y_vals) + 1))
        trend  = result['trend_line']

        fig, ax = plt.subplots(figsize=(4.2, 2.0), dpi=88)
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#F8FAFC')

        ax.bar(x_idx, y_vals, color='#2563EB', alpha=0.75,
               label='Actual', zorder=2)
        ax.plot(x_idx, trend, color='#DC2626', linewidth=1.8,
                linestyle='--', label='Trend', zorder=3)

        ax.set_title(f"{vcol} — Actual vs Trend",
                     fontsize=8, color='#1E293B', pad=4)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc='upper right')
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:,.0f}")
        )
        fig.tight_layout(pad=0.5)

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        self._chart_fig    = fig
        self._chart_canvas = canvas

   
    def _run_ai(self):
        if self.df_clean is None:
            messagebox.showwarning("No Data", "Please complete Steps 2, 3 and 4 first.")
            return
        if self.drops_df is None:
            messagebox.showwarning("No Analysis", "Please run Drop Analysis (Step 3) first.")
            return
        if not self.pred_result:
            messagebox.showwarning("No Prediction", "Please run Prediction (Step 4) first.")
            return

        api_key = self.api_key_var.get().strip()
        if not api_key:
            messagebox.showwarning(
                "No API Key",
                "Please enter your Groq API key.\n\n"
                "Get FREE key at: https://console.groq.com\n"
                "Sign up with Google → API Keys → Create API Key"
            )
            return

        self._status("Calling Groq AI... please wait (~5-10 sec)...")
        self._set_text(self.why_text,     "⏳ Asking AI why values dropped...")
        self._set_text(self.rec_text,     "⏳ Generating recommendations...")
        self._set_text(self.insight_text, "⏳ Generating insight summary...")
        self.root.update_idletasks()

        threading.Thread(target=self._ai_worker, daemon=True).start()

    def _ai_worker(self):
        try:
            vcol  = self.value_col.get()
            fname = os.path.basename(self.filepath.get())
            key   = self.api_key_var.get().strip()

            self.root.after(0, self._status,
                            "AI Call 1/3 — Analysing why values dropped...")
            why = get_why_drops(self.drops_df, self.stats, vcol, key, fname)
            self.root.after(0, self._set_text, self.why_text, why)

            self.root.after(0, self._status,
                            "AI Call 2/3 — Generating recommendations...")
            rec = get_recommendations(self.drops_df, self.stats,
                                      self.pred_result, vcol, key, fname)
            self.root.after(0, self._set_text, self.rec_text, rec)

            self.root.after(0, self._status,
                            "AI Call 3/3 — Generating insight summary...")
            ins = get_insight_summary(self.drops_df, self.stats,
                                     self.pred_result, vcol, key, fname)
            self.root.after(0, self._set_text, self.insight_text, ins)

            self.root.after(0, self._status,
                            "✅ AI analysis complete! All insights generated.")

        except Exception as e:
            err_msg = str(e)
            self.root.after(0, self._status, f"AI Error: {err_msg}")
            self.root.after(0, messagebox.showerror, "AI Error", err_msg)

   
    def _test_api_key(self):
        key = self.api_key_var.get().strip()
        if not key:
            messagebox.showwarning("No Key", "Please enter a Groq API key first.")
            return

        self._status("Testing API key...")
        ok, err = test_api_key(key)
        if ok:
            messagebox.showinfo("Key Valid", "✅ Groq API key is valid and working!")
            self._status("✅ API key is valid.")
        else:
            messagebox.showerror("Key Invalid", f"❌ {err}")
            self._status("❌ API key invalid.")


if __name__ == '__main__':
    root = tk.Tk()
    app  = SalesDropApp(root)
    root.mainloop()