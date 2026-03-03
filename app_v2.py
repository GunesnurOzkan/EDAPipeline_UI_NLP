import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ===================================================================
# NLP — T5 Model
# ===================================================================
@st.cache_resource
def load_nlp_model():
    model_name = "t5-small"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def generate_nlp_insight(tokenizer, model, context_data):
    prompt = f"Summarize the following data analysis insights in simple terms: {context_data}"
    try:
        inputs  = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs.input_ids, max_length=150, num_return_sequences=1)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Insight error: {str(e)}"


# ===================================================================
# Smart Data Analyzer
# ===================================================================
class SmartDataAnalyzerStreamlit:
    def __init__(self, df):
        self.df      = df
        self.results = {}

    def phase_1_basic_eda(self):
        self.results['shape']    = self.df.shape
        self.results['describe'] = self.df.describe().T

        missing_count   = self.df.isnull().sum()
        missing_percent = (missing_count / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Eksik Sayısı':    missing_count,
            'Eksik Yüzdesi (%)': missing_percent
        })
        missing_df = missing_df[missing_df['Eksik Sayısı'] > 0].sort_values(
            by='Eksik Yüzdesi (%)', ascending=False)
        self.results['missing_values'] = missing_df

        if not missing_df.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(missing_df.index, missing_df['Eksik Yüzdesi (%)'], color='salmon')
            ax.set_title('Eksik Veri Oranları (%)', fontsize=12)
            ax.set_xticklabels(missing_df.index, rotation=45, ha='right')
            ax.set_ylabel('%')
            plt.tight_layout()
            self.results['missing_plot_fig'] = fig
            plt.close(fig)

        numeric_df = self.df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr()
            self.results['correlation'] = corr_matrix
            fig2, ax2 = plt.subplots(figsize=(9, 7))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',
                        fmt=".2f", linewidths=.5, ax=ax2)
            ax2.set_title('Korelasyon Matrisi Heatmap', fontsize=14)
            plt.tight_layout()
            self.results['corr_plot_fig'] = fig2
            plt.close(fig2)

    def phase_2_outliers(self):
        numeric_df   = self.df.select_dtypes(include=[np.number])
        outliers_dict = {}
        for col in numeric_df.columns:
            Q1  = numeric_df[col].quantile(0.25)
            Q3  = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lb  = Q1 - 1.5 * IQR
            ub  = Q3 + 1.5 * IQR
            mask = (numeric_df[col] < lb) | (numeric_df[col] > ub)
            cnt  = mask.sum()
            if cnt > 0:
                outliers_dict[col] = {
                    'count': int(cnt),
                    'percent': (cnt / len(numeric_df)) * 100,
                    'lower_bound': lb,
                    'upper_bound': ub
                }
        self.results['outliers'] = outliers_dict

    def phase_3_business_insights(self):
        insights = []
        mv = self.results.get('missing_values', pd.DataFrame())
        if not mv.empty:
            for idx, row in mv.iterrows():
                pct = row['Eksik Yüzdesi (%)']
                insights.append(f"'{idx}' kolonunda %{pct:.1f} eksik veri var.")
        for col, data in self.results.get('outliers', {}).items():
            insights.append(
                f"'{col}' kolonunda aykırı değer yoğunluğu tespit edildi (%{data['percent']:.1f}).")
        corr = self.results.get('correlation')
        if corr is not None:
            cols = corr.columns.tolist()
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    val = corr.iloc[i, j]
                    if abs(val) > 0.80:
                        d = "pozitif" if val > 0 else "negatif"
                        insights.append(
                            f"'{cols[i]}' ve '{cols[j]}' arasında güçlü {d} korelasyon var ({val:.2f}).")
        self.results['insights'] = insights


# ===================================================================
# Inline EDA Plot Generator  (replaces EDAPipeline file-save approach)
# ===================================================================
def build_eda_plots(df):
    """
    Returns a dict of {tab_name: [(title, fig), ...]} with all EDA plots
    rendered inline — no plt.show(), no disk I/O.
    """
    numeric_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    sections = {
        "📈 Sayısal Dağılımlar": [],
        "📦 Box / Violin":       [],
        "🗂️ Kategorik":         [],
        "🔗 Korelasyon":         [],
        "🔀 Bivariate":          [],
        "🚨 Aykırı (Boxplot)":  [],
    }

    # --- 1. Histograms for numeric columns ---
    for col in numeric_cols:
        try:
            fig, ax = plt.subplots(figsize=(5, 3))
            data = df[col].dropna()
            ax.hist(data, bins=20, color='steelblue', edgecolor='white', alpha=0.85)
            ax.set_title(f'Dağılım: {col}', fontsize=10)
            ax.set_xlabel(col)
            ax.set_ylabel('Frekans')
            plt.tight_layout()
            sections["📈 Sayısal Dağılımlar"].append((col, fig))
            plt.close(fig)
        except Exception:
            pass

    # --- 2. Box + Violin per numeric column ---
    for col in numeric_cols:
        try:
            data = df[col].dropna()
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            axes[0].boxplot(data, patch_artist=True,
                            boxprops=dict(facecolor='lightblue'))
            axes[0].set_title(f'Boxplot: {col}', fontsize=9)
            axes[1].violinplot(data, showmedians=True)
            axes[1].set_title(f'Violin: {col}', fontsize=9)
            plt.tight_layout()
            sections["📦 Box / Violin"].append((col, fig))
            plt.close(fig)
        except Exception:
            pass

    # --- 3. Bar charts for categorical columns (top 15 values) ---
    for col in categorical_cols:
        try:
            vc = df[col].value_counts().head(15)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(vc.index.astype(str), vc.values, color='coral', edgecolor='white')
            ax.set_title(f'Kategorik: {col}', fontsize=10)
            ax.set_xticklabels(vc.index.astype(str), rotation=45, ha='right', fontsize=7)
            ax.set_ylabel('Sayı')
            plt.tight_layout()
            sections["🗂️ Kategorik"].append((col, fig))
            plt.close(fig)
        except Exception:
            pass

    # --- 4. Correlation heatmap ---
    if len(numeric_cols) > 1:
        try:
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(min(12, len(numeric_cols) + 2),
                                            min(10, len(numeric_cols) + 1)))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                        linewidths=.5, ax=ax)
            ax.set_title('Korelasyon Matrisi', fontsize=13)
            plt.tight_layout()
            sections["🔗 Korelasyon"].append(("Korelasyon Heatmap", fig))
            plt.close(fig)
        except Exception:
            pass

        # Pair scatter (max 5 columns to keep it fast)
        try:
            sample_cols = numeric_cols[:5]
            pair_df     = df[sample_cols].dropna()
            if len(pair_df) > 1 and len(sample_cols) > 1:
                pair_fig = sns.pairplot(pair_df, diag_kind='kde',
                                        plot_kws={'alpha': 0.5, 's': 15})
                pair_fig.fig.suptitle("Çift Değişken Scatter (ilk 5 sayısal kolon)",
                                       y=1.02, fontsize=11)
                sections["🔀 Bivariate"].append(("Pairplot", pair_fig.fig))
                plt.close(pair_fig.fig)
        except Exception:
            pass

    # --- 5. Bivariate: scatter for each pair with target if available ---
    if len(numeric_cols) >= 2:
        for i in range(min(len(numeric_cols), 4)):
            for j in range(i + 1, min(len(numeric_cols), 4)):
                col_x = numeric_cols[i]
                col_y = numeric_cols[j]
                try:
                    tmp = df[[col_x, col_y]].dropna()
                    if len(tmp) < 2:
                        continue
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.scatter(tmp[col_x], tmp[col_y], alpha=0.6,
                               color='mediumseagreen', edgecolors='white', s=30)
                    # Trend line
                    z = np.polyfit(tmp[col_x], tmp[col_y], 1)
                    p = np.poly1d(z)
                    ax.plot(sorted(tmp[col_x]), p(sorted(tmp[col_x])),
                            color='crimson', linewidth=1.5, linestyle='--')
                    ax.set_xlabel(col_x)
                    ax.set_ylabel(col_y)
                    ax.set_title(f'{col_x} vs {col_y}', fontsize=9)
                    plt.tight_layout()
                    sections["🔀 Bivariate"].append((f"{col_x} vs {col_y}", fig))
                    plt.close(fig)
                except Exception:
                    pass

    # --- 6. Outlier boxplots (pure matplotlib, seaborn-safe) ---
    for col in numeric_cols:
        try:
            data = df[col].dropna()
            Q1   = data.quantile(0.25)
            Q3   = data.quantile(0.75)
            IQR  = Q3 - Q1
            lb   = Q1 - 1.5 * IQR
            ub   = Q3 + 1.5 * IQR
            outliers = data[(data < lb) | (data > ub)]
            if len(outliers) == 0:
                continue
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.boxplot(data, patch_artist=True,
                       boxprops=dict(facecolor='#ffe0b2'),
                       flierprops=dict(marker='o', color='crimson',
                                       markerfacecolor='crimson', markersize=5))
            ax.axhline(ub, color='red',    linestyle='--', linewidth=1, label=f'Üst ({ub:.1f})')
            ax.axhline(lb, color='orange', linestyle='--', linewidth=1, label=f'Alt ({lb:.1f})')
            ax.set_title(f'Aykırı Değer: {col} ({len(outliers)} adet)', fontsize=9)
            ax.legend(fontsize=7)
            plt.tight_layout()
            sections["🚨 Aykırı (Boxplot)"].append((col, fig))
            plt.close(fig)
        except Exception:
            pass

    # Remove empty sections
    return {k: v for k, v in sections.items() if v}


# ===================================================================
# Streamlit UI
# ===================================================================
st.set_page_config(page_title="Smart Data Analyzer — NLP + EDA", layout="wide")

st.title("📊 Smart Data Analyzer — NLP + EDA Dashboard")
st.markdown(
    "Veri setinizi yükleyin; **temel analizler**, **T5 NLP içgörüleri** ve "
    "**detaylı EDA görselleştirmelerini** tek ekranda görün."
)

# Load T5 once (cached)
try:
    with st.spinner("T5 NLP modeli yükleniyor... Lütfen bekleyin."):
        tokenizer, nlp_model = load_nlp_model()
    st.sidebar.success("T5 modeli hazır ✅")
except Exception as e:
    st.error(f"Model yüklenirken hata oluştu: {e}")
    st.stop()

# Sidebar
st.sidebar.title("⚙️ Ayarlar")
uploaded_file = st.sidebar.file_uploader(
    "CSV veya Excel yükleyin", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') \
             else pd.read_excel(uploaded_file)
        st.sidebar.success(f"✅ {uploaded_file.name} yüklendi")

        target_col = st.sidebar.selectbox(
            "Hedef kolon (isteğe bağlı)", ["Yok"] + list(df.columns))
        target_col = None if target_col == "Yok" else target_col

        run_btn = st.sidebar.button("🚀 Analizi Başlat")

        if run_btn:
            # ---- SmartDataAnalyzer ----
            with st.spinner("Temel veri analizi yapılıyor..."):
                analyzer = SmartDataAnalyzerStreamlit(df)
                analyzer.phase_1_basic_eda()
                analyzer.phase_2_outliers()
                analyzer.phase_3_business_insights()
                results = analyzer.results

            # --- 1. Shape ---
            st.header("📌 1. Veri Boyutu ve Önizleme")
            c1, c2 = st.columns(2)
            c1.metric("Satır", results['shape'][0])
            c2.metric("Sütun", results['shape'][1])
            st.dataframe(df.head())
            with st.expander("İstatistiksel Özet (describe)"):
                st.dataframe(results.get('describe', pd.DataFrame()))

            st.divider()

            # --- 2. NLP ---
            st.header("🧠 2. T5 NLP Model İçgörüsü")
            with st.spinner("T5 modeli yorumluyor..."):
                ctx = ". ".join(results.get('insights', []))
                if not ctx:
                    ctx = ("The dataset appears clean: no missing values, "
                           "no extreme outliers, and no very high correlations detected.")
                t5_text = generate_nlp_insight(tokenizer, nlp_model, ctx)
            st.info(f"**T5 Yorumu:** {t5_text}")
            with st.expander("Ham Kural Tabanlı Bulgular"):
                if results.get('insights'):
                    for ins in results['insights']:
                        st.write(f"• {ins}")
                else:
                    st.write("Belirgin bir anormallik bulunamadı — veri temiz görünüyor.")

            st.divider()

            # --- 3. Missing ---
            st.header("🔍 3. Eksik Veriler")
            mv = results.get('missing_values', pd.DataFrame())
            if not mv.empty:
                st.dataframe(mv)
                if 'missing_plot_fig' in results:
                    st.pyplot(results['missing_plot_fig'])
            else:
                st.success("Veri setinde hiç eksik veri yok! ✅")

            st.divider()

            # --- 4. Outliers ---
            st.header("📊 4. Aykırı Değerler (IQR)")
            outliers = results.get('outliers', {})
            if outliers:
                odf = pd.DataFrame(outliers).T[['count', 'percent']]
                odf.columns = ['Aykırı Değer Sayısı', 'Yüzdesi (%)']
                st.dataframe(odf.sort_values(by='Yüzdesi (%)', ascending=False))
            else:
                st.success("Önemli bir aykırı değer bulunamadı! ✅")

            st.divider()

            # --- 5. Correlation ---
            st.header("🔗 5. Korelasyon Haritası")
            if 'corr_plot_fig' in results:
                st.pyplot(results['corr_plot_fig'])
            else:
                st.warning("Yeterli sayısal kolon bulunamadı.")

            st.divider()

            # --- 6. Full EDA Visual Gallery ---
            st.header("🔬 6. Detaylı EDA Görsel Galerisi")
            st.caption(
                "Sayısal dağılımlar, box/violin grafikleri, kategorik çubuk grafikler, "
                "korelasyon heatmap, bivariate scatter ve aykırı değer boxplot'ları."
            )
            with st.spinner("Grafikler oluşturuluyor..."):
                eda_sections = build_eda_plots(df)

            if eda_sections:
                st.success(f"✅ {sum(len(v) for v in eda_sections.values())} grafik oluşturuldu.")
                tabs = st.tabs(list(eda_sections.keys()))
                for tab, (section_name, plots) in zip(tabs, eda_sections.items()):
                    with tab:
                        st.markdown(f"**{len(plots)} grafik** bu kategoride.")
                        cols3 = st.columns(3)
                        for i, (title, fig) in enumerate(plots):
                            with cols3[i % 3]:
                                st.pyplot(fig)
                                st.caption(title)
            else:
                st.warning("Grafik üretilemedi. Veri setinizde sayısal veya kategorik kolon bulunmuyor olabilir.")

    except Exception as e:
        st.error(f"Dosya işlenirken hata oluştu: {str(e)}")
else:
    st.info("⬅️ Sol kenar çubuğundan CSV veya Excel dosyanızı yükleyin.")
