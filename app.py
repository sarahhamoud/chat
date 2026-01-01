import os
import pandas as pd
import streamlit as st
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, mean_squared_error, r2_score, mean_absolute_error

from pycaret.classification import (
    setup as clf_setup, compare_models as clf_compare_models, pull as clf_pull,
    save_model as clf_save_model, predict_model as clf_predict_model,
    get_config as clf_get_config, create_model as clf_create_model,
    finalize_model as clf_finalize_model
)
from pycaret.regression import (
    setup as reg_setup, compare_models as reg_compare_models, pull as reg_pull,
    save_model as reg_save_model, predict_model as reg_predict_model,
    get_config as reg_get_config, create_model as reg_create_model,
    finalize_model as reg_finalize_model
)

from streamlit_chat import message
from openai import OpenAI

# =========================
# Page Config
# =========================
st.set_page_config(page_title="منصة AutoML", layout="wide")

# =========================
# Light RTL Theme + Clean UI
# =========================
st.markdown("""
<style>
/* RTL + Fonts */
html, body, [data-testid="stApp"] {
    direction: rtl;
    text-align: right;
    font-family: "Segoe UI", "Tahoma", sans-serif;
    background: #F6F8FB;
}

/* Hide Streamlit default header & menu */
header, #MainMenu, footer {visibility: hidden; height: 0px;}

/* Main container spacing */
.block-container {
    padding-top: 2.2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Top bar */
.topbar {
    background: linear-gradient(90deg, #ffffff, #f3f6ff);
    border: 1px solid #E6EAF2;
    border-radius: 18px;
    padding: 14px 18px;
    margin-bottom: 16px;
    box-shadow: 0 8px 20px rgba(16,24,40,0.06);
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.brand-title {
    font-size: 22px;
    font-weight: 800;
    color: #0F172A;
    letter-spacing: 0.3px;
}
.brand-sub {
    font-size: 13px;
    color: #475569;
    margin-top: 2px;
}
.owner {
    direction: ltr;
    text-align: left;
    font-size: 14px;
    font-weight: 700;
    color: #0F172A;
    padding: 8px 12px;
    border-radius: 12px;
    border: 1px solid #E6EAF2;
    background: #FFFFFF;
}

/* Cards */
.card {
    background: #FFFFFF;
    border: 1px solid #E6EAF2;
    border-radius: 18px;
    padding: 16px;
    box-shadow: 0 8px 18px rgba(16,24,40,0.06);
    margin-bottom: 16px;
}
.card h3 {
    margin: 0 0 10px 0;
    color: #0F172A;
    font-size: 18px;
}
.small {
    color: #475569;
    font-size: 14px;
}

/* Inputs labels */
label, .stRadio, .stSelectbox, .stSlider, .stTextInput, .stFileUploader {
    font-size: 15px !important;
    color: #0F172A !important;
}

/* Buttons */
.stButton>button {
    background: #2563EB;
    color: white;
    border-radius: 14px;
    padding: 10px 14px;
    border: 0;
    font-weight: 700;
}
.stButton>button:hover {
    background: #1D4ED8;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E6EAF2;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.2rem;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid #E6EAF2;
}

/* Alerts */
.stAlert {
    border-radius: 14px !important;
}

/* Chat message container override (optional) */
</style>
""", unsafe_allow_html=True)

# =========================
# Top Bar
# =========================
st.markdown(f"""
<div class="topbar">
  <div>
    <div class="brand-title"> AutoML المنصة الذكية لرفع البيانات وتحليلها</div>
    <div class="brand-sub">رفع بيانات • تحليل استكشافي • تدريب نماذج • تحميل النموذج • مساعد ذكي</div>
  </div>
  <div class="owner">sarah hamoud hussien</div>
</div>
""", unsafe_allow_html=True)

# =========================
# Helpers
# =========================
DATA_PATH = "dataset.csv"

def load_df_if_exists():
    if os.path.exists(DATA_PATH):
        try:
            return pd.read_csv(DATA_PATH)
        except Exception:
            return None
    return None

def require_df(df):
    if df is None or not isinstance(df, pd.DataFrame):
        st.error("لا توجد بيانات محمّلة. اذهبي إلى (رفع البيانات) وارفعـي ملف CSV.")
        st.stop()
    if df.empty:
        st.error("البيانات فارغة. ارفعي ملف CSV صحيح.")
        st.stop()

def kpi_row(df):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("عدد الصفوف", f"{df.shape[0]:,}")
    with c2:
        st.metric("عدد الأعمدة", f"{df.shape[1]:,}")
    with c3:
        missing = int(df.isna().sum().sum())
        st.metric("القيم المفقودة", f"{missing:,}")

# =========================
# Load df
# =========================
df = load_df_if_exists()

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2019/12/shutterstock_1166533285-Converted-02.png")
    st.markdown("### لوحة التحكم")
    choice = st.radio(
        "اختر القسم",
        ["رفع البيانات", "تحليل البيانات", "تدريب النماذج", "تحميل النموذج", "المساعد الذكي"],
        index=0
    )
    st.info("هذه المنصة تساعدك على بناء نماذج تعلم آلي تلقائيًا وعرض تحليل شامل للبيانات.")

# =========================
# Upload
# =========================
if choice == "رفع البيانات":
    st.markdown('<div class="card"><h3>رفع ملف البيانات</h3><div class="small">ارفع ملف CSV وسيتم حفظه تلقائيًا داخل التطبيق.</div></div>', unsafe_allow_html=True)

    file = st.file_uploader("ارفع ملف CSV", type=["csv"])
    if file:
        try:
            df = pd.read_csv(file)
            df.to_csv(DATA_PATH, index=False)
            st.success("✅ تم رفع البيانات وحفظها بنجاح.")
            kpi_row(df)
            st.dataframe(df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"تعذر قراءة الملف: {e}")

# =========================
# Profiling (EDA)
# =========================
if choice == "تحليل البيانات":
    st.markdown('<div class="card"><h3>التحليل الاستكشافي للبيانات (EDA)</h3><div class="small">سيتم إنشاء تقرير شامل عن البيانات. إذا كان الملف كبيرًا، اختاري وضع سريع.</div></div>', unsafe_allow_html=True)

    require_df(df)
    kpi_row(df)
    st.dataframe(df.head(10), use_container_width=True)

    minimal = st.toggle("وضع سريع (أخف)", value=True, key="eda_minimal")
    rows = st.number_input("عدد الصفوف للتقرير (لتخفيف الحمل)", min_value=100, max_value=200000, value=5000, step=500, key="eda_rows")

    data_for_report = df.head(int(rows))

    drop_text = st.toggle("تجاهل الأعمدة النصية الثقيلة (لتفادي أعطال)", value=True, key="eda_drop_text")
    if drop_text:
        text_cols = [c for c in data_for_report.columns if data_for_report[c].dtype == "object"]
        if len(text_cols) > 0:
            data_for_report = data_for_report.drop(columns=text_cols)

    # ✅ مكان لحفظ التقرير داخل الجلسة
    if "profile_html" not in st.session_state:
        st.session_state["profile_html"] = None

    colA, colB = st.columns([1,1])
    with colA:
        create_btn = st.button("إنشاء التقرير", key="btn_make_report")
    with colB:
        clear_btn = st.button("مسح التقرير", key="btn_clear_report")

    if clear_btn:
        st.session_state["profile_html"] = None
        st.success("تم مسح التقرير من الجلسة.")

    if create_btn:
        with st.spinner("⏳ جاري إنشاء التقرير..."):
            try:
                profile = ydata_profiling.ProfileReport(
                    data_for_report,
                    explorative=True,
                    minimal=bool(minimal),
                )

                # ✅ أهم خطوة: تحويله إلى HTML وحفظه
                st.session_state["profile_html"] = profile.to_html()
                st.success("✅ تم إنشاء التقرير وحفظه. سيظهر أدناه ولن يختفي.")

            except Exception as e:
                st.error(f"فشل إنشاء التقرير: {e}")
                st.stop()

    # ✅ عرض التقرير إذا موجود
    if st.session_state["profile_html"]:
        st.markdown('<div class="card"><h3>عرض التقرير</h3><div class="small">إذا لم يظهر داخل الإطار، استخدمي زر التحميل وافتحيه محليًا.</div></div>', unsafe_allow_html=True)

        st.download_button(
            "⬇️ تحميل تقرير التحليل (HTML)",
            data=st.session_state["profile_html"].encode("utf-8"),
            file_name="profiling_report.html",
            mime="text/html",
            key="download_profile_html"
        )

        # عرض داخل التطبيق
        st.components.v1.html(st.session_state["profile_html"], height=900, scrolling=True)
    else:
        st.info("لا يوجد تقرير معروض الآن. اضغطي (إنشاء التقرير).")


# =========================
# Modeling
# =========================
if choice == "تدريب النماذج":
    st.markdown('<div class="card"><h3>تدريب النماذج</h3><div class="small">اختاري العمود الهدف ونوع المهمة، ثم شغّلي أفضل نموذج أو نموذج محدد.</div></div>', unsafe_allow_html=True)

    require_df(df)
    kpi_row(df)

    chosen_target = st.selectbox("اختاري عمود الهدف (Target)", df.columns)

    algorithm_type = st.radio("نوع المهمة", ["تصنيف (Classification)", "انحدار (Regression)"])
    run_mode = st.radio("طريقة التشغيل", ["تشغيل كل النماذج (أفضل نموذج)", "تشغيل نموذج محدد"])

    if algorithm_type == "تصنيف (Classification)":
        model_options = {
            "Random Forest": "rf",
            "KNN": "knn",
            "Naive Bayes": "nb",
            "SVM": "svm",
            "XGBoost": "xgboost",
            "Decision Tree": "dt",
        }
        setup = clf_setup
        compare_models = clf_compare_models
        create_model = clf_create_model
        finalize_model = clf_finalize_model
        save_model = clf_save_model
        predict_model = clf_predict_model
        get_config = clf_get_config
        pull = clf_pull
        is_classification = True
    else:
        model_options = {
            "Linear Regression": "lr",
            "Ridge": "ridge",
            "Lasso": "lasso",
            "Random Forest Regressor": "rf",
            "Gradient Boosting Regressor": "gbr",
            "Elastic Net": "en",
        }
        setup = reg_setup
        compare_models = reg_compare_models
        create_model = reg_create_model
        finalize_model = reg_finalize_model
        save_model = reg_save_model
        predict_model = reg_predict_model
        get_config = reg_get_config
        pull = reg_pull
        is_classification = False

    chosen_model_name = None
    if run_mode == "تشغيل نموذج محدد":
        chosen_model_name = st.selectbox("اختاري نموذج للتشغيل", list(model_options.keys()))

    if st.button("تشغيل التدريب"):
        with st.spinner("⏳ جاري إعداد وتجهيز البيانات..."):
            try:
                setup(data=df, target=chosen_target, normalize=True, verbose=False, html=False, session_id=123)
                st.success("✅ تم إعداد البيانات بنجاح.")
            except Exception as e:
                st.error(f"فشل setup: {e}")
                st.stop()

        # Train
        with st.spinner("⏳ جاري تدريب النموذج..."):
            try:
                if run_mode == "تشغيل كل النماذج (أفضل نموذج)":
                    best_model = compare_models()
                    save_model(best_model, "best_model")
                    model_to_use = best_model
                    st.success("✅ تم تدريب أفضل نموذج وحفظه باسم best_model.pkl")
                    st.write("أفضل نموذج:")
                    st.write(model_to_use)

                    st.write("نتائج المقارنة:")
                    st.dataframe(pull(), use_container_width=True)

                else:
                    code = model_options[chosen_model_name]
                    model_to_use = create_model(code)
                    model_to_use = finalize_model(model_to_use)
                    save_model(model_to_use, "best_model")
                    st.success("✅ تم تدريب النموذج المحدد وحفظه باسم best_model.pkl")
                    st.write("النموذج:")
                    st.write(model_to_use)

                    st.write("نتائج النموذج:")
                    st.dataframe(pull(), use_container_width=True)

            except Exception as e:
                st.error(f"فشل التدريب: {e}")
                st.stop()

        # Metrics (Train/Test)
        try:
            X_train = get_config('X_train')
            y_train = get_config('y_train')
            X_test = get_config('X_test')
            y_test = get_config('y_test')

            train_predictions = predict_model(model_to_use, data=X_train)
            test_predictions = predict_model(model_to_use, data=X_test)

            label_col = 'Label' if 'Label' in test_predictions.columns else 'prediction_label'
            if label_col not in test_predictions.columns:
                st.error("لم يتم العثور على عمود التنبؤ داخل النتائج.")
                st.stop()

            st.markdown("### تقييم الأداء")

            if is_classification:
                train_metrics = pd.DataFrame({
                    "الدقة Accuracy": [accuracy_score(y_train, train_predictions[label_col])],
                    "F1 (Weighted)": [f1_score(y_train, train_predictions[label_col], average='weighted')],
                })
                test_metrics = pd.DataFrame({
                    "الدقة Accuracy": [accuracy_score(y_test, test_predictions[label_col])],
                    "F1 (Weighted)": [f1_score(y_test, test_predictions[label_col], average='weighted')],
                })

                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("بيانات التدريب")
                    st.dataframe(train_metrics, use_container_width=True)
                with c2:
                    st.subheader("بيانات الاختبار")
                    st.dataframe(test_metrics, use_container_width=True)

                # Confusion Matrix
                cm = confusion_matrix(y_test, test_predictions[label_col])
                st.subheader("مصفوفة الالتباس (Confusion Matrix)")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", ax=ax)
                ax.set_xlabel("القيمة المتوقعة")
                ax.set_ylabel("القيمة الفعلية")
                st.pyplot(fig)

            else:
                train_metrics = pd.DataFrame({
                    "MSE": [mean_squared_error(y_train, train_predictions[label_col])],
                    "MAE": [mean_absolute_error(y_train, train_predictions[label_col])],
                    "R²": [r2_score(y_train, train_predictions[label_col])],
                })
                test_metrics = pd.DataFrame({
                    "MSE": [mean_squared_error(y_test, test_predictions[label_col])],
                    "MAE": [mean_absolute_error(y_test, test_predictions[label_col])],
                    "R²": [r2_score(y_test, test_predictions[label_col])],
                })

                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("بيانات التدريب")
                    st.dataframe(train_metrics, use_container_width=True)
                with c2:
                    st.subheader("بيانات الاختبار")
                    st.dataframe(test_metrics, use_container_width=True)

        except Exception as e:
            st.warning(f"تعذر حساب المقاييس بالكامل: {e}")

# =========================
# Download Model
# =========================
if choice == "تحميل النموذج":
    st.markdown('<div class="card"><h3>تحميل النموذج</h3><div class="small">يمكنك تحميل ملف النموذج المدرب (best_model.pkl) بعد التدريب.</div></div>', unsafe_allow_html=True)

    if os.path.exists("best_model.pkl"):
        with open("best_model.pkl", "rb") as f:
            st.download_button("⬇️ تحميل النموذج (best_model.pkl)", f, file_name="best_model.pkl")
    else:
        st.warning("لا يوجد نموذج محفوظ بعد. اذهبي إلى (تدريب النماذج) أولاً.")

# =========================
# AI Assistant (OpenRouter)
# =========================
if choice == "المساعد الذكي":
    st.markdown('<div class="card"><h3>المساعد الذكي</h3><div class="small">اسألي عن كيفية استخدام المنصة أو تحسين البيانات والنماذج.</div></div>', unsafe_allow_html=True)

    api_key = None
    try:
        api_key = st.secrets.get("OPENROUTER_API_KEY", None)
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        st.error("يرجى إضافة مفتاح OPENROUTER_API_KEY داخل Secrets أو Environment.")
        st.stop()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    SYSTEM_PROMPT = """
أنت مساعد ذكي داخل تطبيق AutoML مبني بـ Streamlit وPyCaret وydata-profiling.
مهمتك: مساعدة المستخدمين العرب على استخدام التطبيق:
- شرح أقسام: رفع البيانات، التحليل الاستكشافي، تدريب النماذج، تحميل النموذج
- إرشادات تنظيف البيانات (قيم مفقودة، ترميز، موازنة، Scaling)
- تفسير المقاييس (Accuracy, F1, MSE, MAE, R2)
أجب دائمًا بالعربية وبأسلوب واضح ومختصر.
"""

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # عرض الرسائل السابقة مع مفاتيح لمنع DuplicateWidgetID
    for i, msg in enumerate(st.session_state["chat_messages"]):
        if msg["role"] == "system":
            continue
        message(msg["content"], is_user=(msg["role"] == "user"), key=f"chat_{i}")

    user_input = st.text_input("اكتبي سؤالك هنا:", key="chat_input")

    if user_input:
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})

        with st.spinner("⏳ جاري المعالجة..."):
            try:
                response = client.chat.completions.create(
                    model="openai/gpt-3.5-turbo",
                    messages=st.session_state["chat_messages"],
                )
                reply = response.choices[0].message.content
            except Exception as e:
                reply = f"حدث خطأ أثناء الاتصال: {e}"

        st.session_state["chat_messages"].append({"role": "assistant", "content": reply})
        message(reply, key=f"chat_{len(st.session_state['chat_messages'])}")


