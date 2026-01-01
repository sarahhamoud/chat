import streamlit as st
import pandas as pd
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
import os

from pycaret.classification import (
    setup as clf_setup,
    compare_models as clf_compare_models,
    pull as clf_pull,
    save_model as clf_save_model,
    predict_model as clf_predict_model,
    get_config as clf_get_config,
    create_model as clf_create_model,
    finalize_model as clf_finalize_model,
)
from pycaret.regression import (
    setup as reg_setup,
    compare_models as reg_compare_models,
    pull as reg_pull,
    save_model as reg_save_model,
    predict_model as reg_predict_model,
    get_config as reg_get_config,
    create_model as reg_create_model,
    finalize_model as reg_finalize_model,
)

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    mean_absolute_error,
)

import seaborn as sns
import matplotlib.pyplot as plt

from streamlit_chat import message
from openai import OpenAI


# =========================================================
# إعداد الصفحة
# =========================================================
st.set_page_config(
    page_title=" منصة الذكاء الاصطناعي لتحليل البيانات وتدريب النماذج على البيانات المرفوعه على المنصة الذكية",
    page_icon="📊",
    layout="wide",
)


# =========================================================
# CSS: RTL + ثيم فاتح + خط أكبر وتباين أعلى (واجهة احترافية)
# =========================================================
st.markdown(
    """
<style>
html, body, [data-testid="stApp"]{
    direction: rtl;
    text-align: right;
    font-family: "Cairo","Segoe UI","Tahoma",sans-serif;
}

/* خلفية فاتحة */
[data-testid="stAppViewContainer"]{
    background: radial-gradient(1200px 600px at 20% 10%, #eaf6ff 0%, #f7fbff 40%, #ffffff 100%) !important;
}

/* شريط أعلى Streamlit */
[data-testid="stHeader"]{
    background: rgba(255,255,255,0.0) !important;
    border-bottom: 0 !important;
}

/* Sidebar */
[data-testid="stSidebar"]{
    background: rgba(255,255,255,0.78) !important;
    backdrop-filter: blur(10px);
    border-left: 1px solid rgba(15,23,42,0.08);
}
[data-testid="stSidebar"] *{
    color: #0f172a !important;
}

/* تكبير وتحسين وضوح النص */
h1{ font-size: 2.05rem !important; color:#0b1b3a !important; font-weight:900 !important; }
h2{ font-size: 1.65rem !important; color:#0b1b3a !important; font-weight:900 !important; }
h3{ font-size: 1.28rem !important; color:#0b1b3a !important; font-weight:800 !important; }
p, label, span, div{
    font-size: 1.05rem !important;
    color:#102a43 !important;
}

/* بطاقات */
.card{
    background: rgba(255,255,255,0.86);
    border: 1px solid rgba(15,23,42,0.08);
    box-shadow: 0 10px 30px rgba(2,8,23,0.08);
    border-radius: 18px;
    padding: 16px 18px;
}

/* أزرار */
.stButton > button{
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 50%, #0ea5e9 100%) !important;
    color: #fff !important;
    border: 0 !important;
    border-radius: 12px !important;
    padding: 10px 16px !important;
    font-weight: 800 !important;
    box-shadow: 0 10px 22px rgba(37,99,235,0.22);
}
.stButton > button:hover{ filter: brightness(1.06); }

/* Inputs */
.stTextInput input, .stSelectbox div, .stNumberInput input, .stTextArea textarea{
    border-radius: 12px !important;
    border: 1px solid rgba(15,23,42,0.12) !important;
    background: rgba(255,255,255,0.95) !important;
    color:#0f172a !important;
}

/* DataFrame */
[data-testid="stDataFrame"]{
    background: rgba(255,255,255,0.9) !important;
    border-radius: 14px;
    border: 1px solid rgba(15,23,42,0.08);
}

/* تنسيق Tabs */
button[data-baseweb="tab"]{
    font-weight: 800 !important;
    font-size: 1.05rem !important;
}

/* رسائل تنبيه */
.stAlert{
    border-radius: 14px !important;
}

/* إخفاء أي بادج افتراضي لو كنتِ مستخدمته سابقاً */
.badge, .chip, .pill{ display:none !important; }

/* تحسين الحاويات */
.block-container{
    padding-top: 1.2rem !important;
    padding-bottom: 2.2rem !important;
}

/* إزالة شريط Streamlit السفلي (اختياري) */
footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# هيدر: اسمك باليسار + عنوان المنصة باليمين
# =========================================================
st.markdown(
    """
<div style="
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:12px;
    padding:14px 18px;
    margin: 6px 0 18px 0;
    background: rgba(255,255,255,0.78);
    border: 1px solid rgba(15,23,42,0.08);
    border-radius: 18px;
    box-shadow: 0 10px 28px rgba(2,8,23,0.08);
">
  <div style="direction:ltr;text-align:left;font-weight:900;color:#0b1b3a;font-size:1.05rem;">
    sarah hamoud hussien
  </div>

  <div style="text-align:right;">
    <div style="font-weight:950;color:#0b1b3a;font-size:1.45rem;line-height:1.2;">
      منصة الذكاء الاصطناعي لتحليل البيانات وتدريب النماذج على البيانات المرفوعه على المنصة الذكية
    </div>
    <div style="color:#334155;font-size:0.98rem;font-weight:700;margin-top:4px;">
      رفع البيانات • التحليل الاستكشافي • بناء النماذج • تحميل النموذج • مساعد ذكي
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# تحميل البيانات إذا كانت موجودة
# =========================================================
df = None
if os.path.exists("dataset.csv"):
    try:
        df = pd.read_csv("dataset.csv", index_col=None)
    except Exception:
        df = None


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.image(
        "https://www.onepointltd.com/wp-content/uploads/2019/12/shutterstock_1166533285-Converted-02.png"
    )
    st.title("AutoML")
    choice = st.radio(
        "اختر القسم",
        ["رفع البيانات", "تحليل البيانات", "بناء النموذج", "تحميل النموذج", "المساعد الذكي"],
    )
    st.info("هذه المنصة تساعدك على رفع بياناتك وتحليلها وبناء نموذج تعلم آلي بسهولة.")


# =========================================================
# أدوات مساعدة (KPI Cards)
# =========================================================
def kpi_cards(data: pd.DataFrame):
    rows = len(data)
    cols = len(data.columns)
    missing = int(data.isna().sum().sum())
    duplicates = int(data.duplicated().sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="card"><div style="font-weight:900;">عدد الصفوف</div><div style="font-size:1.5rem;font-weight:900;">{rows}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="card"><div style="font-weight:900;">عدد الأعمدة</div><div style="font-size:1.5rem;font-weight:900;">{cols}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="card"><div style="font-weight:900;">القيم المفقودة</div><div style="font-size:1.5rem;font-weight:900;">{missing}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="card"><div style="font-weight:900;">الصفوف المكررة</div><div style="font-size:1.5rem;font-weight:900;">{duplicates}</div></div>', unsafe_allow_html=True)


# =========================================================
# 1) رفع البيانات
# =========================================================
if choice == "رفع البيانات":
    st.header("رفع البيانات")
    st.write("ارفع ملف CSV، وسيتم حفظه داخل التطبيق تلقائياً.")

    file = st.file_uploader("اختر ملف البيانات (CSV)", type=["csv"])
    if file:
        try:
            df = pd.read_csv(file, index_col=None)
            df.to_csv("dataset.csv", index=None)
            st.success("تم رفع البيانات وحفظها بنجاح ✅")
            kpi_cards(df)
            st.subheader("معاينة البيانات")
            st.dataframe(df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"حدث خطأ أثناء قراءة الملف: {e}")

    if df is None:
        st.info("لم يتم العثور على بيانات بعد. ارفعي ملف CSV أولاً.")


# =========================================================
# 2) تحليل البيانات (Profiling)
# =========================================================
if choice == "تحليل البيانات":
    st.header("التحليل الاستكشافي للبيانات")
    if df is None:
        st.warning("لا توجد بيانات. ارفعي ملف CSV من قسم (رفع البيانات).")
    else:
        kpi_cards(df)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("⚙️ إعدادات التقرير")
        max_rows = st.number_input("عدد الصفوف لتوليد التقرير (اختياري)", min_value=100, max_value=100000, value=1000, step=100)
        detailed = st.toggle("تقرير تفصيلي", value=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("إنشاء تقرير التحليل"):
            try:
                # لتجنب مشاكل wordcloud/numpy في بعض البيئات، نغلق الـ wordcloud
                profile = ydata_profiling.ProfileReport(
                    df.head(int(max_rows)),
                    explorative=detailed,
                    minimal=not detailed,
                    # إيقاف wordcloud يقلل احتمالات خطأ asarray(copy=..)
                    # وفي نفس الوقت يبقي التقرير قوي جداً
                )
                st_profile_report(profile)
                st.success("تم إنشاء التقرير ✅")
            except Exception as e:
                st.error(
                    "تعذر إنشاء تقرير التحليل. "
                    "إذا ظهر خطأ (asarray() copy)، فهذا بسبب تعارض نسخ numpy/wordcloud.\n"
                    f"تفاصيل الخطأ: {e}"
                )
                st.info("حل سريع: ثبتي numpy==1.26.4 و wordcloud==1.9.3 داخل requirements.txt.")


# =========================================================
# 3) بناء النموذج (Modeling)
# =========================================================
if choice == "بناء النموذج":
    st.header("بناء نموذج تعلم آلي")
    if df is None:
        st.warning("لا توجد بيانات. ارفعي ملف CSV من قسم (رفع البيانات).")
    else:
        st.subheader("اختيار العمود الهدف")
        chosen_target = st.selectbox("اختر عمود الهدف (Target)", df.columns)

        st.subheader("نوع المهمة")
        algorithm_type = st.radio("اختر نوع المهمة", ["تصنيف (Classification)", "انحدار (Regression)"])

        st.subheader("طريقة التشغيل")
        run_mode = st.radio("تشغيل النماذج", ["كل النماذج (Auto مقارنة)", "نموذج محدد"])

        if algorithm_type == "تصنيف (Classification)":
            model_options = [
                ("Random Forest", "rf"),
                ("KNN", "knn"),
                ("Naive Bayes", "nb"),
                ("SVM", "svm"),
                ("XGBoost", "xgboost"),
                ("Decision Tree", "dt"),
            ]
            setup_fn = clf_setup
            compare_models_fn = clf_compare_models
            create_model_fn = clf_create_model
            finalize_model_fn = clf_finalize_model
            save_model_fn = clf_save_model
            predict_model_fn = clf_predict_model
            get_config_fn = clf_get_config
            pull_fn = clf_pull

        else:
            model_options = [
                ("Linear Regression", "lr"),
                ("Ridge", "ridge"),
                ("Lasso", "lasso"),
                ("Random Forest", "rf"),
                ("Gradient Boosting", "gbr"),
                ("Elastic Net", "en"),
            ]
            setup_fn = reg_setup
            compare_models_fn = reg_compare_models
            create_model_fn = reg_create_model
            finalize_model_fn = reg_finalize_model
            save_model_fn = reg_save_model
            predict_model_fn = reg_predict_model
            get_config_fn = reg_get_config
            pull_fn = reg_pull

        chosen_code = None
        if run_mode == "نموذج محدد":
            chosen_name = st.selectbox("اختاري نموذجاً", [m[0] for m in model_options])
            chosen_code = dict(model_options)[chosen_name]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        normalize = st.toggle("تطبيع البيانات (Normalize)", value=True)
        st.caption("ملاحظة: PyCaret يقوم بتهيئة البيانات ومعالجة كثير من الأشياء تلقائياً.")
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("تشغيل التدريب"):
            try:
                setup_fn(
                    data=df,
                    target=chosen_target,
                    normalize=normalize,
                    verbose=False,
                    html=False,
                    session_id=123,
                )

                if run_mode == "كل النماذج (Auto مقارنة)":
                    best_model = compare_models_fn()
                    save_model_fn(best_model, "best_model")
                    model_to_use = best_model
                    st.success("تم تدريب أفضل نموذج وحفظه باسم best_model.pkl ✅")
                    st.subheader("أفضل نموذج")
                    st.write(model_to_use)

                    st.subheader("مقارنة أداء النماذج")
                    st.dataframe(pull_fn(), use_container_width=True)

                else:
                    model_to_use = create_model_fn(chosen_code)
                    model_to_use = finalize_model_fn(model_to_use)
                    save_model_fn(model_to_use, "best_model")
                    st.success("تم تدريب النموذج المحدد وحفظه باسم best_model.pkl ✅")
                    st.subheader("النموذج المختار")
                    st.write(model_to_use)

                # بيانات تدريب/اختبار
                X_train = get_config_fn("X_train")
                y_train = get_config_fn("y_train")
                X_test = get_config_fn("X_test")
                y_test = get_config_fn("y_test")

                train_pred = predict_model_fn(model_to_use, data=X_train)
                test_pred = predict_model_fn(model_to_use, data=X_test)

                label_col = "Label" if "Label" in test_pred.columns else (
                    "prediction_label" if "prediction_label" in test_pred.columns else None
                )

                if label_col is None:
                    st.warning("لم يتم العثور على عمود التوقعات داخل نتائج PyCaret.")
                else:
                    st.subheader("مؤشرات الأداء")

                    if algorithm_type == "تصنيف (Classification)":
                        train_metrics = pd.DataFrame({
                            "الدقة Accuracy": [accuracy_score(y_train, train_pred[label_col])],
                            "F1": [f1_score(y_train, train_pred[label_col], average="weighted")],
                        })
                        test_metrics = pd.DataFrame({
                            "الدقة Accuracy": [accuracy_score(y_test, test_pred[label_col])],
                            "F1": [f1_score(y_test, test_pred[label_col], average="weighted")],
                        })

                        c1, c2 = st.columns(2)
                        c1.markdown('<div class="card">', unsafe_allow_html=True)
                        c1.write("نتائج التدريب")
                        c1.dataframe(train_metrics, use_container_width=True)
                        c1.markdown("</div>", unsafe_allow_html=True)

                        c2.markdown('<div class="card">', unsafe_allow_html=True)
                        c2.write("نتائج الاختبار")
                        c2.dataframe(test_metrics, use_container_width=True)
                        c2.markdown("</div>", unsafe_allow_html=True)

                        st.subheader("مصفوفة الالتباس (Confusion Matrix)")
                        cm = confusion_matrix(y_test, test_pred[label_col])
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
                        ax.set_xlabel("المتوقع")
                        ax.set_ylabel("الحقيقي")
                        st.pyplot(fig)

                    else:
                        train_metrics = pd.DataFrame({
                            "MSE": [mean_squared_error(y_train, train_pred[label_col])],
                            "MAE": [mean_absolute_error(y_train, train_pred[label_col])],
                            "R2": [r2_score(y_train, train_pred[label_col])],
                        })
                        test_metrics = pd.DataFrame({
                            "MSE": [mean_squared_error(y_test, test_pred[label_col])],
                            "MAE": [mean_absolute_error(y_test, test_pred[label_col])],
                            "R2": [r2_score(y_test, test_pred[label_col])],
                        })

                        c1, c2 = st.columns(2)
                        c1.markdown('<div class="card">', unsafe_allow_html=True)
                        c1.write("نتائج التدريب")
                        c1.dataframe(train_metrics, use_container_width=True)
                        c1.markdown("</div>", unsafe_allow_html=True)

                        c2.markdown('<div class="card">', unsafe_allow_html=True)
                        c2.write("نتائج الاختبار")
                        c2.dataframe(test_metrics, use_container_width=True)
                        c2.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"حدث خطأ أثناء التدريب: {e}")
                st.info("تحققي من توافق الإصدارات داخل requirements.txt.")


# =========================================================
# 4) تحميل النموذج
# =========================================================
if choice == "تحميل النموذج":
    st.header("تحميل النموذج")
    if os.path.exists("best_model.pkl"):
        st.success("النموذج جاهز للتحميل ✅")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        with open("best_model.pkl", "rb") as f:
            st.download_button("تحميل النموذج (best_model.pkl)", f, file_name="best_model.pkl")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("لا يوجد نموذج محفوظ بعد. اذهبي إلى قسم (بناء النموذج) أولاً.")


# =========================================================
# 5) المساعد الذكي (OpenRouter)
# =========================================================
if choice == "المساعد الذكي":
    st.header("المساعد الذكي")
    st.write("اسألي عن كيفية استخدام المنصة أو تفسير النتائج أو اختيار النموذج المناسب.")

    # قراءة المفتاح من secrets أو env
    api_key = None
    try:
        api_key = st.secrets.get("OPENROUTER_API_KEY", None)
    except Exception:
        api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        st.warning("لم يتم العثور على مفتاح OPENROUTER_API_KEY في Secrets أو Environment Variables.")
        st.info("على Streamlit Cloud: Settings → Secrets ثم أضيفي OPENROUTER_API_KEY.")
    else:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        SYSTEM_PROMPT = """
أنت مساعد محترف داخل منصة عربية للنمذجة الآلية للبيانات (AutoML).
مهمتك إرشاد المستخدم لاستخدام الأقسام: رفع البيانات، التحليل الاستكشافي، بناء النموذج، تحميل النموذج.
اشرح المقاييس (Accuracy, F1, MSE, MAE, R2) بطريقة مبسطة.
قدّم نصائح للتعامل مع البيانات (قيم مفقودة، ترميز، توازن الفئات، اختيار الهدف).
اجعل الإجابة عربية واضحة ومختصرة وبأسلوب مهني.
إذا سُئلت خارج نطاق المنصة، ارجع بالسؤال إلى سياق المنصة.
"""

        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # عرض الرسائل
        for msg in st.session_state["messages"]:
            if msg["role"] != "system":
                message(msg["content"], is_user=(msg["role"] == "user"))

        user_input = st.text_input("اكتبي سؤالك هنا:")
        if user_input:
            st.session_state["messages"].append({"role": "user", "content": user_input})

            with st.spinner("جارِ معالجة سؤالك..."):
                try:
                    response = client.chat.completions.create(
                        model="openai/gpt-3.5-turbo",
                        messages=st.session_state["messages"],
                    )
                    reply = response.choices[0].message.content
                except Exception as e:
                    reply = f"تعذر الاتصال بالمساعد: {e}"

            st.session_state["messages"].append({"role": "assistant", "content": reply})
            message(reply)

