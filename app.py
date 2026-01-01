import os
import pandas as pd
import streamlit as st
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report

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


# =========================
# إعدادات الصفحة
# =========================
st.set_page_config(
    page_title="منصة الذكاء الاصطناعي  ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# CSS: RTL + تصميم فاتح/احترافي
# =========================
st.markdown(
    """
<style>
/* RTL عام */
html, body, [data-testid="stApp"] {
    direction: RTL;
    text-align: right;
    font-family: "Segoe UI", Tahoma, Arial, sans-serif;
}

/* خلفية عامة فاتحة */
[data-testid="stAppViewContainer"]{
    background: radial-gradient(1200px 800px at 80% 10%, rgba(120,180,255,0.18), transparent 55%),
                radial-gradient(900px 700px at 10% 25%, rgba(255,220,120,0.12), transparent 55%),
                linear-gradient(135deg, #f7fbff 0%, #eef5ff 35%, #f8fafc 100%);
}

/* الشريط العلوي */
.header-wrap{
    background: linear-gradient(90deg, rgba(255,255,255,0.75), rgba(255,255,255,0.55));
    border: 1px solid rgba(20,60,120,0.10);
    box-shadow: 0 10px 30px rgba(10,30,60,0.08);
    backdrop-filter: blur(12px);
    border-radius: 18px;
    padding: 18px 22px;
    margin-bottom: 18px;
}

.header-grid{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
}

.brand-title{
    font-size: 28px;
    font-weight: 800;
    color: #0b1f3b;
    letter-spacing: 0.2px;
}

.brand-sub{
    margin-top: 4px;
    font-size: 14px;
    color: #365a86;
}

.user-name{
    direction: ltr;
    text-align: left;
    font-size: 14px;
    font-weight: 700;
    color: #0b1f3b;
    padding: 10px 14px;
    border-radius: 14px;
    border: 1px solid rgba(20,60,120,0.12);
    background: rgba(255,255,255,0.65);
}

/* بطاقات */
.card{
    background: rgba(255,255,255,0.75);
    border: 1px solid rgba(20,60,120,0.10);
    box-shadow: 0 10px 30px rgba(10,30,60,0.06);
    border-radius: 18px;
    padding: 18px;
}

.kpi{
    background: rgba(255,255,255,0.8);
    border: 1px solid rgba(20,60,120,0.12);
    border-radius: 16px;
    padding: 16px;
    box-shadow: 0 10px 25px rgba(10,30,60,0.05);
}

.kpi .label{
    font-size: 14px;
    color: #345b86;
    margin-bottom: 6px;
}
.kpi .value{
    font-size: 24px;
    font-weight: 800;
    color: #0b1f3b;
}

/* عناوين */
h1, h2, h3{
    color: #0b1f3b !important;
}
p, span, label{
    color: #203a5a !important;
    font-size: 15px !important;
}

/* تكبير عناصر الإدخال */
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea{
    font-size: 15px !important;
}

/* الأزرار */
.stButton button{
    border-radius: 14px !important;
    padding: 10px 16px !important;
    font-weight: 700 !important;
}

/* الشريط الجانبي */
section[data-testid="stSidebar"]{
    background: linear-gradient(180deg, rgba(255,255,255,0.80), rgba(255,255,255,0.65));
    border-right: 1px solid rgba(20,60,120,0.10);
}

/* مسافة لطيفة */
.block-container{
    padding-top: 16px;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# هيدر علوي (يمين عنوان + يسار اسمك)
# =========================
st.markdown(
    """
<div class="header-wrap">
  <div class="header-grid">
    <div class="user-name">sarah hamoud hussien</div>
    <div>
      <div class="brand-title">منصة الذكاء الاصطناعي وتحليل البيانات   </div>
      <div class="brand-sub">رفع البيانات • التحليل الاستكشافي • بناء النماذج • تحميل النموذج • المساعد</div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# =========================
# تحميل الداتا إن كانت موجودة
# =========================
df = None
if os.path.exists("dataset.csv"):
    try:
        df = pd.read_csv("dataset.csv")
    except Exception:
        df = None

# =========================
# Sidebar: قائمة واضحة
# =========================
with st.sidebar:
    st.markdown("### التحكم")
    choice = st.radio(
        "اختر القسم",
        ["رفع البيانات", "تحليل البيانات", "بناء النماذج", "تحميل النموذج", "المساعد"],
        index=0,
    )
    st.caption("تطبيق يساعدك على استكشاف البيانات وبناء نماذج تعلم آلة بسهولة.")

# =========================
# أدوات مساعدة
# =========================
def require_df():
    if df is None or df.empty:
        st.warning("لا توجد بيانات محمّلة بعد. الرجاء الذهاب إلى قسم **رفع البيانات** أولاً.")
        st.stop()

def kpi_row(d: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="kpi"><div class="label">عدد الصفوف</div><div class="value">{}</div></div>'.format(len(d)), unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="kpi"><div class="label">عدد الأعمدة</div><div class="value">{}</div></div>'.format(d.shape[1]), unsafe_allow_html=True)
    with c3:
        missing = int(d.isna().sum().sum())
        st.markdown('<div class="kpi"><div class="label">القيم المفقودة</div><div class="value">{}</div></div>'.format(missing), unsafe_allow_html=True)
    with c4:
        dup = int(d.duplicated().sum())
        st.markdown('<div class="kpi"><div class="label">الصفوف المكررة</div><div class="value">{}</div></div>'.format(dup), unsafe_allow_html=True)

# =========================
# 1) رفع البيانات
# =========================
if choice == "رفع البيانات":
    st.markdown("## رفع البيانات")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    file = st.file_uploader("ارفع ملف البيانات بصيغة CSV", type=["csv"])
    if file:
        try:
            df = pd.read_csv(file)
            df.to_csv("dataset.csv", index=False)
            st.success("تم رفع الملف بنجاح ✅")
            kpi_row(df)
            st.markdown("### معاينة أول 10 صفوف")
            st.dataframe(df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"تعذر قراءة الملف: {e}")
    else:
        if df is not None:
            st.info("تم العثور على ملف بيانات محفوظ مسبقاً.")
            kpi_row(df)
            st.dataframe(df.head(10), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# 2) تحليل البيانات (Profiling)
# =========================
elif choice == "تحليل البيانات":
    st.markdown("## التحليل الاستكشافي للبيانات")
    require_df()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    kpi_row(df)

    st.markdown("### إعدادات التقرير")
    colA, colB, colC = st.columns([1, 1, 1])

    with colA:
        minimal = st.toggle("وضع سريع (أخف)", value=True, help="يقلل بعض الرسوم لتسريع التقرير")
    with colB:
        samples = st.number_input("عدد الصفوف للتقرير (اختياري)", min_value=0, max_value=200000, value=0, step=1000)
    with colC:
        run_report = st.button("إنشاء التقرير")

    st.divider()

    if run_report:
        try:
            data_for_report = df
            if samples and samples > 0:
                data_for_report = df.head(int(samples))

            # تقليل مشاكل wordcloud عند وجود أعمدة نصية كثيفة
            profile = ydata_profiling.ProfileReport(
                data_for_report,
                explorative=True,
                minimal=minimal,
            )
            st_profile_report(profile)
        except Exception as e:
            st.error("تعذر إنشاء تقرير التحليل.")
            st.code(str(e))

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# 3) بناء النماذج (PyCaret)
# =========================
elif choice == "بناء النماذج":
    st.markdown("## بناء النماذج")
    require_df()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    kpi_row(df)

    st.markdown("### إعدادات النموذج")
    col1, col2, col3 = st.columns([1.2, 1, 1])

    with col1:
        chosen_target = st.selectbox("اختر عمود الهدف (Target)", df.columns)

    with col2:
        algorithm_type = st.radio("نوع المشكلة", ["تصنيف", "انحدار"], horizontal=True)

    with col3:
        run_mode = st.radio("طريقة التشغيل", ["أفضل نموذج تلقائي", "نموذج محدد"], horizontal=True)

    st.divider()

    if algorithm_type == "تصنيف":
        model_options = [
            ("Random Forest", "rf"),
            ("KNN", "knn"),
            ("Naive Bayes", "nb"),
            ("SVM", "svm"),
            ("XGBoost", "xgboost"),
            ("Decision Tree", "dt"),
        ]
        setup_fn = clf_setup
        compare_fn = clf_compare_models
        create_fn = clf_create_model
        finalize_fn = clf_finalize_model
        save_fn = clf_save_model
        predict_fn = clf_predict_model
        get_cfg = clf_get_config
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
        compare_fn = reg_compare_models
        create_fn = reg_create_model
        finalize_fn = reg_finalize_model
        save_fn = reg_save_model
        predict_fn = reg_predict_model
        get_cfg = reg_get_config
        pull_fn = reg_pull

    chosen_model_code = None
    if run_mode == "نموذج محدد":
        chosen_model_name = st.selectbox("اختر النموذج", [m[0] for m in model_options])
        chosen_model_code = dict(model_options)[chosen_model_name]

    normalize = st.toggle("تطبيق Normalization", value=True)
    session_id = st.number_input("Session ID", min_value=1, max_value=999999, value=123, step=1)

    run_btn = st.button("تشغيل وبناء النموذج")

    if run_btn:
        try:
            st.info("جارِ إعداد البيئة التدريبية...")
            setup_fn(
                data=df,
                target=chosen_target,
                normalize=normalize,
                verbose=False,
                html=False,
                session_id=int(session_id),
            )

            if run_mode == "أفضل نموذج تلقائي":
                st.info("جارِ مقارنة النماذج واختيار الأفضل...")
                best_model = compare_fn()
                model_to_use = best_model
                save_fn(model_to_use, "best_model")
                st.success("تم تدريب وحفظ أفضل نموذج باسم: best_model.pkl ✅")
                st.markdown("### مقارنة الأداء")
                st.dataframe(pull_fn(), use_container_width=True)

            else:
                st.info("جارِ تدريب النموذج المحدد...")
                model_to_use = create_fn(chosen_model_code)
                model_to_use = finalize_fn(model_to_use)
                save_fn(model_to_use, "best_model")
                st.success("تم تدريب وحفظ النموذج باسم: best_model.pkl ✅")

            # بيانات التدريب والاختبار من PyCaret
            X_train = get_cfg("X_train")
            y_train = get_cfg("y_train")
            X_test = get_cfg("X_test")
            y_test = get_cfg("y_test")

            train_pred = predict_fn(model_to_use, data=X_train)
            test_pred = predict_fn(model_to_use, data=X_test)

            label_col = "Label" if "Label" in test_pred.columns else ("prediction_label" if "prediction_label" in test_pred.columns else None)
            if label_col is None:
                st.error("تعذر العثور على عمود التنبؤ داخل النتائج.")
                st.stop()

            st.divider()
            st.markdown("### مؤشرات الأداء")

            if algorithm_type == "تصنيف":
                train_metrics = pd.DataFrame(
                    {
                        "الدقة Accuracy": [accuracy_score(y_train, train_pred[label_col])],
                        "F1 (Weighted)": [f1_score(y_train, train_pred[label_col], average="weighted")],
                    }
                )
                test_metrics = pd.DataFrame(
                    {
                        "الدقة Accuracy": [accuracy_score(y_test, test_pred[label_col])],
                        "F1 (Weighted)": [f1_score(y_test, test_pred[label_col], average="weighted")],
                    }
                )

                cA, cB = st.columns(2)
                with cA:
                    st.markdown("#### تدريب")
                    st.dataframe(train_metrics, use_container_width=True)
                with cB:
                    st.markdown("#### اختبار")
                    st.dataframe(test_metrics, use_container_width=True)

                cm = confusion_matrix(y_test, test_pred[label_col])
                st.markdown("#### مصفوفة الالتباس (اختبار)")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", ax=ax)
                ax.set_xlabel("المتوقع")
                ax.set_ylabel("الحقيقي")
                st.pyplot(fig)

            else:
                train_metrics = pd.DataFrame(
                    {
                        "MSE": [mean_squared_error(y_train, train_pred[label_col])],
                        "MAE": [mean_absolute_error(y_train, train_pred[label_col])],
                        "R2": [r2_score(y_train, train_pred[label_col])],
                    }
                )
                test_metrics = pd.DataFrame(
                    {
                        "MSE": [mean_squared_error(y_test, test_pred[label_col])],
                        "MAE": [mean_absolute_error(y_test, test_pred[label_col])],
                        "R2": [r2_score(y_test, test_pred[label_col])],
                    }
                )

                cA, cB = st.columns(2)
                with cA:
                    st.markdown("#### تدريب")
                    st.dataframe(train_metrics, use_container_width=True)
                with cB:
                    st.markdown("#### اختبار")
                    st.dataframe(test_metrics, use_container_width=True)

        except Exception as e:
            st.error("حدث خطأ أثناء التدريب.")
            st.code(str(e))

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# 4) تحميل النموذج
# =========================
elif choice == "تحميل النموذج":
    st.markdown("## تحميل النموذج")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if os.path.exists("best_model.pkl"):
        with open("best_model.pkl", "rb") as f:
            st.success("النموذج جاهز للتحميل ✅")
            st.download_button("تحميل النموذج (best_model.pkl)", f, file_name="best_model.pkl")
    else:
        st.warning("لا يوجد نموذج محفوظ حالياً. الرجاء بناء نموذج أولاً من قسم **بناء النماذج**.")

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# 5) المساعد
# =========================
elif choice == "المساعد":
    st.markdown("## المساعد")

    st.markdown('<div class="card">', unsafe_allow_html=True)

    # مفتاح OpenRouter من Secrets أو ENV
    api_key = None
    if "OPENROUTER_API_KEY" in st.secrets:
        api_key = st.secrets["OPENROUTER_API_KEY"]
    else:
        api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        st.error("لم يتم العثور على مفتاح OPENROUTER_API_KEY في Secrets أو Environment Variables.")
        st.stop()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    SYSTEM_PROMPT = """
أنت مساعد داخل منصة نمذجة آلية للبيانات.
مهمتك مساعدة المستخدمين في:
- رفع البيانات CSV
- تفسير التحليل الاستكشافي (Profiling)
- اختيار نوع المشكلة (تصنيف/انحدار)
- تفسير المقاييس Accuracy/F1/MSE/MAE/R2
- اقتراح تحسينات على جودة البيانات (قيم مفقودة، ترميز، موازنة الفئات)
اجب باللغة العربية وبأسلوب واضح ومختصر.
"""

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    # عرض الرسائل (مع keys فريدة لمنع DuplicateWidgetID)
    for i, msg in enumerate(st.session_state["chat_messages"]):
        if msg["role"] == "system":
            continue
        message(
            msg["content"],
            is_user=(msg["role"] == "user"),
            key=f"chat_{i}_{msg['role']}"
        )

    st.divider()

    user_input = st.text_input("اكتبي سؤالك هنا:", key="chat_input")

    if user_input:
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})

        with st.spinner("جارِ تجهيز الرد..."):
            try:
                response = client.chat.completions.create(
                    model="openai/gpt-3.5-turbo",
                    messages=st.session_state["chat_messages"],
                )
                reply = response.choices[0].message.content
            except Exception as e:
                reply = f"تعذر الاتصال بالمساعد: {e}"

        st.session_state["chat_messages"].append({"role": "assistant", "content": reply})

        # تفريغ الإدخال + تحديث الواجهة
        st.session_state["chat_input"] = ""
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

