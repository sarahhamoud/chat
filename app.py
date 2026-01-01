import os
import streamlit as st
import pandas as pd
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report

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

from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error
)

import seaborn as sns
import matplotlib.pyplot as plt

from openai import OpenAI
from streamlit_chat import message


# =========================
# إعداد الصفحة
# =========================
st.set_page_config(page_title="منصة النمذجة الآلية", layout="wide")

# =========================
# تنسيق RTL + واجهة هندسية احترافية
# =========================
st.markdown(
    """
    <style>
    :root{
        --bg0:#0b1220;
        --bg1:#0f1a33;
        --card:#0e1730;
        --card2:#101c38;
        --line:#223057;
        --text:#e6edf6;
        --muted:#a7b4cf;
        --accent:#2dd4bf;   /* تركواز برمجي */
        --accent2:#60a5fa;  /* أزرق */
        --warn:#f59e0b;
        --ok:#22c55e;
        --danger:#ef4444;
        --shadow: 0 12px 32px rgba(0,0,0,.35);
        --radius: 16px;
    }

    html, body, [data-testid="stApp"]{
        background: radial-gradient(1200px 700px at 80% -10%, rgba(96,165,250,.25), transparent 60%),
                    radial-gradient(900px 500px at 20% 0%, rgba(45,212,191,.20), transparent 55%),
                    linear-gradient(180deg, var(--bg0), var(--bg1));
        color: var(--text);
        direction: RTL;
        text-align: right;
        font-family: "Segoe UI", "Tahoma", sans-serif;
    }

    /* إخفاء شريط Streamlit الافتراضي */
    header, footer {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}

    /* تحسين المسافات */
    .block-container{
        padding-top: 24px !important;
        padding-bottom: 28px !important;
    }

    /* العناوين */
    h1,h2,h3,h4{
        letter-spacing: .2px;
        color: var(--text);
    }

    /* الحاويات */
    .app-shell{
        border: 1px solid rgba(34,48,87,.7);
        background: linear-gradient(180deg, rgba(14,23,48,.65), rgba(16,28,56,.45));
        border-radius: var(--radius);
        box-shadow: var(--shadow);
        padding: 18px 18px 10px 18px;
        margin-bottom: 16px;
        position: relative;
        overflow: hidden;
    }

    .app-shell::before{
        content:"";
        position:absolute;
        top:-120px; left:-120px;
        width:240px; height:240px;
        background: radial-gradient(circle, rgba(45,212,191,.22), transparent 60%);
        filter: blur(2px);
    }

    .app-shell::after{
        content:"";
        position:absolute;
        bottom:-140px; right:-140px;
        width:280px; height:280px;
        background: radial-gradient(circle, rgba(96,165,250,.20), transparent 60%);
        filter: blur(2px);
    }

    .topbar{
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:12px;
        margin-bottom: 12px;
        position: relative;
        z-index: 1;
    }

    .brand{
        display:flex;
        flex-direction:column;
        gap:4px;
    }

    .brand-title{
        font-size: 22px;
        font-weight: 700;
        line-height: 1.2;
    }

    .brand-sub{
        font-size: 13px;
        color: var(--muted);
    }

    .badge{
        display:inline-flex;
        align-items:center;
        gap:8px;
        font-size: 12px;
        color: var(--muted);
        border: 1px solid rgba(34,48,87,.8);
        background: rgba(14,23,48,.55);
        padding: 8px 10px;
        border-radius: 999px;
        position: relative;
        z-index: 1;
        white-space: nowrap;
    }

    .dot{
        width:10px;height:10px;border-radius:999px;
        background: var(--accent);
        box-shadow: 0 0 0 6px rgba(45,212,191,.10);
    }

    .grid{
        display:grid;
        grid-template-columns: repeat(12, 1fr);
        gap: 12px;
        position: relative;
        z-index: 1;
    }

    .card{
        border: 1px solid rgba(34,48,87,.75);
        background: linear-gradient(180deg, rgba(14,23,48,.70), rgba(16,28,56,.55));
        border-radius: var(--radius);
        box-shadow: 0 10px 28px rgba(0,0,0,.25);
        padding: 14px 14px;
        overflow: hidden;
    }

    .card-title{
        font-size: 14px;
        color: var(--muted);
        margin-bottom: 6px;
    }

    .card-value{
        font-size: 18px;
        font-weight: 700;
    }

    .hr{
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(34,48,87,.9), transparent);
        margin: 10px 0 6px 0;
    }

    /* السايدبار */
    [data-testid="stSidebar"]{
        background: linear-gradient(180deg, rgba(11,18,32,.92), rgba(15,26,51,.92)) !important;
        border-left: 1px solid rgba(34,48,87,.65);
    }
    [data-testid="stSidebar"] *{
        color: var(--text) !important;
    }

    /* عناصر الإدخال RTL */
    label, .stRadio, .stSelectbox, .stSlider, .stTextInput, .stTextArea{
        direction: RTL !important;
        text-align: right !important;
    }

    /* الأزرار */
    .stButton>button{
        width: 100%;
        border-radius: 14px;
        border: 1px solid rgba(34,48,87,.9);
        background: linear-gradient(180deg, rgba(45,212,191,.18), rgba(96,165,250,.10));
        color: var(--text);
        padding: 10px 12px;
        font-weight: 700;
        transition: .2s ease-in-out;
    }
    .stButton>button:hover{
        transform: translateY(-1px);
        border-color: rgba(45,212,191,.75);
        box-shadow: 0 10px 22px rgba(0,0,0,.25);
    }

    /* تبويبات */
    button[data-baseweb="tab"]{
        font-weight: 700 !important;
    }

    /* Dataframe */
    .stDataFrame, .stTable { direction: RTL; }

    /* تنبيه/نجاح */
    .stAlert{
        border-radius: 14px;
        border: 1px solid rgba(34,48,87,.75);
        background: rgba(14,23,48,.55);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# قراءة بيانات محفوظة
# =========================
df = None
if os.path.exists("dataset.csv"):
    try:
        df = pd.read_csv("dataset.csv", index_col=None)
    except Exception as e:
        st.error(f"تعذر قراءة ملف البيانات المحفوظ: {e}")


def require_dataset():
    if df is None or df.empty:
        st.warning("لا توجد بيانات حالياً. يرجى رفع ملف CSV أولاً من تبويب رفع البيانات.")
        st.stop()


def kpi_cards(data: pd.DataFrame):
    cols = len(data.columns)
    rows = len(data)
    missing = int(data.isna().sum().sum())
    dup = int(data.duplicated().sum())
    mem_mb = float(data.memory_usage(deep=True).sum() / (1024**2))

    st.markdown(
        f"""
        <div class="grid">
          <div class="card" style="grid-column: span 3;">
            <div class="card-title">عدد الصفوف</div>
            <div class="card-value">{rows:,}</div>
          </div>
          <div class="card" style="grid-column: span 3;">
            <div class="card-title">عدد الأعمدة</div>
            <div class="card-value">{cols:,}</div>
          </div>
          <div class="card" style="grid-column: span 3;">
            <div class="card-title">القيم المفقودة</div>
            <div class="card-value">{missing:,}</div>
          </div>
          <div class="card" style="grid-column: span 3;">
            <div class="card-title">الصفوف المكررة</div>
            <div class="card-value">{dup:,}</div>
          </div>
        </div>
        <div class="hr"></div>
        <div class="badge"><span class="dot"></span><span>حجم الذاكرة التقريبي: {mem_mb:.2f} MB</span></div>
        """,
        unsafe_allow_html=True
    )


# =========================
# رأس الصفحة
# =========================
st.markdown(
    """
    <div class="app-shell">
      <div class="topbar">
        <div class="brand">
          <div class="brand-title">منصة النمذجة الآلية للبيانات</div>
          <div class="brand-sub">رفع البيانات، التحليل الاستكشافي، بناء النماذج، وإدارة الملفات</div>
        </div>
        <div class="badge"><span class="dot"></span><span>بيئة عربية واتجاه من اليمين إلى اليسار</span></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# تبويبات التطبيق (واجهة عربية)
# =========================
tabs = st.tabs(["رفع البيانات", "تحليل البيانات", "النمذجة", "تحميل النموذج", "المساعد الذكي"])


# =========================
# تبويب: رفع البيانات
# =========================
with tabs[0]:
    st.subheader("رفع ملف بيانات بصيغة CSV")
    file = st.file_uploader("اختر ملف CSV", type=["csv"])

    c1, c2 = st.columns([2, 1])
    with c1:
        if file:
            try:
                df = pd.read_csv(file, index_col=None)
                df.to_csv("dataset.csv", index=None)
                st.success("تم رفع الملف وحفظه بنجاح.")
                kpi_cards(df)
                st.write("معاينة البيانات")
                st.dataframe(df.head(20), use_container_width=True)
            except Exception as e:
                st.error(f"حدث خطأ أثناء قراءة الملف: {e}")
        else:
            if df is not None and not df.empty:
                st.info("تم العثور على بيانات محفوظة مسبقاً.")
                kpi_cards(df)
                st.dataframe(df.head(20), use_container_width=True)
            else:
                st.info("لا يوجد ملف مرفوع بعد.")
    with c2:
        st.markdown(
            """
            <div class="card">
              <div class="card-title">إرشادات</div>
              <div style="color:var(--muted);font-size:13px;line-height:1.7">
                يفضّل أن يحتوي الملف على صفّ عناوين واضح.<br>
                تأكد من أن عمود الهدف موجود عند استخدام تبويب النمذجة.<br>
                يمكن رفع الملف مرة أخرى لاستبدال البيانات الحالية.
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================
# تبويب: تحليل البيانات
# =========================
with tabs[1]:
    require_dataset()
    st.subheader("التحليل الاستكشافي للبيانات")
    kpi_cards(df)

    st.markdown(
        """
        <div class="card">
          <div class="card-title">إعدادات التقرير</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        explorative = st.toggle("تقرير تفصيلي", value=True)
    with c2:
        minimal = st.toggle("تقرير خفيف", value=False)
    with c3:
        sample = st.number_input("عدد الصفوف للتقرير (اختياري)", min_value=0, max_value=20000, value=0, step=100)

    data_for_profile = df.copy()
    if sample and sample > 0 and len(df) > sample:
        data_for_profile = df.sample(sample, random_state=42)

    try:
        profile = ydata_profiling.ProfileReport(
            data_for_profile,
            explorative=explorative,
            minimal=minimal
        )
        st_profile_report(profile)
    except Exception as e:
        st.error("تعذر إنشاء تقرير التحليل.")
        st.code(str(e))


# =========================
# تبويب: النمذجة
# =========================
with tabs[2]:
    require_dataset()
    st.subheader("بناء نموذج تعلم آلة")
    kpi_cards(df)

    st.markdown(
        """
        <div class="card">
          <div class="card-title">إعدادات التدريب</div>
          <div style="color:var(--muted);font-size:13px;line-height:1.7">
            اختر عمود الهدف ونوع المشكلة ثم ابدأ التدريب. يمكن تشغيل كل النماذج لاختيار الأفضل أو اختيار نموذج محدد.
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    cA, cB, cC = st.columns([1.2, 1, 1])
    with cA:
        chosen_target = st.selectbox("عمود الهدف", df.columns)
    with cB:
        algorithm_type = st.radio("نوع المشكلة", ["تصنيف", "انحدار"], horizontal=True)
    with cC:
        run_mode = st.radio("طريقة التشغيل", ["كل النماذج", "نموذج محدد"], horizontal=True)

    # إعداد خيارات النماذج
    if algorithm_type == "تصنيف":
        model_options = [
            ("Random Forest", "rf"),
            ("KNN", "knn"),
            ("Naive Bayes", "nb"),
            ("SVM", "svm"),
            ("XGBoost", "xgboost"),
            ("Decision Tree", "dt"),
        ]
        setup = clf_setup
        compare_models = clf_compare_models
        create_model = clf_create_model
        finalize_model = clf_finalize_model
        save_model = clf_save_model
        predict_model = clf_predict_model
        get_config = clf_get_config
        pull = clf_pull
    else:
        model_options = [
            ("Linear Regression", "lr"),
            ("Ridge", "ridge"),
            ("Lasso", "lasso"),
            ("Random Forest", "rf"),
            ("Gradient Boosting", "gbr"),
            ("Elastic Net", "en"),
        ]
        setup = reg_setup
        compare_models = reg_compare_models
        create_model = reg_create_model
        finalize_model = reg_finalize_model
        save_model = reg_save_model
        predict_model = reg_predict_model
        get_config = reg_get_config
        pull = reg_pull

    chosen_code = None
    if run_mode == "نموذج محدد":
        chosen_name = st.selectbox("اختر النموذج", [m[0] for m in model_options])
        chosen_code = dict(model_options)[chosen_name]

    cX, cY, cZ = st.columns([1, 1, 1])
    with cX:
        normalize = st.toggle("تطبيع البيانات", value=True)
    with cY:
        fold = st.number_input("عدد الطيات (Cross Validation)", min_value=2, max_value=15, value=10, step=1)
    with cZ:
        seed = st.number_input("رقم ثابت (Seed)", min_value=0, max_value=999999, value=123, step=1)

    run_btn = st.button("بدء التدريب")

    if run_btn:
        try:
            setup(
                data=df,
                target=chosen_target,
                normalize=normalize,
                fold=int(fold),
                verbose=False,
                html=False,
                session_id=int(seed)
            )

            if run_mode == "كل النماذج":
                st.info("جاري مقارنة النماذج واختيار الأفضل...")
                best_model = compare_models()
                model_to_use = finalize_model(best_model)
                save_model(model_to_use, "best_model")
                st.success("تم حفظ أفضل نموذج باسم best_model.pkl")

                st.subheader("أفضل نموذج")
                st.write(model_to_use)

                st.subheader("نتائج المقارنة")
                st.dataframe(pull(), use_container_width=True)

            else:
                st.info("جاري تدريب النموذج المحدد...")
                model_to_use = create_model(chosen_code)
                model_to_use = finalize_model(model_to_use)
                save_model(model_to_use, "best_model")
                st.success("تم حفظ النموذج باسم best_model.pkl")

            # بيانات التدريب/الاختبار
            X_train = get_config("X_train")
            y_train = get_config("y_train")
            X_test = get_config("X_test")
            y_test = get_config("y_test")

            train_pred = predict_model(model_to_use, data=X_train)
            test_pred = predict_model(model_to_use, data=X_test)

            pred_col = "Label" if "Label" in test_pred.columns else "prediction_label"
            if pred_col not in test_pred.columns:
                st.error("تعذر العثور على عمود التنبؤ في نتائج PyCaret.")
                st.stop()

            st.divider()
            st.subheader("تقييم الأداء")

            if algorithm_type == "تصنيف":
                train_metrics = pd.DataFrame({
                    "الدقة": [accuracy_score(y_train, train_pred[pred_col])],
                    "F1": [f1_score(y_train, train_pred[pred_col], average="weighted")]
                })
                test_metrics = pd.DataFrame({
                    "الدقة": [accuracy_score(y_test, test_pred[pred_col])],
                    "F1": [f1_score(y_test, test_pred[pred_col], average="weighted")]
                })

                c1, c2 = st.columns(2)
                with c1:
                    st.write("أداء التدريب")
                    st.dataframe(train_metrics, use_container_width=True)
                with c2:
                    st.write("أداء الاختبار")
                    st.dataframe(test_metrics, use_container_width=True)

                cm = confusion_matrix(y_test, test_pred[pred_col])
                st.write("مصفوفة الالتباس")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", ax=ax)
                ax.set_xlabel("المتوقع")
                ax.set_ylabel("الحقيقي")
                st.pyplot(fig)

            else:
                train_metrics = pd.DataFrame({
                    "MSE": [mean_squared_error(y_train, train_pred[pred_col])],
                    "MAE": [mean_absolute_error(y_train, train_pred[pred_col])],
                    "R2": [r2_score(y_train, train_pred[pred_col])]
                })
                test_metrics = pd.DataFrame({
                    "MSE": [mean_squared_error(y_test, test_pred[pred_col])],
                    "MAE": [mean_absolute_error(y_test, test_pred[pred_col])],
                    "R2": [r2_score(y_test, test_pred[pred_col])]
                })

                c1, c2 = st.columns(2)
                with c1:
                    st.write("أداء التدريب")
                    st.dataframe(train_metrics, use_container_width=True)
                with c2:
                    st.write("أداء الاختبار")
                    st.dataframe(test_metrics, use_container_width=True)

        except Exception as e:
            st.error("حدث خطأ أثناء النمذجة.")
            st.code(str(e))


# =========================
# تبويب: تحميل النموذج
# =========================
with tabs[3]:
    st.subheader("تحميل النموذج المدرب")
    st.markdown(
        """
        <div class="card">
          <div class="card-title">ملاحظة</div>
          <div style="color:var(--muted);font-size:13px;line-height:1.7">
            يتم حفظ النموذج باسم best_model.pkl بعد التدريب. يمكن تحميله واستخدامه في تطبيقات أخرى.
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if os.path.exists("best_model.pkl"):
        with open("best_model.pkl", "rb") as f:
            st.download_button("تحميل ملف النموذج", f, file_name="best_model.pkl", use_container_width=True)
    else:
        st.warning("لا يوجد نموذج محفوظ حالياً. قم بتدريب نموذج من تبويب النمذجة أولاً.")


# =========================
# تبويب: المساعد الذكي
# =========================
with tabs[4]:
    st.subheader("المساعد الذكي")
    st.markdown(
        """
        <div class="card">
          <div class="card-title">تعليمات</div>
          <div style="color:var(--muted);font-size:13px;line-height:1.7">
            أدخل سؤالك حول استخدام التطبيق، تجهيز البيانات، اختيار نوع النموذج، أو تفسير النتائج.
            يجب إضافة المفتاح في Secrets باسم OPENROUTER_API_KEY.
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    api_key = None
    if "OPENROUTER_API_KEY" in st.secrets:
        api_key = st.secrets["OPENROUTER_API_KEY"]
    else:
        api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        st.error("لم يتم العثور على المفتاح OPENROUTER_API_KEY. يرجى إضافته في Secrets أو Environment.")
        st.stop()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    SYSTEM_PROMPT = """
أنت مساعد ذكي داخل منصة نمذجة آلية للبيانات مبنية باستخدام Streamlit وPyCaret وydata_profiling.
وظيفتك:
- شرح أقسام المنصة وكيفية استخدامها.
- إرشاد المستخدم في تجهيز البيانات (القيم المفقودة، الترميز، التوازن، التطبيع).
- اقتراح نوع النموذج المناسب (تصنيف أو انحدار).
- تفسير مؤشرات الأداء (Accuracy, F1, MSE, MAE, R2) ومصفوفة الالتباس.
أجب دائمًا بجزئين:
1) شرح بالعربية واضح وبسيط.
2) شرح بالإنجليزية مهني ومنظم.
إذا كان السؤال خارج نطاق المنصة، قم بتوجيه المستخدم بلطف لنطاق المنصة.
"""

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in st.session_state["messages"]:
        if msg["role"] != "system":
            message(msg["content"], is_user=(msg["role"] == "user"))

    user_input = st.text_input("اكتب سؤالك")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})

        with st.spinner("جاري المعالجة"):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=st.session_state["messages"]
                )
                reply = response.choices[0].message.content
            except Exception as e:
                reply = f"حدث خطأ أثناء الاتصال بالخدمة: {e}"

        st.session_state["messages"].append({"role": "assistant", "content": reply})
        message(reply)
