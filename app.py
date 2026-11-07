import re
import html
import streamlit as st

from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="CampusMate | Smart College Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# COLLEGE FAQ DATASET
# Replace sample information with actual college details
# =====================================================

FAQS = [
    {
        "category": "Admissions",
        "question": "What are the admission requirements?",
        "answer": "Students must satisfy the eligibility criteria for their selected course. Required documents may include academic marksheets, transfer certificate, identity proof and passport-size photographs. Contact the admission office for the latest requirements."
    },
    {
        "category": "Admissions",
        "question": "How can I apply for admission?",
        "answer": "Students can apply through the official college admission portal or visit the admission office. Complete the application form and submit all required documents."
    },
    {
        "category": "Courses",
        "question": "Which courses are offered by the college?",
        "answer": "The college offers undergraduate and postgraduate programmes through its academic departments. Please refer to the official college prospectus for the updated course list."
    },
    {
        "category": "Courses",
        "question": "How can I get course details?",
        "answer": "Course duration, syllabus, eligibility and subject details are available through the respective academic departments and college prospectus."
    },
    {
        "category": "Examinations",
        "question": "When will the examinations be conducted?",
        "answer": "Examination schedules are published by the examination cell. Students should regularly check official notifications and the student portal."
    },
    {
        "category": "Examinations",
        "question": "How can I check examination results?",
        "answer": "Students can check examination results through the official college examination portal using their student credentials."
    },
    {
        "category": "Library",
        "question": "What are the library timings?",
        "answer": "Library timings depend on the college working schedule. Please contact the librarian or check the library notice board for current opening and closing times."
    },
    {
        "category": "Library",
        "question": "How can I borrow books?",
        "answer": "Students can borrow books using their valid college ID card. Borrowing limits and return dates are determined by library rules."
    },
    {
        "category": "Hostel",
        "question": "Is hostel accommodation available?",
        "answer": "Hostel accommodation is subject to availability. Contact the hostel administration for applications, room availability and fee details."
    },
    {
        "category": "Hostel",
        "question": "What facilities are available in the hostel?",
        "answer": "Hostel facilities may include furnished rooms, dining facilities, drinking water, security and study spaces. Confirm available facilities with the hostel office."
    },
    {
        "category": "Transport",
        "question": "Does the college provide bus facilities?",
        "answer": "College transport availability, routes and pickup points are managed by the transport department. Contact the transport office for updated details."
    },
    {
        "category": "Transport",
        "question": "How can I get bus route details?",
        "answer": "Students can obtain bus route numbers, pickup locations and timings from the college transport office."
    },
    {
        "category": "Fees",
        "question": "How can I pay college fees?",
        "answer": "Students can pay fees using the official payment portal or the payment methods provided by the accounts department. Always keep your payment receipt."
    },
    {
        "category": "Fees",
        "question": "Who should I contact for fee problems?",
        "answer": "For fee-related issues, contact the college accounts department with your student ID and payment receipt."
    },
    {
        "category": "Scholarships",
        "question": "Are scholarships available for students?",
        "answer": "Eligible students may apply for government, merit-based and other scholarships. Contact the scholarship office for eligibility requirements and application deadlines."
    },
    {
        "category": "Scholarships",
        "question": "How can I apply for scholarships?",
        "answer": "Students should check current scholarship notifications, eligibility conditions and deadlines, then submit the required documents through the official application process."
    },
    {
        "category": "Placements",
        "question": "What placement opportunities are available?",
        "answer": "The placement cell coordinates recruitment activities, career guidance and company interactions. Contact the placement department for current opportunities."
    },
    {
        "category": "Placements",
        "question": "Does the college provide placement training?",
        "answer": "Placement training may include aptitude preparation, communication skills, resume building, technical training and interview preparation."
    },
    {
        "category": "Contact",
        "question": "How can I contact the college?",
        "answer": "Please visit the official college website or contact the main reception for verified phone numbers, email addresses and office timings."
    }
]


# =====================================================
# NLP CHATBOT ENGINE
# =====================================================

STOPWORDS = {
    "the", "is", "are", "am", "a", "an", "to", "for",
    "of", "in", "on", "at", "and", "or", "my", "me",
    "i", "we", "you", "your", "our", "can", "could",
    "would", "what", "when", "where", "how", "which",
    "please", "tell", "about", "do", "does", "did"
}


def preprocess_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    tokenizer = TreebankWordTokenizer()

    tokens = tokenizer.tokenize(text)

    tokens = [
        word for word in tokens
        if word not in STOPWORDS
        and len(word) > 1
        and not word.isdigit()
    ]

    return tokens


@st.cache_resource
def create_chatbot():

    questions = [
        faq["question"] for faq in FAQS
    ]

    vectorizer = TfidfVectorizer(
        tokenizer=preprocess_text,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2)
    )

    faq_vectors = vectorizer.fit_transform(questions)

    return vectorizer, faq_vectors


vectorizer, faq_vectors = create_chatbot()


def get_answer(user_question):

    if not user_question.strip():

        return {
            "answer": "Please enter a question so I can help you.",
            "score": 0,
            "matched": False
        }

    question_vector = vectorizer.transform(
        [user_question]
    )

    if question_vector.nnz == 0:

        return fallback_response()

    similarities = cosine_similarity(
        question_vector,
        faq_vectors
    ).flatten()

    best_index = similarities.argmax()

    best_score = float(similarities[best_index])

    # Configurable similarity threshold
    SIMILARITY_THRESHOLD = 0.20

    if best_score < SIMILARITY_THRESHOLD:

        return fallback_response(best_score)

    return {
        "answer": FAQS[best_index]["answer"],
        "score": best_score,
        "matched": True,
        "category": FAQS[best_index]["category"],
        "matched_question": FAQS[best_index]["question"]
    }


def fallback_response(score=0):

    return {
        "answer": (
            "Sorry! I couldn't find a suitable answer "
            "in my college FAQ database. Please ask me "
            "about admissions, courses, examinations, "
            "library, hostel, transport, fees, scholarships "
            "or placements."
        ),
        "score": score,
        "matched": False
    }


# =====================================================
# PROFESSIONAL UI DESIGN
# =====================================================

st.markdown("""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

:root {
    --navy: #102A56;
    --blue: #2864C8;
    --light: #EEF4FF;
    --white: #FFFFFF;
    --dark: #172B4D;
    --muted: #596D86;
}

html, body, [data-testid="stAppViewContainer"] {
    background: #F4F7FC !important;
    color: #172B4D !important;
    font-family: 'Inter', Arial, sans-serif !important;
}

[data-testid="stMain"] {
    background: #F4F7FC !important;
}

.main .block-container {
    max-width: 1250px;
    padding: 2rem 2.5rem 3rem;
}

h1, h2, h3, h4, p, label {
    color: #172B4D !important;
}

h1 {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
}

h2, h3 {
    font-weight: 700 !important;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #102A56,
        #193F78
    ) !important;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] h2 {
    font-size: 1.7rem !important;
    font-weight: 800 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 10px;
    border-radius: 10px;
}

/* HERO */

.hero {
    background: linear-gradient(
        120deg,
        #12366F,
        #3477E8
    );
    padding: 38px;
    border-radius: 22px;
    margin: 20px 0 30px;
    box-shadow: 0 10px 30px rgba(16,42,86,0.14);
}

.hero h2 {
    color: white !important;
    font-size: 2rem !important;
}

.hero p {
    color: #FFFFFF !important;
    font-size: 1rem;
}

/* CARDS */

.service-card {
    background: #FFFFFF;
    border: 1px solid #DFE8F5;
    border-radius: 18px;
    padding: 25px;
    min-height: 165px;
    box-shadow: 0 5px 20px rgba(16,42,86,0.06);
    transition: 0.25s;
}

.service-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 25px rgba(16,42,86,0.12);
}

.service-card h3 {
    color: #173B70 !important;
    font-size: 1.1rem;
}

.service-card p {
    color: #526781 !important;
    line-height: 1.6;
}

/* CHAT */

.chat-user {
    background: #DCEAFF;
    color: #12366F;
    border: 1px solid #BDD5FA;
    border-radius: 18px 18px 4px 18px;
    padding: 16px;
    margin: 12px 0 12px auto;
    max-width: 80%;
    line-height: 1.7;
    overflow-wrap: anywhere;
}

.chat-assistant {
    background: #FFFFFF;
    color: #172B4D;
    border: 1px solid #DFE8F5;
    border-radius: 18px 18px 18px 4px;
    padding: 18px;
    margin: 12px auto 15px 0;
    max-width: 88%;
    line-height: 1.7;
    box-shadow: 0 4px 15px rgba(16,42,86,0.06);
    overflow-wrap: anywhere;
}

.chat-user strong {
    color: #12366F !important;
}

.chat-assistant strong {
    color: #2864C8 !important;
}

/* CHAT INPUT */

[data-testid="stChatInput"] {
    background: white !important;
    border: 2px solid #C8D9F0 !important;
    border-radius: 15px !important;
}

[data-testid="stChatInput"] textarea {
    color: #172B4D !important;
    background: white !important;
    caret-color: #2864C8 !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #71829A !important;
    opacity: 1 !important;
}

/* INPUTS */

div[data-baseweb="input"] {
    background: white !important;
}

div[data-baseweb="input"] input {
    color: #172B4D !important;
}

/* BUTTONS */

.stButton > button {
    background: white !important;
    color: #245FC7 !important;
    border: 1px solid #C8D9F0 !important;
    border-radius: 11px !important;
    font-weight: 650 !important;
    min-height: 42px;
}

.stButton > button:hover {
    background: #EAF2FF !important;
    border-color: #2864C8 !important;
}

/* EXPANDER */

[data-testid="stExpander"] {
    background: white !important;
    border: 1px solid #DFE8F5 !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span {
    color: #173B70 !important;
}

/* MOBILE */

@media(max-width: 768px) {

    .main .block-container {
        padding: 1rem !important;
    }

    h1 {
        font-size: 1.7rem !important;
    }

    .hero {
        padding: 22px;
    }

    .chat-user,
    .chat-assistant {
        max-width: 100%;
    }

}

</style>
""", unsafe_allow_html=True)


# =====================================================
# SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:

    st.markdown("## 🎓 CampusMate")

    st.caption("SMART COLLEGE ASSISTANT")

    st.markdown("---")

    st.markdown("### Explore Topics")

    pages = [
        "Dashboard",
        "Admissions",
        "Courses",
        "Examinations",
        "Library",
        "Transport",
        "Hostel",
        "Fees",
        "Scholarships",
        "Placements",
        "Contact"
    ]

    selected_page = st.radio(
        "Navigation",
        pages,
        label_visibility="collapsed"
    )

    st.session_state.page = selected_page

    st.markdown("---")

    st.caption("Your College Assistant")


# =====================================================
# DASHBOARD HEADER
# =====================================================

st.title("Welcome to CampusMate 👋")

st.markdown(
    "Your Smart College FAQ Assistant"
)


# =====================================================
# DASHBOARD
# =====================================================

if selected_page == "Dashboard":

    st.markdown("""
    <div class="hero">
    <h2>How can I help you today?</h2>
    <p>
        Ask questions about admissions, courses,
        examinations, library, transport, hostel,
        fees, scholarships and placements.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

        st.markdown("""
        <div class="service-card">

            <h3>📚 Academic Support</h3>

            <p>
            Find information about courses,
            admissions and examinations.
            </p>

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="service-card">

            <h3>🏫 Campus Services</h3>

            <p>
            Explore library, hostel and
            transport information.
            </p>

        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown("""
        <div class="service-card">

            <h3>🎯 Student Services</h3>

            <p>
            Get help with placements,
            fees and scholarships.
            </p>

        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.subheader("Quick Questions")

    questions = [
        "What are the admission requirements?",
        "Which courses are offered?",
        "What are the library timings?",
        "Is hostel accommodation available?",
        "How can I pay college fees?",
        "What placement opportunities are available?"
    ]

    cols = st.columns(2)

    for i, question in enumerate(questions):

        with cols[i % 2]:

            if st.button(
                question,
                key=f"quick_{i}",
                use_container_width=True
            ):

                st.session_state.pending_question = question
                st.rerun()


# =====================================================
# CATEGORY NAVIGATION
# =====================================================

else:

    st.subheader(f"📌 {selected_page}")

    st.write("Explore frequently asked questions.")

    category_faqs = [
        faq for faq in FAQS
        if faq["category"] == selected_page
    ]

    for index, faq in enumerate(category_faqs):

        with st.expander(faq["question"]):

            st.write(faq["answer"])

            if st.button(
                "Ask CampusMate",
                key=f"category_{selected_page}_{index}"
            ):

                st.session_state.pending_question = faq["question"]
                st.rerun()


# =====================================================
# CHATBOT SECTION
# =====================================================

st.markdown("---")

st.subheader("💬 Chat with CampusMate")

st.caption(
    "Ask your questions and get instant answers "
    "from our NLP-based college FAQ system."
)

left, right = st.columns([5, 1])

with right:

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.rerun()


# =====================================================
# DISPLAY CHAT HISTORY
# =====================================================

if not st.session_state.messages:

    st.markdown("""
    <div class="chat-assistant">

        <strong>🎓 CampusMate</strong>

        <p>
        Hello! Welcome to CampusMate.

        I am your smart college FAQ assistant.

        Ask me anything about college services!
        </p>

    </div>
    """, unsafe_allow_html=True)


for message in st.session_state.messages:

    safe_content = html.escape(message["content"])

    if message["role"] == "user":

        st.markdown(
            f"""
            <div class="chat-user">
                <strong>You</strong><br>
                {safe_content}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="chat-assistant">
                <strong>🎓 CampusMate</strong><br>
                {safe_content}
            </div>
            """,
            unsafe_allow_html=True
        )


# =====================================================
# CHAT INPUT AND RESPONSE
# =====================================================

pending_question = st.session_state.pop(
    "pending_question",
    None
)

user_question = st.chat_input(
    "Type your college-related question here..."
)

question = pending_question or user_question

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.spinner("CampusMate is searching for an answer..."):

        result = get_answer(question)

    answer = result["answer"]

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()


# =====================================================
# FAQ SEARCH
# =====================================================

st.markdown("---")

st.subheader("🔎 Search College FAQs")

search_query = st.text_input(
    "Search FAQs",
    placeholder="Search admissions, library, hostel, fees..."
)

if search_query:

    results = []

    for faq in FAQS:

        combined = (
            faq["question"] + " " +
            faq["answer"] + " " +
            faq["category"]
        ).lower()

        if search_query.lower() in combined:
            results.append(faq)

    if results:

        for faq in results:

            with st.expander(faq["question"]):

                st.write(faq["answer"])

    else:

        st.info(
            "No matching FAQs found. Try another keyword."
        )


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown("""
<div style="text-align:center; color:#526781; padding:20px;">
    🎓 CampusMate | NLP-Based College FAQ Assistant
    <br>
    Built with Python and Streamlit
</div>
""", unsafe_allow_html=True)
