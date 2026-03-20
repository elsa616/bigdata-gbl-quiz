import time
import os
import random
import re
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# ----------------------------
# Paths (relative to this file)
# ----------------------------
APP_DIR = os.path.dirname(__file__)
QUESTION_BANK_FILE = os.path.join(APP_DIR, "question_bank.csv")
MODEL_FILE = os.path.join(APP_DIR, "lr_quiz_model.pkl")

TOPICS = ["History", "Geography", "Science", "Sports", "ArtsCulture", "Technology"]
MIN_DIFF, MAX_DIFF = 1, 3

DEFAULT_SESSION_LEN = 15
LOW_T = 0.35
HIGH_T = 0.70


# ----------------------------
# Helpers: clean "base" question text (to prevent repeats)
# ----------------------------
def normalize_base_text(text: str) -> str:
    """
    Turn near-duplicate question texts into a stable base form.
    Removes bracket prefixes, and repeated whitespace.
    """
    t = str(text)
    t = re.sub(r"^\[[^\]]+\]\s*", "", t)  # remove "[Geo-1]" style
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower()


# ----------------------------
# Recommendation logic
# ----------------------------
def recommend_next_difficulty(current_difficulty, p_correct):
    if p_correct < LOW_T:
        return max(MIN_DIFF, current_difficulty - 1), "Likely to struggle → easier + support"
    elif p_correct > HIGH_T:
        return min(MAX_DIFF, current_difficulty + 1), "Likely to succeed → harder challenge"
    else:
        return current_difficulty, "Balanced zone → keep difficulty"


def recommend_support(p_correct, hints_used, attempts_count):
    if p_correct < LOW_T:
        return "Offer hint early + suggest 2-minute revision card", "Reduce frustration / prevent failure loop"
    elif p_correct > HIGH_T:
        return "Limit hints + suggest harder question bonus", "Maintain challenge and engagement"
    else:
        if hints_used >= 1 or attempts_count >= 3:
            return "Give gentle hint + brief feedback", "Support without lowering challenge"
        return "Standard feedback", "Keep flow"


def compute_weak_topic(history_df):
    if len(history_df) == 0:
        return None
    tp = history_df.groupby("topic")["correct"].mean().sort_values()
    return tp.index[0]


def recommend_next_topic(current_topic, weak_topic, p_correct):
    if weak_topic is None:
        return current_topic, "No history yet"
    if p_correct < LOW_T:
        return weak_topic, "Focus weak area (support learning gaps)"
    elif p_correct > HIGH_T:
        return current_topic, "Maintain momentum (variety/challenge)"
    else:
        return current_topic, "Keep topic stable (balanced zone)"


def choose_topic_with_variety(rec_topic, history_df, cooldown=2):
    """
    If the same topic appears repeatedly, switch to another topic.
    This makes the game feel more realistic and less repetitive.
    """
    if len(history_df) < cooldown:
        return rec_topic

    recent = list(history_df["topic"].tail(cooldown))
    if all(t == rec_topic for t in recent):
        other = [t for t in TOPICS if t != rec_topic]
        return random.choice(other)

    return rec_topic


# ----------------------------
# Feature building for the model
# ----------------------------
def build_features_from_history(history_df, next_topic, next_difficulty, hints_used_now, attempts_now, time_now):
    if len(history_df) == 0:
        recent_success_5 = 0.5
        recent_hints_5 = 0.0
        recent_time_5 = 50.0
        avg_difficulty_so_far = 2.0
        questions_completed = 0
    else:
        last = history_df.tail(5)
        recent_success_5 = float(last["correct"].mean())
        recent_hints_5 = float(last["hints_used"].mean())
        recent_time_5 = float(last["time_spent_seconds"].mean())
        avg_difficulty_so_far = float(history_df["difficulty"].mean())
        questions_completed = int(len(history_df))

    time_per_attempt = float(time_now) / max(1, int(attempts_now))

    return {
        "difficulty": int(next_difficulty),
        "time_spent_seconds": int(time_now),
        "attempts_count": int(attempts_now),
        "hints_used": int(hints_used_now),
        "time_per_attempt": float(time_per_attempt),
        "recent_success_5": float(recent_success_5),
        "recent_hints_5": float(recent_hints_5),
        "recent_time_5": float(recent_time_5),
        "avg_difficulty_so_far": float(avg_difficulty_so_far),
        "questions_completed": int(questions_completed),
        "topic": str(next_topic),
    }


# ----------------------------
# Question picker (prevents repeats by base text)
# ----------------------------
def pick_question(bank_df, used_ids, used_base_texts, topic, difficulty):
    def filter_pool(df):
        df = df[~df["question_id"].isin(used_ids)].copy()
        df["base_text"] = df["question_text"].apply(normalize_base_text)
        df = df[~df["base_text"].isin(used_base_texts)]
        return df.drop(columns=["base_text"])

    # 1) strict topic+difficulty
    pool = bank_df[(bank_df["topic"] == topic) & (bank_df["difficulty"] == difficulty)]
    pool = filter_pool(pool)
    if len(pool) > 0:
        return pool.sample(1).iloc[0]

    # 2) same topic any difficulty
    pool = bank_df[bank_df["topic"] == topic]
    pool = filter_pool(pool)
    if len(pool) > 0:
        return pool.sample(1).iloc[0]

    # 3) any unused
    pool = filter_pool(bank_df)
    if len(pool) > 0:
        return pool.sample(1).iloc[0]

    return None


# ----------------------------
# Safe model loader (FAST fallback)
# ----------------------------
@st.cache_resource
def load_or_train_model(model_path: str):
    try:
        if os.path.exists(model_path):
            return joblib.load(model_path)
    except Exception:
        pass

    # Fast fallback model (works on Streamlit Cloud)
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(42)

    feature_cols = [
        "difficulty", "time_spent_seconds", "attempts_count", "hints_used", "time_per_attempt",
        "recent_success_5", "recent_hints_5", "recent_time_5", "avg_difficulty_so_far",
        "questions_completed", "topic"
    ]

    n = 800
    df_sim = pd.DataFrame({
        "difficulty": rng.integers(1, 4, size=n),
        "time_spent_seconds": rng.integers(10, 180, size=n),
        "attempts_count": rng.integers(1, 6, size=n),
        "hints_used": rng.integers(0, 3, size=n),
        "recent_success_5": rng.uniform(0, 1, size=n),
        "recent_hints_5": rng.uniform(0, 2, size=n),
        "recent_time_5": rng.uniform(20, 120, size=n),
        "avg_difficulty_so_far": rng.uniform(1, 3, size=n),
        "questions_completed": rng.integers(0, 50, size=n),
        "topic": rng.choice(TOPICS, size=n)
    })
    df_sim["time_per_attempt"] = df_sim["time_spent_seconds"] / df_sim["attempts_count"].clip(lower=1)

    logit = (
        1.2 * (df_sim["recent_success_5"] - 0.5)
        - 0.8 * (df_sim["difficulty"] - 1)
        - 0.12 * (df_sim["attempts_count"] - 1)
        + 0.20 * df_sim["hints_used"]
        - 0.003 * (df_sim["time_spent_seconds"] - 30)
    )
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, np.clip(p, 0.05, 0.95))

    X = df_sim[feature_cols]

    numeric_features = [c for c in feature_cols if c != "topic"]
    categorical_features = ["topic"]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=200, class_weight="balanced"))
    ])

    model.fit(X, y)
    return model


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Adaptive Quiz (GBL Analytics)", layout="wide")
st.title("Adaptive General Knowledge Quiz (University Demo)")

st.sidebar.header("Session settings")
session_len = st.sidebar.slider("Questions this session", 5, 30, DEFAULT_SESSION_LEN, 1)
start_topic = st.sidebar.selectbox("Start topic", ["Any"] + TOPICS, 0)
start_diff = st.sidebar.selectbox("Start difficulty", [1, 2, 3], 1)
st.sidebar.markdown("---")
st.sidebar.caption("This demo adapts difficulty/topic using ML probability (p_correct).")


@st.cache_data
def load_question_bank(path):
    df = pd.read_csv(path)

    required = {"question_id", "topic", "difficulty", "question_text", "A", "B", "C", "D", "correct_option"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"question_bank.csv is missing columns: {missing}")

    df["difficulty"] = df["difficulty"].astype(int)
    df["topic"] = df["topic"].astype(str)

    return df


if not os.path.exists(QUESTION_BANK_FILE):
    st.error(f"Missing question bank: {QUESTION_BANK_FILE}")
    st.stop()

bank = load_question_bank(QUESTION_BANK_FILE)

with st.spinner("Loading ML model (or training fallback model)..."):
    model = load_or_train_model(MODEL_FILE)

# ----------------------------
# Session state
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "used_ids" not in st.session_state:
    st.session_state.used_ids = set()
if "used_base_texts" not in st.session_state:
    st.session_state.used_base_texts = set()
if "current_q" not in st.session_state:
    st.session_state.current_q = None
if "q_start_time" not in st.session_state:
    st.session_state.q_start_time = None
if "attempts_now" not in st.session_state:
    st.session_state.attempts_now = 1
if "hints_now" not in st.session_state:
    st.session_state.hints_now = 0
if "eliminated_options" not in st.session_state:
    st.session_state.eliminated_options = set()

# NEW: feedback state so we can show correct answer after submit
if "last_feedback" not in st.session_state:
    st.session_state.last_feedback = None
if "await_next" not in st.session_state:
    st.session_state.await_next = False


def reset_session():
    st.session_state.history = []
    st.session_state.used_ids = set()
    st.session_state.used_base_texts = set()
    st.session_state.current_q = None
    st.session_state.q_start_time = None
    st.session_state.attempts_now = 1
    st.session_state.hints_now = 0
    st.session_state.eliminated_options = set()

    st.session_state.last_feedback = None
    st.session_state.await_next = False


st.sidebar.button("Start / Reset session", on_click=reset_session)

history_df = pd.DataFrame(st.session_state.history)

# Determine current topic/difficulty
if len(history_df) == 0:
    cur_topic = random.choice(TOPICS) if start_topic == "Any" else start_topic
    cur_diff = int(start_diff)
else:
    cur_topic = history_df.iloc[-1]["topic"]
    cur_diff = int(history_df.iloc[-1]["difficulty"])

weak_topic = compute_weak_topic(history_df)

feat_row = build_features_from_history(
    history_df,
    next_topic=cur_topic,
    next_difficulty=cur_diff,
    hints_used_now=st.session_state.hints_now,
    attempts_now=st.session_state.attempts_now,
    time_now=30
)
X_one = pd.DataFrame([feat_row])
p_correct = float(model.predict_proba(X_one)[:, 1][0])

rec_diff, rec_reason = recommend_next_difficulty(cur_diff, p_correct)
rec_topic, rec_topic_reason = recommend_next_topic(cur_topic, weak_topic, p_correct)

# Force variety so topics don't feel repetitive
topic_for_q = choose_topic_with_variety(rec_topic, history_df, cooldown=2)

support_action, support_reason = recommend_support(
    p_correct, st.session_state.hints_now, st.session_state.attempts_now
)

# Pick question if none active (BUT don't change if waiting for "Next question")
if st.session_state.current_q is None and not st.session_state.await_next:
    q = pick_question(
        bank,
        st.session_state.used_ids,
        st.session_state.used_base_texts,
        topic_for_q,
        rec_diff
    )

    if q is None:
        st.error("No more new questions available in this session. Please reset the session.")
        st.stop()

    st.session_state.current_q = q
    st.session_state.q_start_time = time.time()
    st.session_state.attempts_now = 1
    st.session_state.hints_now = 0
    st.session_state.eliminated_options = set()

    st.session_state.used_ids.add(q["question_id"])
    st.session_state.used_base_texts.add(normalize_base_text(q["question_text"]))

q = st.session_state.current_q

# End condition
if len(history_df) >= session_len:
    st.success(f"Session complete! You answered {session_len} questions.")
    hist = pd.DataFrame(st.session_state.history)

    st.subheader("Session summary")
    st.write("Accuracy:", round(hist["correct"].mean(), 3))
    st.write("Avg time:", round(hist["time_spent_seconds"].mean(), 1), "seconds")
    st.write("Avg attempts:", round(hist["attempts_count"].mean(), 2))
    st.write("Avg hints:", round(hist["hints_used"].mean(), 2))

    st.dataframe(hist.tail(15))
    st.stop()

left, right = st.columns([2, 1])

with right:
    st.subheader("Adaptive Engine")
    st.metric("p_correct", f"{p_correct:.3f}")

    st.write("**Next difficulty:**", rec_diff)
    st.write(rec_reason)

    st.write("**Next topic:**", topic_for_q)
    st.write(rec_topic_reason)

    st.write("**Support action:**", support_action)
    st.caption(support_reason)

with left:
    st.subheader(f"Question {len(history_df)+1} of {session_len}")
    st.caption(f"Recommended: topic={topic_for_q}, difficulty={rec_diff}")

    st.markdown(f"**{q['question_text']}**")
    st.write(f"Topic: **{q['topic']}** | Difficulty: **{int(q['difficulty'])}**")

    options = ["A", "B", "C", "D"]
    visible = [o for o in options if o not in st.session_state.eliminated_options]
    option_text = {o: q[o] for o in visible}

    # disable changing answers after submit
    choice = st.radio(
        "Choose an answer:",
        list(option_text.keys()),
        format_func=lambda k: f"{k}: {option_text[k]}",
        disabled=st.session_state.await_next
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("Use hint (50/50)", disabled=st.session_state.await_next):
            wrong = [o for o in options if o != q["correct_option"]]
            random.shuffle(wrong)
            st.session_state.eliminated_options.update(wrong[:2])
            st.session_state.hints_now += 1
            st.info("Hint used: removed two wrong options.")

    with c2:
        if st.button("Submit answer", disabled=st.session_state.await_next):
            time_spent = int(round(time.time() - st.session_state.q_start_time))
            is_correct = int(choice == q["correct_option"])

            st.session_state.history.append({
                "question_id": q["question_id"],
                "topic": q["topic"],
                "difficulty": int(q["difficulty"]),
                "time_spent_seconds": time_spent,
                "attempts_count": int(st.session_state.attempts_now),
                "hints_used": int(st.session_state.hints_now),
                "correct": is_correct
            })

            correct_letter = q["correct_option"]
            correct_text = q[correct_letter]

            st.session_state.last_feedback = {
                "is_correct": is_correct,
                "correct_letter": correct_letter,
                "correct_text": correct_text
            }
            st.session_state.await_next = True

    with c3:
        if st.button("Retry (counts as another attempt)", disabled=st.session_state.await_next):
            st.session_state.attempts_now += 1
            st.warning(f"Attempts for this question: {st.session_state.attempts_now}")

    # Feedback + correct answer
    if st.session_state.await_next and st.session_state.last_feedback:
        fb = st.session_state.last_feedback
        if fb["is_correct"] == 1:
            st.success("Correct ✅")
        else:
            st.error("Incorrect ❌")

        st.info(f"✅ Correct answer: {fb['correct_letter']}: {fb['correct_text']}")

        if st.button("Next question ▶️"):
            # move to next question
            st.session_state.current_q = None
            st.session_state.q_start_time = None
            st.session_state.attempts_now = 1
            st.session_state.hints_now = 0
            st.session_state.eliminated_options = set()

            st.session_state.last_feedback = None
            st.session_state.await_next = False

            st.rerun()

    st.markdown("---")
    if len(history_df) > 0:
        st.subheader("Recent history (last 8)")
        st.dataframe(history_df.tail(8))
