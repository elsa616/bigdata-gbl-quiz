import time
import os
import random
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

DEFAULT_SESSION_LEN = 15  # your choice (B)
LOW_T = 0.35
HIGH_T = 0.70


# ----------------------------
# Recommendations
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


def pick_question(bank_df, used_ids, topic, difficulty):
    pool = bank_df[(bank_df["topic"] == topic) & (bank_df["difficulty"] == difficulty)]
    pool = pool[~pool["question_id"].isin(used_ids)]
    if len(pool) > 0:
        return pool.sample(1).iloc[0]

    pool = bank_df[bank_df["topic"] == topic]
    pool = pool[~pool["question_id"].isin(used_ids)]
    if len(pool) > 0:
        return pool.sample(1).iloc[0]

    pool = bank_df[~bank_df["question_id"].isin(used_ids)]
    if len(pool) > 0:
        return pool.sample(1).iloc[0]

    return None


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
    return pd.read_csv(path)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


# Guard checks (helpful error messages)
if not os.path.exists(QUESTION_BANK_FILE):
    st.error(f"Missing question bank: {QUESTION_BANK_FILE}")
    st.stop()

if not os.path.exists(MODEL_FILE):
    st.error(f"Missing model file: {MODEL_FILE}")
    st.stop()

bank = load_question_bank(QUESTION_BANK_FILE)
model = load_model(MODEL_FILE)

# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "used_ids" not in st.session_state:
    st.session_state.used_ids = set()
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


def reset_session():
    st.session_state.history = []
    st.session_state.used_ids = set()
    st.session_state.current_q = None
    st.session_state.q_start_time = None
    st.session_state.attempts_now = 1
    st.session_state.hints_now = 0
    st.session_state.eliminated_options = set()


st.sidebar.button("Start / Reset session", on_click=reset_session)

history_df = pd.DataFrame(st.session_state.history)

# Determine current topic/difficulty
if len(history_df) == 0:
    cur_topic = random.choice(TOPICS) if start_topic == "Any" else start_topic
    cur_diff = int(start_diff)
else:
    cur_topic = history_df.iloc[-1]["topic"]
    cur_diff = int(history_df.iloc[-1]["difficulty"])

# Estimate probability for next step
weak_topic = compute_weak_topic(history_df)

feat_row = build_features_from_history(
    history_df, next_topic=cur_topic, next_difficulty=cur_diff,
    hints_used_now=st.session_state.hints_now,
    attempts_now=st.session_state.attempts_now,
    time_now=30
)
X_one = pd.DataFrame([feat_row])
p_correct = float(model.predict_proba(X_one)[:, 1][0])

rec_diff, rec_reason = recommend_next_difficulty(cur_diff, p_correct)
rec_topic, rec_topic_reason = recommend_next_topic(cur_topic, weak_topic, p_correct)
support_action, support_reason = recommend_support(p_correct, st.session_state.hints_now, st.session_state.attempts_now)

# Pick question if none active
if st.session_state.current_q is None:
    q = pick_question(bank, st.session_state.used_ids, rec_topic, rec_diff)
    st.session_state.current_q = q
    st.session_state.q_start_time = time.time()
    st.session_state.attempts_now = 1
    st.session_state.hints_now = 0
    st.session_state.eliminated_options = set()

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
    st.write("**Next topic:**", rec_topic)
    st.write(rec_topic_reason)
    st.write("**Support action:**", support_action)
    st.caption(support_reason)

with left:
    st.subheader(f"Question {len(history_df)+1} of {session_len}")
    st.caption(f"Recommended: topic={rec_topic}, difficulty={rec_diff}")
    st.markdown(f"**{q['question_text']}**")
    st.write(f"Topic: **{q['topic']}** | Difficulty: **{int(q['difficulty'])}**")

    options = ["A", "B", "C", "D"]
    visible = [o for o in options if o not in st.session_state.eliminated_options]
    option_text = {o: q[o] for o in visible}

    choice = st.radio("Choose an answer:", list(option_text.keys()),
                      format_func=lambda k: f"{k}: {option_text[k]}")

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("Use hint (50/50)"):
            wrong = [o for o in options if o != q["correct_option"]]
            random.shuffle(wrong)
            st.session_state.eliminated_options.update(wrong[:2])
            st.session_state.hints_now += 1
            st.info("Hint used: removed two wrong options.")

    with c2:
        if st.button("Submit answer"):
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
            st.session_state.used_ids.add(q["question_id"])

            if is_correct:
                st.success("Correct ✅")
            else:
                st.error(f"Incorrect ❌ (Correct answer: {q['correct_option']}: {q[q['correct_option']]})")

            st.session_state.current_q = None

    with c3:
        if st.button("Retry (counts as another attempt)"):
            st.session_state.attempts_now += 1
            st.warning(f"Attempts for this question: {st.session_state.attempts_now}")

    st.markdown("---")
    if len(history_df) > 0:
        st.subheader("Recent history (last 8)")
        st.dataframe(history_df.tail(8))
