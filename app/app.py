# Feedback (after submit)
if st.session_state.await_next and st.session_state.last_feedback:
    fb = st.session_state.last_feedback

    if learning_mode:
        # Learning mode: show correctness + answer + explanation
        if fb["is_correct"] == 1:
            st.success("Correct ✅")
        else:
            st.error("Incorrect ❌")

        st.info(f"✅ Correct answer: {fb['correct_letter']}: {fb['correct_text']}")

        if fb.get("explanation"):
            st.write("**Explanation:**")
            st.caption(fb["explanation"])

    else:
        # Exam mode: DO NOT reveal correctness during the session
        st.info("✅ Answer saved (Exam mode). Feedback will be available after the session or in Learning mode.")

    if st.button("Next question ▶️"):
        # reset per-question state
        st.session_state.current_q = None
        st.session_state.q_start_time = None
        st.session_state.attempts_now = 1
        st.session_state.hints_now = 0
        st.session_state.eliminated_options = set()

        # clear feedback state
        st.session_state.last_feedback = None
        st.session_state.await_next = False

        st.rerun()
