import streamlit as st
import pandas as pd
import plotly.express as px

from services.claim_analysis import (
    extract_claim,
    analyze_evidence
)

from services.prediction_service import (
    predict_content
)

from services.evidence_service import (
    search_evidence
)

from services.credibility_service import (
    rank_sources
)

from services.final_decision_service import (
    make_final_decision
)

from services.history_service import (
    save_detection_history,
    load_detection_history
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Fake Content Detection",
    page_icon="🔎",
    layout="wide"
)


# ==================================================
# HEADER
# ==================================================

st.title(
    "🔎 Fake Content Detection & Evidence Verification"
)

st.write(
    "AI-based detection of news/social-media content "
    "using machine learning and real-time web evidence."
)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("📌 Project Information")

    st.write(
        "**Feature Extraction:** TF-IDF"
    )

    st.write(
        "**Machine Learning Model:** Linear SVM"
    )

    st.write(
        "**Model Accuracy:** 99.58%"
    )

    st.write(
        "**F1 Score:** 99.62%"
    )

    st.markdown("---")

    st.write(
        "🌐 Real-time evidence is retrieved "
        "using web search."
    )

    st.write(
        "🛡️ Sources are analyzed for credibility."
    )

    st.write(
        "📊 Detection results are saved in history."
    )

    st.write(
        "⚖️ ML prediction and evidence are combined "
        "for a final decision."
    )


# ==================================================
# INPUT SECTION
# ==================================================

content = st.text_area(
    "📰 Enter News / Social Media Content",
    height=250,
    placeholder=(
        "Paste a news article, headline, or "
        "social-media claim here..."
    )
)


# ==================================================
# ANALYZE BUTTON
# ==================================================

if st.button(
    "🔍 Analyze Content",
    type="primary"
):

    # ==============================================
    # INPUT VALIDATION
    # ==============================================

    if not content.strip():

        st.warning(
            "Please enter some content before analyzing."
        )

        st.stop()


    # ==============================================
    # MACHINE LEARNING PREDICTION
    # ==============================================

    with st.spinner(
        "Running machine-learning analysis..."
    ):

        prediction_result = predict_content(
            content
        )


    # ==============================================
    # GET ML RESULTS
    # ==============================================

    prediction = prediction_result.get(
        "prediction",
        "UNKNOWN"
    )

    decision_score = prediction_result.get(
        "decision_score",
        None
    )

    ml_confidence = prediction_result.get(
        "confidence",
        50.0
    )

    prediction_strength = prediction_result.get(
        "prediction_strength",
        "LOW"
    )


    # ==============================================
    # MACHINE LEARNING RESULT DISPLAY
    # ==============================================

    st.markdown("---")

    st.header(
        "🤖 Machine Learning Result"
    )


    if prediction == "FAKE":

        st.error(
            "🔴 LIKELY FAKE"
        )

    elif prediction == "REAL":

        st.success(
            "🟢 LIKELY REAL"
        )

    else:

        st.info(
            "⚪ PREDICTION UNKNOWN"
        )


    st.write(
        f"**Prediction:** {prediction}"
    )


    # ==============================================
    # ML METRICS
    # ==============================================

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "ML Prediction",
            prediction
        )


    with col2:

        st.metric(
            "ML Confidence",
            f"{ml_confidence}%"
        )


    # ==============================================
    # MODEL PREDICTION STRENGTH
    # ==============================================

    if prediction_strength == "HIGH":

        st.success(
            "🟢 Model Prediction Strength: HIGH"
        )

    elif prediction_strength == "MODERATE":

        st.warning(
            "🟡 Model Prediction Strength: MODERATE"
        )

    else:

        st.info(
            "⚪ Model Prediction Strength: LOW"
        )


    # ==============================================
    # TECHNICAL MODEL DETAILS
    # ==============================================

    if decision_score is not None:

        with st.expander(
            "🔧 View Technical Model Details"
        ):

            st.write(
                f"**SVM Decision Score:** "
                f"`{decision_score:.4f}`"
            )

            st.caption(
                "The decision score represents the "
                "distance from the SVM decision boundary. "
                "Higher absolute values generally indicate "
                "a stronger model prediction."
            )


    # ==============================================
    # REAL-TIME EVIDENCE SEARCH
    # ==============================================

    st.markdown("---")

    st.header(
        "🌐 Real-Time Evidence Search"
    )


    with st.spinner(
        "Searching the web for relevant evidence..."
    ):

        try:

            evidence = search_evidence(
                content,
                max_results=5
            )

        except Exception as error:

            evidence = []

            st.error(
                f"Evidence search failed: {error}"
            )


    # ==============================================
    # DISPLAY SEARCH STATUS
    # ==============================================

    if evidence:

        st.success(
            f"✅ Successfully retrieved "
            f"{len(evidence)} relevant sources "
            f"from the web."
        )

    else:

        st.warning(
            "⚠️ No relevant evidence sources were found."
        )


    # ==============================================
    # SOURCE CREDIBILITY RANKING
    # ==============================================

    if evidence:

        evidence = rank_sources(
            evidence
        )


    # ==============================================
    # ADVANCED EVIDENCE VERIFICATION
    # ==============================================

    st.markdown("---")

    st.header(
        "🧠 Advanced Evidence Verification"
    )


    claim = extract_claim(
        content
    )


    analysis = analyze_evidence(
        claim,
        evidence
    )


    status = analysis.get(
        "status",
        "UNCLEAR"
    )

    score = analysis.get(
        "score",
        0
    )

    reason = analysis.get(
        "reason",
        "No analysis available."
    )


    # ==============================================
    # DISPLAY EVIDENCE STATUS
    # ==============================================

    if status == "SUPPORTING":

        st.success(
            "🟢 Evidence appears SUPPORTING"
        )

    elif status == "CONTRADICTING":

        st.error(
            "🔴 Evidence appears CONTRADICTING"
        )

    elif status == "RELEVANT":

        st.warning(
            "🟡 Relevant evidence found, but the "
            "claim cannot be conclusively verified."
        )

    else:

        st.info(
            "⚪ Evidence status: UNCLEAR"
        )


    st.write(
        f"**Evidence Relevance Score:** {score}%"
    )

    st.write(
        f"**Analysis:** {reason}"
    )


    # ==============================================
    # SOURCE CREDIBILITY ANALYSIS
    # ==============================================

    st.markdown("---")

    st.header(
        "🏆 Source Credibility Analysis"
    )


    if evidence:

        credibility_scores = []


        for source in evidence:

            credibility_scores.append(
                source.get(
                    "credibility_score",
                    0
                )
            )


        average_credibility = round(
            sum(credibility_scores)
            / len(credibility_scores),
            2
        )


        if average_credibility >= 80:

            st.success(
                "🟢 High Credibility Sources"
            )

        elif average_credibility >= 50:

            st.warning(
                "🟡 Moderate Credibility Sources"
            )

        else:

            st.error(
                "🔴 Low Credibility Sources"
            )


        st.write(
            f"**Average Source Credibility:** "
            f"{average_credibility}%"
        )


    else:

        average_credibility = 0

        st.info(
            "⚪ No sources available for "
            "credibility analysis."
        )


    # ==============================================
    # SMART FINAL DECISION
    # ==============================================

    st.markdown("---")

    st.header(
        "⚖️ Smart Final Decision"
    )


    final_result = make_final_decision(
        prediction=prediction,
        evidence_status=status,
        evidence=evidence
    )


    verdict = final_result.get(
        "verdict",
        "UNCLEAR"
    )

    confidence = final_result.get(
        "confidence",
        0
    )

    final_reason = final_result.get(
        "reason",
        "No final decision explanation available."
    )

    final_average_credibility = final_result.get(
        "average_credibility",
        average_credibility
    )


    if (
        final_average_credibility == 0
        and average_credibility > 0
    ):

        final_average_credibility = (
            average_credibility
        )


    # ==============================================
    # DISPLAY FINAL VERDICT
    # ==============================================

    if verdict == "LIKELY TRUE":

        st.success(
            f"🟢 {verdict}"
        )

    elif verdict == "LIKELY FALSE":

        st.error(
            f"🔴 {verdict}"
        )

    elif verdict == "CONFLICTING SIGNALS":

        st.warning(
            f"🟡 {verdict}"
        )

    else:

        st.info(
            f"⚪ {verdict}"
        )


    # ==============================================
    # FINAL METRICS
    # ==============================================

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Final Confidence",
            f"{confidence}%"
        )


    with col2:

        st.metric(
            "Average Source Credibility",
            f"{final_average_credibility}%"
        )


    st.write(
        f"**Decision Explanation:** "
        f"{final_reason}"
    )


    # ==============================================
    # SAVE DETECTION HISTORY
    # ==============================================

    try:

        save_detection_history(

            content=content,

            prediction=prediction,

            evidence_status=status,

            evidence_score=score,

            source_credibility=final_average_credibility,

            final_verdict=verdict,

            final_confidence=confidence

        )


    except Exception as error:

        st.warning(
            f"History could not be saved: {error}"
        )


    # ==============================================
    # RETRIEVED EVIDENCE SOURCES
    # ==============================================

    st.markdown("---")

    st.header(
        "📚 Retrieved Evidence Sources"
    )


    if evidence:

        st.success(
            f"Found {len(evidence)} relevant sources."
        )


        for number, source in enumerate(
            evidence,
            start=1
        ):


            title = source.get(
                "title",
                "Unknown Source"
            )


            st.subheader(
                f"{number}. {title}"
            )


            credibility_level = source.get(
                "credibility_level",
                "UNKNOWN"
            )

            credibility_score = source.get(
                "credibility_score",
                0
            )

            credibility_reason = source.get(
                "credibility_reason",
                ""
            )


            if credibility_level == "HIGH":

                st.success(
                    f"🛡️ HIGH CREDIBILITY "
                    f"({credibility_score}%)"
                )

            elif credibility_level == "MEDIUM":

                st.warning(
                    f"🟡 MEDIUM CREDIBILITY "
                    f"({credibility_score}%)"
                )

            elif credibility_level == "LOW":

                st.error(
                    f"🔴 LOW CREDIBILITY "
                    f"({credibility_score}%)"
                )

            else:

                st.info(
                    f"⚪ UNKNOWN CREDIBILITY "
                    f"({credibility_score}%)"
                )


            if credibility_reason:

                st.caption(
                    credibility_reason
                )


            if source.get("url"):

                st.write(
                    f"🔗 {source['url']}"
                )


            if source.get("content"):

                source_content = source[
                    "content"
                ]


                if len(source_content) > 600:

                    source_content = (
                        source_content[:600]
                        + "..."
                    )


                st.write(
                    source_content
                )


            if "relevance_score" in source:

                st.write(
                    f"**Search Relevance:** "
                    f"{source['relevance_score']}%"
                )


            st.markdown("---")


    else:

        st.warning(
            "No relevant evidence sources were found."
        )


    # ==============================================
    # DISCLAIMER
    # ==============================================

    st.info(
        "⚠️ The machine-learning prediction, "
        "real-time web evidence, source credibility "
        "analysis, and final verdict are automated "
        "analytical signals. Important claims should "
        "still be manually verified using trusted "
        "fact-checking sources."
    )


# ==================================================
# ADVANCED DETECTION ANALYTICS DASHBOARD
# STEP 29
# ==================================================

st.markdown("---")

st.header(
    "📊 Advanced Detection Analytics Dashboard"
)


try:

    history = load_detection_history()


    if history:

        history_df = pd.DataFrame(history)


        # ==============================================
        # TOTAL DETECTIONS
        # ==============================================

        st.success(
            f"📁 Total Saved Detections: {len(history_df)}"
        )


        # ==============================================
        # DETECTION SUMMARY
        # ==============================================

        st.subheader(
            "📈 Detection Summary"
        )


        total_detections = len(history_df)


        fake_count = 0
        real_count = 0


        if "ml_prediction" in history_df.columns:

            fake_count = len(
                history_df[
                    history_df["ml_prediction"] == "FAKE"
                ]
            )

            real_count = len(
                history_df[
                    history_df["ml_prediction"] == "REAL"
                ]
            )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Total Detections",
                total_detections
            )


        with col2:

            st.metric(
                "🔴 ML Fake Predictions",
                fake_count
            )


        with col3:

            st.metric(
                "🟢 ML Real Predictions",
                real_count
            )


        # ==============================================
        # PREDICTION DISTRIBUTION CHART
        # ==============================================

        st.markdown("---")

        st.subheader(
            "📊 Machine Learning Prediction Distribution"
        )


        prediction_data = pd.DataFrame({

            "Prediction": [
                "FAKE",
                "REAL"
            ],

            "Count": [
                fake_count,
                real_count
            ]

        })


        if (
            fake_count > 0
            or real_count > 0
        ):

            prediction_chart = px.pie(

                prediction_data,

                names="Prediction",

                values="Count",

                title=(
                    "Fake vs Real Content Predictions"
                ),

                hole=0.4

            )


            st.plotly_chart(
                prediction_chart,
                width="stretch"
            )


        else:

            st.info(
                "Not enough prediction data "
                "available for visualization."
            )


        # ==============================================
        # FINAL VERDICT SUMMARY
        # ==============================================

        st.markdown("---")

        st.subheader(
            "⚖️ Final Decision Summary"
        )


        likely_true_count = 0
        likely_false_count = 0
        conflicting_count = 0
        unclear_count = 0


        if "final_verdict" in history_df.columns:

            likely_true_count = len(
                history_df[
                    history_df["final_verdict"]
                    == "LIKELY TRUE"
                ]
            )

            likely_false_count = len(
                history_df[
                    history_df["final_verdict"]
                    == "LIKELY FALSE"
                ]
            )

            conflicting_count = len(
                history_df[
                    history_df["final_verdict"]
                    == "CONFLICTING SIGNALS"
                ]
            )

            unclear_count = len(
                history_df[
                    history_df["final_verdict"]
                    == "UNCLEAR"
                ]
            )


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "🟢 Likely True",
                likely_true_count
            )


        with col2:

            st.metric(
                "🔴 Likely False",
                likely_false_count
            )


        with col3:

            st.metric(
                "🟡 Conflicting",
                conflicting_count
            )


        with col4:

            st.metric(
                "⚪ Unclear",
                unclear_count
            )


        # ==============================================
        # FINAL VERDICT BAR CHART
        # ==============================================

        verdict_data = pd.DataFrame({

            "Verdict": [

                "LIKELY TRUE",

                "LIKELY FALSE",

                "CONFLICTING SIGNALS",

                "UNCLEAR"

            ],

            "Count": [

                likely_true_count,

                likely_false_count,

                conflicting_count,

                unclear_count

            ]

        })


        if verdict_data["Count"].sum() > 0:

            verdict_chart = px.bar(

                verdict_data,

                x="Verdict",

                y="Count",

                title=(
                    "Final Decision Distribution"
                ),

                text="Count"

            )


            st.plotly_chart(
                verdict_chart,
                width="stretch"
            )


        else:

            st.info(
                "No final verdict data available yet."
            )


        # ==============================================
        # AVERAGE ANALYTICS
        # ==============================================

        st.markdown("---")

        st.subheader(
            "📊 Average Detection Metrics"
        )


        average_confidence = 0
        average_credibility = 0


        if "final_confidence" in history_df.columns:

            confidence_values = pd.to_numeric(

                history_df["final_confidence"],

                errors="coerce"

            )


            average_confidence = confidence_values.mean()


            if pd.isna(average_confidence):

                average_confidence = 0

            else:

                average_confidence = round(
                    average_confidence,
                    2
                )


        if "source_credibility" in history_df.columns:

            credibility_values = pd.to_numeric(

                history_df["source_credibility"],

                errors="coerce"

            )


            average_credibility = credibility_values.mean()


            if pd.isna(average_credibility):

                average_credibility = 0

            else:

                average_credibility = round(
                    average_credibility,
                    2
                )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Average Final Confidence",
                f"{average_confidence}%"
            )


        with col2:

            st.metric(
                "Average Source Credibility",
                f"{average_credibility}%"
            )


        # ==============================================
        # CONFIDENCE VS CREDIBILITY CHART
        # ==============================================

        if (
            "final_confidence" in history_df.columns
            and "source_credibility" in history_df.columns
        ):

            chart_df = history_df.copy()


            chart_df["final_confidence"] = pd.to_numeric(

                chart_df["final_confidence"],

                errors="coerce"

            )


            chart_df["source_credibility"] = pd.to_numeric(

                chart_df["source_credibility"],

                errors="coerce"

            )


            chart_df = chart_df.dropna(

                subset=[
                    "final_confidence",
                    "source_credibility"
                ]

            )


            if not chart_df.empty:

                st.markdown("---")

                st.subheader(
                    "📈 Confidence vs Source Credibility"
                )


                confidence_chart = px.bar(

                    chart_df,

                    y=[
                        "final_confidence",
                        "source_credibility"
                    ],

                    title=(
                        "Detection Confidence and "
                        "Source Credibility Comparison"
                    ),

                    barmode="group"

                )


                st.plotly_chart(
                    confidence_chart,
                    width="stretch"
                )


        # ==============================================
        # DETECTION HISTORY TABLE
        # ==============================================

        st.markdown("---")

        st.subheader(
            "📋 Detection History"
        )


        history_df = history_df.iloc[
            ::-1
        ].reset_index(
            drop=True
        )


        st.dataframe(

            history_df,

            width="stretch",

            hide_index=True

        )


    else:

        st.info(
            "No detection history available yet."
        )


except Exception as error:

    st.warning(
        f"Unable to load analytics dashboard: {error}"
    )
