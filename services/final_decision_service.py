def make_final_decision(
    prediction,
    evidence_status,
    evidence
):

    # ==========================================
    # CALCULATE AVERAGE SOURCE CREDIBILITY
    # ==========================================

    credibility_scores = []

    for source in evidence:

        score = source.get(
            "credibility_score",
            50
        )

        credibility_scores.append(score)


    if credibility_scores:

        average_credibility = round(
            sum(credibility_scores)
            / len(credibility_scores),
            2
        )

    else:

        average_credibility = 0


    # ==========================================
    # COUNT HIGH QUALITY SOURCES
    # ==========================================

    strong_sources = 0


    for source in evidence:

        credibility = source.get(
            "credibility_score",
            0
        )

        if credibility >= 60:

            strong_sources += 1


    # ==========================================
    # HYBRID DECISION LOGIC
    # ==========================================

    # ------------------------------------------
    # CASE 1: Evidence strongly supports claim
    # ------------------------------------------

    if (
        evidence_status == "SUPPORTING"
        and strong_sources >= 2
        and average_credibility >= 50
    ):

        return {

            "verdict": "LIKELY TRUE",

            "confidence": 90,

            "average_credibility": average_credibility,

            "reason": (
                "Multiple relevant web sources support "
                "the claim. Real-time evidence verification "
                "indicates that the content is likely true, "
                "even if the machine-learning model prediction "
                "differs."
            )

        }


    # ------------------------------------------
    # CASE 2: Evidence contradicts claim
    # ------------------------------------------

    elif (
        evidence_status == "CONTRADICTING"
        and strong_sources >= 2
    ):

        return {

            "verdict": "LIKELY FALSE",

            "confidence": 90,

            "average_credibility": average_credibility,

            "reason": (
                "Multiple relevant evidence sources contradict "
                "the claim. Real-time verification indicates "
                "that the content is likely false."
            )

        }


    # ------------------------------------------
    # CASE 3: ML and evidence agree on REAL
    # ------------------------------------------

    elif (
        prediction == "REAL"
        and evidence_status == "SUPPORTING"
    ):

        return {

            "verdict": "LIKELY TRUE",

            "confidence": 85,

            "average_credibility": average_credibility,

            "reason": (
                "The machine-learning model and real-time "
                "evidence both support the authenticity "
                "of the content."
            )

        }


    # ------------------------------------------
    # CASE 4: ML and evidence agree on FAKE
    # ------------------------------------------

    elif (
        prediction == "FAKE"
        and evidence_status == "CONTRADICTING"
    ):

        return {

            "verdict": "LIKELY FALSE",

            "confidence": 85,

            "average_credibility": average_credibility,

            "reason": (
                "The machine-learning model and real-time "
                "evidence both indicate that the content "
                "is likely false."
            )

        }


    # ------------------------------------------
    # CASE 5: ML says REAL but evidence unclear
    # ------------------------------------------

    elif prediction == "REAL":

        return {

            "verdict": "LIKELY TRUE",

            "confidence": 65,

            "average_credibility": average_credibility,

            "reason": (
                "The machine-learning model predicts real "
                "content, but real-time evidence is not "
                "strong enough for complete verification."
            )

        }


    # ------------------------------------------
    # CASE 6: ML says FAKE but evidence unclear
    # ------------------------------------------

    elif prediction == "FAKE":

        return {

            "verdict": "LIKELY FALSE",

            "confidence": 65,

            "average_credibility": average_credibility,

            "reason": (
                "The machine-learning model predicts fake "
                "content, but sufficient real-time evidence "
                "was not available for stronger verification."
            )

        }


    # ------------------------------------------
    # DEFAULT CASE
    # ------------------------------------------

    else:

        return {

            "verdict": "UNCLEAR",

            "confidence": 50,

            "average_credibility": average_credibility,

            "reason": (
                "There is insufficient information to make "
                "a reliable final decision."
            )

        }