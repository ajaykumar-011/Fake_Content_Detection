import re


# ==================================================
# EXTRACT CLAIM
# ==================================================

def extract_claim(text):

    text = text.strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ==================================================
# HELPER FUNCTION
# ==================================================

def get_meaningful_words(text):

    stop_words = {
        "the", "and", "that", "this", "with",
        "from", "have", "has", "had", "were",
        "was", "are", "for", "into", "onto",
        "their", "they", "them", "been",
        "successfully", "officially"
    }

    words = re.findall(
        r"\b[a-zA-Z0-9]{3,}\b",
        text.lower()
    )

    return {
        word for word in words
        if word not in stop_words
    }


# ==================================================
# DEBUNKING LANGUAGE DETECTION
# ==================================================

def has_debunking_language(text):

    """
    Detects language that explicitly indicates a
    source is refuting or debunking a claim.

    NOTE: Ordinary negation words like "not", "never",
    "didn't" are intentionally excluded here, since they
    appear naturally in normal factual sentences (e.g.
    "the capsule did not land on the Moon") and do not,
    by themselves, indicate that a source disputes a claim.
    """

    debunking_patterns = [
        r"\bfalse\b",
        r"\bhoax\b",
        r"\bfake\b",
        r"\bdebunked\b",
        r"\bdenied\b",
        r"\bincorrect\b",
        r"\bmyth\b",
        r"\bnot true\b",
        r"\bno evidence\b",
        r"\bmisleading\b"
    ]

    text = text.lower()

    return any(
        re.search(pattern, text)
        for pattern in debunking_patterns
    )


# ==================================================
# ADVANCED EVIDENCE ANALYSIS
# ==================================================

def analyze_evidence(claim, evidence_sources):

    if not evidence_sources:

        return {
            "status": "INSUFFICIENT EVIDENCE",
            "score": 0,
            "reason": "No evidence sources were found.",
            "best_source": None
        }


    claim_words = get_meaningful_words(
        claim
    )


    if not claim_words:

        return {
            "status": "INSUFFICIENT EVIDENCE",
            "score": 0,
            "reason": (
                "The claim does not contain enough "
                "meaningful words for analysis."
            ),
            "best_source": None
        }


    best_score = 0
    best_source = None
    contradiction_detected = False


    # ==================================================
    # ANALYZE EACH SOURCE
    # ==================================================

    for source in evidence_sources:

        title = source.get(
            "title",
            ""
        )

        content = source.get(
            "content",
            ""
        )


        evidence_text = (
            title + " " + content
        )


        evidence_words = get_meaningful_words(
            evidence_text
        )


        common_words = claim_words.intersection(
            evidence_words
        )


        score = (
            len(common_words)
            / len(claim_words)
        ) * 100


        # Update best source

        if score > best_score:

            best_score = score
            best_source = source


        # ==============================================
        # IMPROVED CONTRADICTION DETECTION
        # ==============================================

        # A source is only treated as contradicting the
        # claim when it is highly relevant AND it contains
        # genuine debunking language (e.g. "hoax", "false",
        # "denied", "debunked") - not just because either
        # text happens to contain an ordinary negation word
        # like "not" or "never".

        evidence_has_debunking = has_debunking_language(
            evidence_text
        )


        if (
            score >= 50
            and evidence_has_debunking
        ):

            contradiction_detected = True


    best_score = round(
        best_score,
        2
    )


    # ==================================================
    # FINAL DECISION
    # ==================================================

    if contradiction_detected:

        status = "CONTRADICTING"

        reason = (
            "The claim and highly relevant evidence "
            "appear to express opposing statements."
        )


    elif best_score >= 60:

        status = "SUPPORTING"

        reason = (
            "Strong similarity was found between the "
            "claim and retrieved evidence."
        )


    elif best_score >= 30:

        status = "RELEVANT"

        reason = (
            "Relevant evidence was found, but the "
            "claim cannot be conclusively verified."
        )


    else:

        status = "INSUFFICIENT EVIDENCE"

        reason = (
            "The retrieved evidence does not contain "
            "enough relevant information to verify "
            "the claim."
        )


    return {

        "status": status,

        "score": best_score,

        "reason": reason,

        "best_source": best_source

    }