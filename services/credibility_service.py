from urllib.parse import urlparse


# ==================================================
# SOURCE CREDIBILITY DATABASE
# ==================================================

HIGH_CREDIBILITY_DOMAINS = [

    # Government / Official
    "isro.gov.in",
    "nasa.gov",
    "esa.int",
    "who.int",
    "un.org",

    # Major trusted news agencies
    "reuters.com",
    "apnews.com",
    "bbc.com",

    # Scientific / Space
    "nature.com",
    "science.org"
]


MEDIUM_CREDIBILITY_DOMAINS = [

    "wikipedia.org",
    "firstpost.com",
    "thehindu.com",
    "indianexpress.com",
    "ndtv.com",
    "timesofindia.com",
    "hindustantimes.com",
    "cnn.com",
    "theguardian.com",
    "nytimes.com",
    "planetary.org"
]


LOW_CREDIBILITY_DOMAINS = [

    "instagram.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "youtube.com",
    "tiktok.com",
    "blogspot.com",
    "wordpress.com"
]


# ==================================================
# EXTRACT DOMAIN
# ==================================================

def get_domain(url):

    try:

        parsed_url = urlparse(url)

        domain = parsed_url.netloc.lower()

        domain = domain.replace(
            "www.",
            ""
        )

        return domain

    except Exception:

        return ""


# ==================================================
# CHECK DOMAIN MATCH
# ==================================================

def domain_matches(domain, trusted_domain):

    return (
        domain == trusted_domain
        or domain.endswith("." + trusted_domain)
    )


# ==================================================
# CALCULATE SOURCE CREDIBILITY
# ==================================================

def calculate_credibility(url):

    domain = get_domain(url)


    # ----------------------------------------------
    # INVALID DOMAIN
    # ----------------------------------------------

    if not domain:

        return {
            "score": 30,
            "level": "LOW",
            "reason": "No valid website domain was available."
        }


    # ----------------------------------------------
    # HIGH CREDIBILITY
    # ----------------------------------------------

    for trusted_domain in HIGH_CREDIBILITY_DOMAINS:

        if domain_matches(domain, trusted_domain):

            return {
                "score": 95,
                "level": "HIGH",
                "reason": (
                    "Recognized official, scientific, "
                    "government, or highly trusted news source."
                )
            }


    # ----------------------------------------------
    # MEDIUM CREDIBILITY
    # ----------------------------------------------

    for trusted_domain in MEDIUM_CREDIBILITY_DOMAINS:

        if domain_matches(domain, trusted_domain):

            return {
                "score": 70,
                "level": "MEDIUM",
                "reason": (
                    "Recognized established news, "
                    "educational, or information source."
                )
            }


    # ----------------------------------------------
    # LOW CREDIBILITY
    # ----------------------------------------------

    for low_domain in LOW_CREDIBILITY_DOMAINS:

        if domain_matches(domain, low_domain):

            return {
                "score": 35,
                "level": "LOW",
                "reason": (
                    "Social-media or user-generated content "
                    "should be independently verified."
                )
            }


    # ----------------------------------------------
    # GOVERNMENT / EDUCATIONAL DOMAIN DETECTION
    # ----------------------------------------------

    if domain.endswith(".gov") or ".gov." in domain:

        return {
            "score": 90,
            "level": "HIGH",
            "reason": (
                "Government or official institutional domain."
            )
        }


    if domain.endswith(".edu"):

        return {
            "score": 85,
            "level": "HIGH",
            "reason": (
                "Recognized educational or academic domain."
            )
        }


    # ----------------------------------------------
    # UNKNOWN SOURCE
    # ----------------------------------------------

    return {
        "score": 50,
        "level": "UNKNOWN",
        "reason": (
            "The source is not currently included "
            "in the credibility database."
        )
    }


# ==================================================
# ADD CREDIBILITY TO ALL SOURCES
# ==================================================

def rank_sources(evidence_sources):

    ranked_sources = []


    for source in evidence_sources:

        # Create a copy to avoid unexpected mutation
        ranked_source = source.copy()


        url = ranked_source.get(
            "url",
            ""
        )


        credibility = calculate_credibility(url)


        ranked_source["credibility_score"] = (
            credibility["score"]
        )

        ranked_source["credibility_level"] = (
            credibility["level"]
        )

        ranked_source["credibility_reason"] = (
            credibility["reason"]
        )


        ranked_sources.append(
            ranked_source
        )


    # Sort highest credibility first

    ranked_sources.sort(
        key=lambda source: source.get(
            "credibility_score",
            0
        ),
        reverse=True
    )


    return ranked_sources