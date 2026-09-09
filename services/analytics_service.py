import pandas as pd


# ==================================================
# CREATE HISTORY DATAFRAME
# ==================================================

def create_history_dataframe(history):

    if not history:

        return pd.DataFrame()

    dataframe = pd.DataFrame(history)

    return dataframe


# ==================================================
# PREDICTION STATISTICS
# ==================================================

def get_prediction_statistics(dataframe):

    if dataframe.empty:

        return {
            "total": 0,
            "fake": 0,
            "real": 0
        }


    prediction_column = "prediction"


    if prediction_column not in dataframe.columns:

        return {
            "total": len(dataframe),
            "fake": 0,
            "real": 0
        }


    fake_count = len(

        dataframe[
            dataframe[prediction_column] == "FAKE"
        ]

    )


    real_count = len(

        dataframe[
            dataframe[prediction_column] == "REAL"
        ]

    )


    return {

        "total": len(dataframe),

        "fake": fake_count,

        "real": real_count

    }


# ==================================================
# FINAL VERDICT STATISTICS
# ==================================================

def get_verdict_statistics(dataframe):

    if dataframe.empty:

        return {}


    verdict_column = "final_verdict"


    if verdict_column not in dataframe.columns:

        return {}


    verdict_counts = (

        dataframe[verdict_column]
        .value_counts()
        .to_dict()

    )


    return verdict_counts


# ==================================================
# AVERAGE CONFIDENCE
# ==================================================

def get_average_confidence(dataframe):

    if dataframe.empty:

        return 0


    confidence_column = "final_confidence"


    if confidence_column not in dataframe.columns:

        return 0


    confidence_values = pd.to_numeric(

        dataframe[confidence_column],

        errors="coerce"

    ).dropna()


    if confidence_values.empty:

        return 0


    return round(

        confidence_values.mean(),

        2

    )


# ==================================================
# AVERAGE SOURCE CREDIBILITY
# ==================================================

def get_average_credibility(dataframe):

    if dataframe.empty:

        return 0


    credibility_column = "source_credibility"


    if credibility_column not in dataframe.columns:

        return 0


    credibility_values = pd.to_numeric(

        dataframe[credibility_column],

        errors="coerce"

    ).dropna()


    if credibility_values.empty:

        return 0


    return round(

        credibility_values.mean(),

        2

    )