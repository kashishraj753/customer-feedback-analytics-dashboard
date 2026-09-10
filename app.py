import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re

st.set_page_config(
    page_title="Customer Feedback Analytics",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/customer_feedback.csv")


# -----------------------------
# Sentiment Classification
# -----------------------------
def classify_sentiment(text):
    positive_words = {
        "excellent", "great", "good", "love", "amazing",
        "happy", "helpful", "fast", "easy", "satisfied",
        "perfect", "awesome"
    }

    negative_words = {
        "bad", "poor", "hate", "slow", "difficult",
        "angry", "late", "broken", "worst", "unhappy",
        "issue", "problem", "refund"
    }

    words = set(re.findall(r"[a-z]+", str(text).lower()))

    positive_count = len(words & positive_words)
    negative_count = len(words & negative_words)

    if positive_count > negative_count:
        return "Positive"
    elif negative_count > positive_count:
        return "Negative"
    else:
        return "Neutral"


# -----------------------------
# Load & Prepare Data
# -----------------------------
df = load_data()

df["date"] = pd.to_datetime(df["date"])
df["sentiment"] = df["feedback"].apply(classify_sentiment)


# -----------------------------
# Dashboard Header
# -----------------------------
st.title("📊 Customer Feedback & Sentiment Analytics")
st.write(
    "Interactive dashboard for analyzing customer feedback, "
    "sentiment, ratings, response time and channel performance."
)


# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("🔎 Filters")

channels = st.sidebar.multiselect(
    "Select Channel",
    sorted(df["channel"].unique()),
    default=sorted(df["channel"].unique())
)

sentiments = st.sidebar.multiselect(
    "Select Sentiment",
    ["Positive", "Neutral", "Negative"],
    default=["Positive", "Neutral", "Negative"]
)

filtered_df = df[
    (df["channel"].isin(channels)) &
    (df["sentiment"].isin(sentiments))
]


if filtered_df.empty:
    st.warning("No records match the selected filters.")
    st.stop()


# -----------------------------
# KPI Section
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Feedback Records",
    f"{len(filtered_df):,}"
)

col2.metric(
    "Average Rating",
    f"{filtered_df['rating'].mean():.2f}/5"
)

col3.metric(
    "Positive Feedback",
    f"{filtered_df['sentiment'].eq('Positive').mean() * 100:.1f}%"
)

col4.metric(
    "Avg Response Time",
    f"{filtered_df['response_time_hours'].mean():.1f} hrs"
)


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(
    ["📈 Overview", "📅 Trend Analysis", "📋 Data Explorer"]
)


# =========================================================
# OVERVIEW
# =========================================================
with tab1:

    col1, col2 = st.columns(2)

    # Sentiment Distribution
    with col1:

        st.subheader("Sentiment Distribution")

        sentiment_counts = (
            filtered_df["sentiment"]
            .value_counts()
            .reindex(["Positive", "Neutral", "Negative"])
            .fillna(0)
        )

        fig, ax = plt.subplots()

        ax.bar(
            sentiment_counts.index,
            sentiment_counts.values
        )

        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Number of Feedback Records")
        ax.set_title("Customer Sentiment")

        st.pyplot(fig)


    # Channel Rating
    with col2:

        st.subheader("Average Rating by Channel")

        channel_rating = (
            filtered_df
            .groupby("channel")["rating"]
            .mean()
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots()

        ax.bar(
            channel_rating.index,
            channel_rating.values
        )

        ax.set_ylim(0, 5)
        ax.set_xlabel("Channel")
        ax.set_ylabel("Average Rating")

        ax.tick_params(axis="x", rotation=25)

        st.pyplot(fig)


    # -----------------------------
    # Business Insights
    # -----------------------------
    st.subheader("💡 Business Insights")

    best_channel = (
        filtered_df
        .groupby("channel")["rating"]
        .mean()
        .idxmax()
    )

    slowest_channel = (
        filtered_df
        .groupby("channel")["response_time_hours"]
        .mean()
        .idxmax()
    )

    negative_percentage = (
        filtered_df["sentiment"].eq("Negative").mean() * 100
    )

    st.markdown(
        f"""
        - ⭐ **Highest-rated channel:** {best_channel}
        - ⏱️ **Slowest-response channel:** {slowest_channel}
        - ⚠️ **Negative feedback:** {negative_percentage:.1f}%
        """
    )


# =========================================================
# TREND ANALYSIS
# =========================================================
with tab2:

    daily_data = (
        filtered_df
        .set_index("date")
        .resample("D")
        .agg(
            feedback_count=("feedback", "count"),
            average_rating=("rating", "mean")
        )
    )

    st.subheader("📊 Feedback Volume Over Time")

    st.line_chart(
        daily_data["feedback_count"]
    )

    st.subheader("⭐ Average Rating Over Time")

    st.line_chart(
        daily_data["average_rating"]
    )


# =========================================================
# DATA EXPLORER
# =========================================================
with tab3:

    st.subheader("📋 Filtered Customer Feedback")

    st.dataframe(
        filtered_df.sort_values(
            "date",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Filtered CSV",
        data=csv,
        file_name="filtered_customer_feedback.csv",
        mime="text/csv"
    )


# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "Portfolio Project | Python • Pandas • NLP • Data Analytics • Streamlit"
)
