import json

import pandas as pd
import plotly.express as px
import streamlit as st

from monitoring import get_monitoring_logs


st.set_page_config(
    page_title="AVYUHA RAG Monitoring",
    page_icon="📊",
    layout="wide"
)


st.markdown(
    """
    <style>
        .main {
            background-color: #f7f8fc;
        }

        .block-container {
            max-width: 1400px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .dashboard-subtitle {
            color: #667085;
            margin-top: -12px;
            margin-bottom: 28px;
        }

        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #e4e7ec;
            border-radius: 14px;
            padding: 18px;
            box-shadow: 0 6px 20px rgba(16, 24, 40, 0.05);
        }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("📊 AVYUHA RAG Monitoring Dashboard")

st.markdown(
    """
    <p class="dashboard-subtitle">
        Live observability for retrieval, generation,
        latency, reliability and user feedback.
    </p>
    """,
    unsafe_allow_html=True
)


if st.button("🔄 Refresh dashboard"):
    st.rerun()


logs = get_monitoring_logs()

if not logs:
    st.info(
        "No monitoring records are available yet. "
        "Ask a question in the AVYUHA RAG Assistant first."
    )
    st.stop()


dataframe = pd.DataFrame(logs)

dataframe["created_at"] = pd.to_datetime(
    dataframe["created_at"],
    utc=True
).dt.tz_convert("Asia/Kolkata")


def format_source_pages(value):
    try:
        pages = json.loads(value)
        return ", ".join(str(page) for page in pages)
    except (TypeError, ValueError, json.JSONDecodeError):
        return "None"


dataframe["formatted_pages"] = dataframe[
    "source_pages"
].apply(format_source_pages)


total_queries = len(dataframe)

successful_queries = (
    dataframe["status"] == "success"
).sum()

success_rate = (
    successful_queries / total_queries
) * 100

average_total_ms = dataframe["total_ms"].mean()
average_retrieval_ms = dataframe["retrieval_ms"].mean()
average_generation_ms = dataframe["generation_ms"].mean()

feedback_records = dataframe[
    dataframe["feedback"].notna()
]

if feedback_records.empty:
    helpful_rate_text = "No ratings"
else:
    helpful_count = (
        feedback_records["feedback"] == "helpful"
    ).sum()

    helpful_rate = (
        helpful_count / len(feedback_records)
    ) * 100

    helpful_rate_text = f"{helpful_rate:.1f}%"


st.subheader("System overview")

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    label="Total queries",
    value=f"{total_queries:,}"
)

metric_2.metric(
    label="Technical success rate",
    value=f"{success_rate:.1f}%"
)

metric_3.metric(
    label="Average response time",
    value=f"{average_total_ms / 1000:.2f} sec"
)

metric_4.metric(
    label="Helpful-answer rate",
    value=helpful_rate_text
)


st.subheader("Performance breakdown")

performance_1, performance_2, performance_3 = st.columns(3)

performance_1.metric(
    label="Average retrieval time",
    value=f"{average_retrieval_ms:.2f} ms"
)

performance_2.metric(
    label="Average generation time",
    value=f"{average_generation_ms:.2f} ms"
)

performance_3.metric(
    label="Average chunks retrieved",
    value=f"{dataframe['retrieved_chunks'].mean():.1f}"
)


chart_column_1, chart_column_2 = st.columns(2)


with chart_column_1:
    st.subheader("Queries over time")

    daily_queries = (
        dataframe
        .assign(
            query_date=dataframe["created_at"].dt.date
        )
        .groupby("query_date")
        .size()
        .reset_index(name="queries")
    )

    query_chart = px.bar(
        daily_queries,
        x="query_date",
        y="queries",
        labels={
            "query_date": "Date",
            "queries": "Queries"
        },
        color_discrete_sequence=["#7c3aed"]
    )

    query_chart.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10)
    )

    st.plotly_chart(
        query_chart,
        use_container_width=True
    )


with chart_column_2:
    st.subheader("Query status")

    status_counts = (
        dataframe["status"]
        .value_counts()
        .reset_index()
    )

    status_counts.columns = ["status", "count"]

    status_chart = px.pie(
        status_counts,
        names="status",
        values="count",
        hole=0.55,
        color="status",
        color_discrete_map={
            "success": "#22c55e",
            "error": "#ef4444"
        }
    )

    status_chart.update_layout(
        margin=dict(l=10, r=10, t=20, b=10)
    )

    st.plotly_chart(
        status_chart,
        use_container_width=True
    )


st.subheader("Retrieval versus generation latency")

latency_data = dataframe[
    [
        "id",
        "retrieval_ms",
        "generation_ms"
    ]
].sort_values("id").tail(20)

latency_data = latency_data.melt(
    id_vars="id",
    value_vars=[
        "retrieval_ms",
        "generation_ms"
    ],
    var_name="stage",
    value_name="milliseconds"
)

latency_data["stage"] = latency_data[
    "stage"
].replace({
    "retrieval_ms": "Retrieval",
    "generation_ms": "Generation"
})

latency_chart = px.line(
    latency_data,
    x="id",
    y="milliseconds",
    color="stage",
    markers=True,
    labels={
        "id": "Query ID",
        "milliseconds": "Time (milliseconds)",
        "stage": "RAG stage"
    },
    color_discrete_map={
        "Retrieval": "#2563eb",
        "Generation": "#7c3aed"
    }
)

latency_chart.update_layout(
    margin=dict(l=10, r=10, t=20, b=10)
)

st.plotly_chart(
    latency_chart,
    use_container_width=True
)


st.subheader("Recent anonymous requests")

public_table = dataframe[
    [
        "id",
        "created_at",
        "formatted_pages",
        "retrieved_chunks",
        "retrieval_ms",
        "generation_ms",
        "total_ms",
        "status",
        "feedback"
    ]
].copy()

public_table["created_at"] = public_table[
    "created_at"
].dt.strftime("%d %b %Y, %I:%M:%S %p")

public_table = public_table.rename(
    columns={
        "id": "Query ID",
        "created_at": "Time",
        "formatted_pages": "Source pages",
        "retrieved_chunks": "Chunks",
        "retrieval_ms": "Retrieval (ms)",
        "generation_ms": "Generation (ms)",
        "total_ms": "Total (ms)",
        "status": "Status",
        "feedback": "Feedback"
    }
)

st.dataframe(
    public_table,
    use_container_width=True,
    hide_index=True
)


with st.expander("View RAG monitoring architecture"):
    st.markdown(
        """
        1. The user submits a question through Streamlit.
        2. ChromaDB retrieves the six most relevant PDF chunks.
        3. Gemini generates an answer using the retrieved context.
        4. Python measures retrieval, generation and total latency.
        5. SQLite stores anonymous monitoring records.
        6. This dashboard converts the records into metrics and charts.
        """
    )


st.caption(
    "AVYUHA RAG Monitoring System • "
    "Built with Python, Streamlit, ChromaDB, Gemini, "
    "SQLite, Pandas and Plotly"
)