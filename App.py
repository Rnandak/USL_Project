import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Mobile Reviews Analytics Platform",
    page_icon="📱",
    layout="wide"
)

# -----------------------------------------------------------------------------
# STYLING
# -----------------------------------------------------------------------------

st.markdown("""
<style>

.main-title {
    font-size:36px;
    font-weight:bold;
    color:#1E3A8A;
}

.sub-title {
    color:#666666;
    font-size:16px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(
        r"C:\Users\Nanda\OneDrive\Documents\USL_Project\Dataset\Mobile_Reviews_Sentiment_null.csv",
        encoding_errors="ignore"
    )

    df["price_local"] = (
        df["price_local"]
        .str.extract(r'(\d+\.?\d*)')[0]
        .astype(float)
    )

    numcols = df.select_dtypes(include="number").columns
    catcols = df.select_dtypes(exclude="number").columns

    df[numcols] = df[numcols].fillna(
        df[numcols].mean()
    )

    df[catcols] = df[catcols].fillna(
        df[catcols].mode().iloc[0]
    )

    return df

df = load_data()

# -----------------------------------------------------------------------------
# TITLE
# -----------------------------------------------------------------------------

st.markdown(
    '<div class="main-title">📱 Mobile Reviews Analytics Platform</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">K-Means • DBSCAN • PCA • SVD • ALS</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# -----------------------------------------------------------------------------
# KPI SECTION
# -----------------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Reviews",
        f"{len(df):,}"
    )

with c2:
    st.metric(
        "Brands",
        df["brand"].nunique()
    )

with c3:
    st.metric(
        "Models",
        df["model"].nunique()
    )

with c4:
    st.metric(
        "Average Rating",
        round(df["rating"].mean(), 2)
    )

st.markdown("---")

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Details and Contents",
    "🎯 K-Means",
    "🚨 DBSCAN",
    "🧬 PCA",
    "🧠 Recommendation Engine"
])

# -----------------------------------------------------------------------------
# PLACEHOLDERS
# -----------------------------------------------------------------------------
with tab1:

    st.header("📋 Project Overview")

    st.markdown("""
    ## Mobile Reviews Analytics & Recommendation System

    This project applies multiple Machine Learning techniques on a mobile
    reviews dataset containing 50,000 records.

    ### Objectives Completed

    ✅ Data Cleaning and Missing Value Treatment

    ✅ Feature Standardization using StandardScaler

    ✅ Customer Segmentation using K-Means Clustering

    ✅ Cluster Evaluation using Silhouette Score

    ✅ Anomaly Detection using DBSCAN

    ✅ Dimensionality Reduction using Principal Component Analysis (PCA)

    ✅ Matrix Factorization using Truncated SVD

    ✅ Recommendation Engine Construction using Collaborative Filtering

    ✅ ALS (Alternating Least Squares) Recommendation Modelling using PySpark

    ### Dataset Features Used

    - age
    - price_usd
    - battery_life_rating
    - camera_rating
    - performance_rating
    - design_rating
    - display_rating
    - rating
    - model
    - brand

    ### Machine Learning Techniques Applied

    #### K-Means Clustering
    Used for customer/device segmentation.

    #### DBSCAN
    Used for anomaly detection and outlier identification.

    #### PCA
    Used for dimensionality reduction and visualization.

    #### Truncated SVD
    Used for collaborative filtering and recommendation generation.

    #### ALS (PySpark)
    Used for large-scale recommendation modelling using latent factors.

    ### Key Outcomes

    - Segmented customers into meaningful groups.
    - Identified anomalous observations.
    - Reduced feature space using PCA.
    - Built recommendation systems using SVD and ALS.
    - Predicted unseen user-item preferences based on latent factors.
    """)

    st.success(
        "Project successfully demonstrates Clustering, Anomaly Detection, "
        "Dimensionality Reduction and Recommendation Systems."
    )


with tab2:

    st.header("🎯 K-Means Customer Segmentation")

    st.markdown("""
    K-Means clustering was performed using:

    - Age
    - Price (USD)
    - Battery Life Rating
    - Camera Rating

    All variables were standardized using StandardScaler prior to clustering.
    """)

    k_input = st.slider(
        "Select Number of Clusters",
        min_value=2,
        max_value=15,
        value=10
    )

    kmeans_data = df[
        [
            "age",
            "price_usd",
            "battery_life_rating",
            "camera_rating"
        ]
    ].copy()

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        kmeans_data
    )

    km = KMeans(
        n_clusters=k_input,
        max_iter=300,
        random_state=42
    )

    labels = km.fit_predict(
        scaled_data
    )

    kmeans_data["Group"] = labels

    silhouette = silhouette_score(
        scaled_data,
        labels,
        sample_size=1000,
        random_state=42
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "No. of Clusters",
            k_input
        )

    with col2:
        st.metric(
            "Silhouette Score",
            round(silhouette, 3)
        )

    st.markdown("---")

    st.subheader("Cluster Profile Summary")

    cluster_profile = (
        kmeans_data
        .groupby("Group")
        .mean()
        .round(2)
        .reset_index()
    )

    st.dataframe(
        cluster_profile,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Cluster Centroid Visualization")

    fig_cluster = px.scatter(
        cluster_profile,
        x="price_usd",
        y="battery_life_rating",
        color="Group",
        size="camera_rating",
        text="Group",
        template="plotly_dark"
    )

    fig_cluster.update_traces(
        textposition="top center"
    )

    st.plotly_chart(
        fig_cluster,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Silhouette Score Analysis")

    st.write(
        f"Current K = {k_input}"
    )

    st.write(
        f"Silhouette Score = {silhouette:.3f}"
    )

with tab3:

    st.header("🚨 DBSCAN Anomaly Detection")

    st.markdown("""
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
    was used to identify dense groups and unusual observations.

    Parameters used:

    - eps = 0.75
    - min_samples = 5
    """)

    kmeans_data = df[
        [
            "age",
            "price_usd",
            "battery_life_rating",
            "camera_rating"
        ]
    ].copy()

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        kmeans_data
    )

    dbs = DBSCAN(
        eps=0.75,
        min_samples=5
    )

    dbs_labels = dbs.fit_predict(
        scaled_data
    )

    kmeans_data["DBS"] = dbs_labels

    cluster_counts = (
        kmeans_data["DBS"]
        .value_counts()
        .sort_index()
    )

    st.subheader("Cluster Distribution")

    st.dataframe(
        cluster_counts.to_frame("Count"),
        use_container_width=True
    )

    outlier_count = (
        kmeans_data["DBS"] == -1
    ).sum()

    st.metric(
        "Outliers Detected",
        outlier_count
    )

    st.markdown("---")

    st.subheader("DBSCAN Cluster Profile")

    dbs_summary = (
        kmeans_data
        .groupby("DBS")
        .mean()
        .round(2)
        .reset_index()
    )

    st.dataframe(
        dbs_summary,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Outlier Visualization")

    fig_dbs = px.scatter(
        dbs_summary,
        x="age",
        y="price_usd",
        color="DBS",
        size="battery_life_rating",
        text="DBS",
        template="plotly_dark"
    )

    fig_dbs.update_traces(
        textposition="top center"
    )

    st.plotly_chart(
        fig_dbs,
        use_container_width=True
    )

    if outlier_count > 0:
        st.warning(
            f"DBSCAN identified {outlier_count:,} observations as anomalies (Cluster = -1)."
        )
    else:
        st.success(
            "No outliers were detected using the current parameters."
        )

with tab4:

    st.header("🧬 Principal Component Analysis (PCA)")

    st.markdown("""
    PCA was applied on the following standardized features:

    - Age
    - Price (USD)
    - Battery Life Rating
    - Camera Rating

    The objective was to reduce dimensionality while retaining
    maximum information from the original feature space.
    """)

    pca = PCA(n_components=4)

    pca_data = pca.fit_transform(
        scaled_data
    )

    pca_df = pd.DataFrame(
        pca_data,
        columns=[
            "PC1",
            "PC2",
            "PC3",
            "PC4"
        ]
    )

    var_ratios = pca.explained_variance_ratio_

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Total Variance Retained",
            f"{var_ratios.sum():.2%}"
        )

    with c2:

        st.metric(
            "Principal Components",
            len(var_ratios)
        )

    st.markdown("---")

    st.subheader("Explained Variance Ratio")

    variance_df = pd.DataFrame({
        "Component": [
            "PC1",
            "PC2",
            "PC3",
            "PC4"
        ],
        "Variance Explained": var_ratios
    })

    fig_variance = px.bar(
        variance_df,
        x="Component",
        y="Variance Explained",
        color="Variance Explained",
        text_auto=".2f",
        template="plotly_dark"
    )

    st.plotly_chart(
        fig_variance,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Principal Component Summary")

    st.dataframe(
        variance_df,
        use_container_width=True
    )

    st.info(
        f"""
        PC1 explains {var_ratios[0]:.2%} of the total variance.

        PC2 explains {var_ratios[1]:.2%} of the total variance.

        PC3 explains {var_ratios[2]:.2%} of the total variance.

        PC4 explains {var_ratios[3]:.2%} of the total variance.
        """
    )

    st.markdown("---")

    st.subheader("3D PCA Projection")

    sample_df = pca_df.sample(
        n=min(5000, len(pca_df)),
        random_state=42
    )

    fig_3d = px.scatter_3d(
        sample_df,
        x="PC1",
        y="PC2",
        z="PC3",
        opacity=0.60,
        template="plotly_dark"
    )

    fig_3d.update_layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )

    st.plotly_chart(
        fig_3d,
        use_container_width=True
    )

    st.success(
        "PCA successfully transformed the original 4-dimensional feature space into orthogonal principal components."
    )

with tab5:

    st.header("🧠 Recommendation Engine - SVD Matrix Factorization")

    st.subheader("Truncated SVD Recommendation Engine")

    mfactorize = (
            df[
                [
                    "review_id",
                    "customer_name",
                    "performance_rating"
                ]
            ]
            .iloc[:5000, :]
        )

    pivotted = mfactorize.pivot(
            index="customer_name",
            columns="review_id",
            values="performance_rating"
        )

    pivottedfillna = pivotted.fillna(0)

    svd = TruncatedSVD(
            n_components=5,
            random_state=42
        )

    cust_pred = svd.fit_transform(
            pivottedfillna
        )

    predicted_ratings = np.dot(
            cust_pred,
            svd.components_
        )

    pred_df = pd.DataFrame(
            predicted_ratings,
            index=pivotted.index,
            columns=pivotted.columns
        )

    customer = st.selectbox(
            "Select Customer",
            sorted(pivotted.index)
        )

    perf_rating = (
            pivotted
            .loc[customer]
            .dropna()
            .index
        )

    recommend = pred_df.loc[customer]

    recommend = recommend.drop(
            perf_rating,
            errors="ignore"
        )

    top5 = recommend.sort_values(
            ascending=False
        ).head(5)

    st.subheader(
            f"Top 5 Predicted Recommendations for {customer}"
        )

    st.dataframe(
            top5.reset_index().rename(
                columns={
                    "review_id": "Recommended Review ID",
                    customer: "Affinity Score"
                }
            ),
            use_container_width=True
        )

    st.info("""
        Higher affinity scores indicate stronger predicted preference
        based on collaborative filtering patterns learned from similar users.
        """)

 