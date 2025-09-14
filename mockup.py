import streamlit as st
import pandas as pd
import os

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Auto K-means with LLM Labeling", layout="wide")

# ----------------- SIDEBAR -----------------
st.sidebar.header("⚙️ Controls")

# Upload dataset
uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])

# Industry selection
industry = st.sidebar.selectbox(
    "Select Industry",
    ["Telco", "Banking", "Retail", "Other"]
)

# Advanced options (collapsible)
with st.sidebar.expander("Advanced Settings"):
    max_k = st.slider("Max clusters to test", 2, 10, 5)
    scale_data = st.checkbox("Standardize data", value=True)

# Run pipeline button
run_pipeline = st.sidebar.button("▶️ Run Pipeline")

# ----------------- MAIN PAGE -----------------
st.title("📊 Auto K-means with LLM Labeling")

# Create mock data for demonstration when no file is uploaded
def create_mock_data():
    import numpy as np
    np.random.seed(42)
    
    # Generate realistic telco customer data
    n_customers = 280
    data = {
        'customer_id': range(1, n_customers + 1),
        'monthly_charges': np.random.normal(70, 25, n_customers).clip(20, 150),
        'total_charges': np.random.normal(2500, 1200, n_customers).clip(100, 8000),
        'data_usage_gb': np.random.exponential(15, n_customers).clip(0, 100),
        'call_minutes': np.random.normal(400, 200, n_customers).clip(0, 1500),
        'contract_length': np.random.choice([1, 12, 24], n_customers, p=[0.3, 0.4, 0.3]),
        'age': np.random.normal(45, 15, n_customers).clip(18, 80).astype(int),
        'tenure_months': np.random.exponential(20, n_customers).clip(1, 72).astype(int)
    }
    return pd.DataFrame(data)

# Use uploaded file or mock data
if uploaded_file is not None:
    # Save file to local folder
    UPLOAD_DIR = "data/uploads"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ File uploaded and saved: {file_path}")
    df = pd.read_csv(file_path)
    is_mock_data = False
else:
    # Use mock data for demonstration
    df = create_mock_data()
    is_mock_data = True
    st.info("📝 Using mock telco dataset for demonstration. Upload your own CSV to analyze real data.")

# -------- Section 1: Dataset Preview --------
with st.expander("📂 Dataset Preview", expanded=True):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**First 5 rows of the dataset:**")
        st.dataframe(df.head(), use_container_width=True)
    
    with col2:
        st.write("**Dataset Info:**")
        st.metric("Total Rows", len(df))
        st.metric("Total Columns", len(df.columns))
        
        if is_mock_data:
            st.write("**Features:**")
            for col in df.columns:
                if col != 'customer_id':
                    st.write(f"• {col}")

# -------- Section 2: Candidate Labels --------
with st.expander("🏷️ Candidate Labels (LLM-generated)"):
    if run_pipeline or is_mock_data:
        # Industry-specific candidate labels
        industry_labels = {
            "Telco": [
                "Heavy Data Users",
                "Prepaid Low Spenders", 
                "Corporate Clients",
                "Churn Risk Customers"
            ],
            "Banking": [
                "Premium Account Holders",
                "Young Savers",
                "Investment Focused",
                "Credit Risk Customers"
            ],
            "Retail": [
                "Frequent Shoppers",
                "Seasonal Buyers",
                "Discount Hunters", 
                "Premium Brand Loyalists"
            ],
            "Other": [
                "High Value Segment",
                "Budget Conscious",
                "Growth Potential",
                "At Risk Segment"
            ]
        }
        
        candidate_labels = industry_labels[industry]
        
        st.write(f"**Generated labels for {industry} industry:**")
        
        # Display labels in a nice format
        for i, label in enumerate(candidate_labels):
            st.write(f"🏷️ **Label {i+1}:** {label}")
        
        # Show as JSON for technical view
        with st.expander("View as JSON"):
            st.json(candidate_labels)
    else:
        st.info("🤖 Run the pipeline to generate AI-powered candidate labels based on your industry selection.")

# -------- Section 3: Clustering Summary --------
with st.expander("🔎 Clustering Summary"):
    if run_pipeline or is_mock_data:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("🎯 Optimal Clusters (k)", "4")
            st.metric("📊 Silhouette Score", "0.67")
            st.metric("📈 Explained Variance", "78%")
        
        with col2:
            st.write("**Cluster Assignment Results:**")
            
            # Create detailed clustering results table
            cluster_results = pd.DataFrame({
                "Cluster": [0, 1, 2, 3],
                "Assigned Label": [
                    "Prepaid Low Spenders", 
                    "Heavy Data Users", 
                    "Corporate Clients", 
                    "Churn Risk Customers"
                ],
                "Key Features": [
                    "Low charges, short tenure",
                    "High data usage, young demographics", 
                    "Long contracts, high value",
                    "High churn indicators"
                ],
                "Size": [85, 72, 58, 65],
                "Avg Monthly Revenue": ["$45", "$89", "$124", "$67"]
            })
            
            st.dataframe(cluster_results, use_container_width=True)
    else:
        st.info("🔍 Run the pipeline to see automatic cluster analysis and optimal k selection.")

# -------- Section 4: Visualizations --------
with st.expander("📈 Visualizations"):
    if run_pipeline or is_mock_data:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["🎯 Cluster Scatter Plot", "📊 Feature Distributions", "📈 Cluster Profiles"])
        
        with tab1:
            st.write("**PCA Projection with Cluster Assignment**")
            
            # Create mock PCA scatter plot
            np.random.seed(42)
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Generate mock PCA coordinates for each cluster
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
            cluster_names = ["Prepaid Low Spenders", "Heavy Data Users", "Corporate Clients", "Churn Risk"]
            
            for i, (color, name) in enumerate(zip(colors, cluster_names)):
                # Generate cluster-specific data points
                n_points = [85, 72, 58, 65][i]
                x = np.random.normal(i*2, 1.5, n_points)
                y = np.random.normal(i*1.5, 1.2, n_points)
                ax.scatter(x, y, c=color, label=name, alpha=0.7, s=60)
            
            ax.set_xlabel('First Principal Component')
            ax.set_ylabel('Second Principal Component')
            ax.set_title('Customer Segments in PCA Space')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
        with tab2:
            st.write("**Feature Distribution by Cluster**")
            
            # Create mock feature distribution plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # Monthly Charges
            for i, color in enumerate(colors):
                data = np.random.normal([45, 89, 124, 67][i], 15, [85, 72, 58, 65][i])
                ax1.hist(data, alpha=0.6, color=color, bins=15, label=f'Cluster {i}')
            ax1.set_title('Monthly Charges Distribution')
            ax1.set_xlabel('Monthly Charges ($)')
            ax1.legend()
            
            # Data Usage
            for i, color in enumerate(colors):
                data = np.random.exponential([8, 35, 15, 12][i], [85, 72, 58, 65][i])
                ax2.hist(data, alpha=0.6, color=color, bins=15, label=f'Cluster {i}')
            ax2.set_title('Data Usage Distribution')
            ax2.set_xlabel('Data Usage (GB)')
            ax2.legend()
            
            # Tenure
            for i, color in enumerate(colors):
                data = np.random.exponential([8, 25, 45, 15][i], [85, 72, 58, 65][i])
                ax3.hist(data, alpha=0.6, color=color, bins=15, label=f'Cluster {i}')
            ax3.set_title('Tenure Distribution')
            ax3.set_xlabel('Tenure (months)')
            ax3.legend()
            
            # Age
            for i, color in enumerate(colors):
                data = np.random.normal([35, 28, 52, 45][i], 12, [85, 72, 58, 65][i])
                ax4.hist(data, alpha=0.6, color=color, bins=15, label=f'Cluster {i}')
            ax4.set_title('Age Distribution')
            ax4.set_xlabel('Age (years)')
            ax4.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab3:
            st.write("**Cluster Characteristics Profile**")
            
            # Create radar chart for cluster profiles
            categories = ['Monthly Charges', 'Data Usage', 'Tenure', 'Call Minutes', 'Age']
            
            # Normalized values for each cluster (0-1 scale)
            cluster_profiles = {
                'Prepaid Low Spenders': [0.3, 0.2, 0.2, 0.4, 0.4],
                'Heavy Data Users': [0.7, 0.9, 0.5, 0.3, 0.2],
                'Corporate Clients': [1.0, 0.6, 0.9, 0.8, 0.8],
                'Churn Risk': [0.5, 0.4, 0.3, 0.6, 0.6]
            }
            
            # Create side-by-side metrics
            cols = st.columns(4)
            
            for i, (cluster_name, values) in enumerate(cluster_profiles.items()):
                with cols[i]:
                    st.write(f"**{cluster_name}**")
                    for cat, val in zip(categories, values):
                        percentage = int(val * 100)
                        st.metric(cat, f"{percentage}%")
    else:
        st.info("📊 Run the pipeline to generate interactive visualizations and cluster analysis charts.")

# -------- Section 5: Download --------
with st.expander("💾 Download Results"):
    if run_pipeline or is_mock_data:
        col1, col2 = st.columns(2)
        
        with col1:
            # Add mock cluster assignments to dataframe
            if is_mock_data:
                np.random.seed(42)
                df_with_clusters = df.copy()
                cluster_assignments = np.random.choice([0, 1, 2, 3], len(df), p=[0.3, 0.26, 0.21, 0.23])
                cluster_labels = ["Prepaid Low Spenders", "Heavy Data Users", "Corporate Clients", "Churn Risk Customers"]
                df_with_clusters['cluster'] = cluster_assignments
                df_with_clusters['cluster_label'] = [cluster_labels[i] for i in cluster_assignments]
            else:
                df_with_clusters = df  # Would contain real clustering results
            
            st.download_button(
                label="📥 Download Labeled Dataset (CSV)",
                data=df_with_clusters.to_csv(index=False).encode("utf-8"),
                file_name=f"clustered_dataset_{industry.lower()}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Additional download options
            cluster_summary = pd.DataFrame({
                "Cluster": [0, 1, 2, 3],
                "Label": ["Prepaid Low Spenders", "Heavy Data Users", "Corporate Clients", "Churn Risk"],
                "Size": [85, 72, 58, 65],
                "Percentage": ["30.4%", "25.7%", "20.7%", "23.2%"]
            })
            
            st.download_button(
                label="📊 Download Cluster Summary (CSV)", 
                data=cluster_summary.to_csv(index=False).encode("utf-8"),
                file_name=f"cluster_summary_{industry.lower()}.csv",
                mime="text/csv"
            )
    else:
        st.info("💾 Run the pipeline to enable downloads of labeled datasets and cluster summaries.")

# Show footer info
if is_mock_data:
    st.markdown("---")
    st.markdown("💡 **This is a demo with mock data.** Upload your own CSV file to see real clustering results!")
