import streamlit as st

st.set_page_config(
    page_title="About QUEST",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 What is QUEST?")

st.info("💡 **QUEST: Quezon City Evacuation Support Tool** is designed to be a dashboard that provides data during times of typhoon- and rainfall-induced floods on which relief operations centers to seek out, the optimal route to travel to these centers in order to maximize preparedness and strengthen our resilience against climate disasters. .")

st.sidebar.info("""Please note that this dashboard is a prototype. 
                Users are advised that the tool may contain errors, 
                bugs, or limitations and should be used with caution 
                and awareness of potential risks, and the developers 
                make no warranties or guarantees regarding its performance, 
                reliability, or suitability for any specific purpose.""")

with st.sidebar:
        st.caption('Developed by Hot Issue')
        col1, col2, col3, col4 = st.columns([0.05,0.05,0.05,0.15])
        with col1:
            st.markdown("""<a href="https://github.com/yumoldianne/QUEST">
                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Octicons-mark-github.svg/1200px-Octicons-mark-github.svg.png?20180806170715" 
                width="30" height="30"></a>""", unsafe_allow_html=True)
        #with col2:
            #st.markdown("""<a href="https://docs.google.com/document/d/1eu39rT-Zh6KhUNwXrAzOwdOJv0_lz3IDv4X_GEmaDsA/edit?usp=sharing">
               #<img src="https://cdn-icons-png.flaticon.com/512/482/482202.png" 
               # #width="30" height="30"></a>""", unsafe_allow_html=True)
        #with col3:
            #st.markdown("""<a href="https://www.instagram.com/eltgnd_v/">
                #<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Instagram_logo_2016.svg/2048px-Instagram_logo_2016.svg.png" 
                #width="30" height="30"></a>""", unsafe_allow_html=True)

#Routing Algorithm
st.subheader("Routing Algorithm")

st.write("""
        The CSR for MSMEs was developed by analyzing various financial and socioeconomic indicators. Transactional and behavioral data were first consolidated into quarterly sums to smooth out short-term fluctuations. Categorical variables were then converted to numeric values through appropriate mapping techniques. After preprocessing the features, correlation-based feature selection was performed to enhance model efficiency, removing variables with correlation coefficients exceeding 0.8 to eliminate redundancy while preserving the most informative indicators.
         
         The framework categorized indicators into four fundamental concepts: `Financial Health`, `Credit Reliability`, `Customer Engagement`, and `Socioeconomic Stability`. Within each concept, the mean of the features was taken and then standardized using z-score transformation to ensure comparability across different measures. These standardized features were then combined through mean aggregation to create composite scores that captured the overall strength of each conceptual dimension.
         
         The final resilience score was computed by taking the average of the four concept-specific composite scores, ensuring equal weight distribution across all dimensions. To enhance interpretability, the resulting score was scaled to a range of 0 to 1 using min-max scaling. This standardized approach produced a robust metric that effectively captures multiple aspects of MSME resilience while maintaining simplicity and clarity in its interpretation.

         """)

st.title("🔥 About Hot Issue")

st.write("""
         The team comprises of Juliana Ambrosio (5 BS MIS), Caitlyn Lee (M DSc), Jan Manzano (4 BS MIS), Andrea Senson (BS AMDSc '25), and Dianne Yumol (M DSc).
         """)