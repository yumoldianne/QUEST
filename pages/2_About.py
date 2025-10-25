import streamlit as st

st.set_page_config(
    page_title="About QUEST",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚨 What is QUEST?")

st.info("💡 **QUEST: Quezon City Evacuation Support Tool** is designed to be a dashboard that provides data during times of typhoon- and rainfall-induced floods on which relief operations centers to seek out, the optimal route to travel to these centers in order to maximize preparedness and strengthen our resilience against climate disasters.")

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

#Routing Algorithm
st.subheader("Flood Evacuation Routing Algorithm")

st.write("""
        Under the hood: we load flood polygons (EPSG:32651), create a smart grid over the affected area, assign per-node risk with batched spatial joins, then build a graph connecting neighboring grid points. Each edge gets a distance and a risk-weighted cost. For routing we run A* with a Euclidean heuristic. A single slider controls the risk penalty factor: 1.0 = fastest, >1 = progressively safer routes. This is soft avoidance — we penalize risky edges instead of forbidding them, which keeps routes feasible while prioritizing safety.
         
         """)

#Accessibility Hueristic
st.subheader("Relief Centers Accessibility Hueristic")

st.write("""
        We measure accessibility around each relief center by buffering the center (default 200 m), seeing how much of that buffer overlaps high-risk flood polygons (Var == 3), and converting that overlap fraction into an accessibility score: accessibility = max(0, 1 − (area_of_high_risk_within_buffer / buffer_area))

        Therefore, a score of 1.0 = fully accessible (no high-risk area in the buffer) while a score of 0.0 = fully surrounded by high risk.         
         """)

st.title("🔥 About Hot Issue")

st.write("""
         The team comprises of uliana Ambrosio (BS MIS '26), Caitlyn Lee (BS AMDSc '25, M DSc '26), Jan Manzano (BS MIS '26) , Andrea Senson (BS AMDSc '25), and Dianne Yumol (BS AMDSc '25, M DSc '26).
         """)