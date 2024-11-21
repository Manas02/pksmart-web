import streamlit as st

st.set_page_config(
    page_title="PKSmart",
    page_icon="logo_icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

left_col, right_col = st.columns(2)

right_col.write("# Welcome to PKSmart")
right_col.write("v3.0")
right_col.write("Created by Srijit Seal and Andreas Bender")
left_col.image("logo_front.png")


st.sidebar.success("")
# Add "Run PKSmart" button in the sidebar
if st.sidebar.button("Run PKSmart"):
    st.query_params(page="test")

st.markdown(
"""
    PKSmart is an open-source app framework built specifically for
    predicting human and animal PK parameters.
    Select from the sidebar to predict single molecule PK properties or submit a bulk job!
    
    ### Want to learn more?
    - Check out our paper at [bioarxiv](https://www.biorxiv.org/content/10.1101/2024.02.02.578658v1)
    """
)
st.markdown("---")

left_col, right_col = st.columns(2)

right_col.image("logo_bottom.png")

left_col.markdown(
        """
        ### Usage
        On the left pane is the main menu for navigating to 
        the following pages in the PK Predictor application:
        - **Home Page:** You are here!
        - **Submit single SMILES:** You can enter the smiles of the query compound here to obtain a detailed analysis of predicted human PK parameters with the chemical space comparisons to training data and also the predicted animal PK parameters used by the model.
        - **Submit batch SMILES:** You can dowload the predicted human and animal parameters for a batch of smiles that you can type while seperating by a comma (,).
        """
    )
st.markdown("---")


left_info_col, right_info_col = st.columns(2)

left_info_col.markdown(

        f"""
        ### Authors
        
        ##### Srijit Seal 
        - Email:  <seal@broadinstitute.org>
        - GitHub: https://github.com/srijitseal
        ##### Andreas Bender 
        - Email: <ab454@cam.ac.uk>
        """,
        unsafe_allow_html=True,
    )

right_info_col.markdown(
        """
        ### Funding
        - Cambridge Centre for Data Driven Discovery and Accelerate Programme for Scientific Discovery under the project title “Theoretical, Scientific, and Philosophical Perspectives on Biological Understanding in the Age of Artificial Intelligence”, made possible by a donation from Schmidt Futures
        - Cambridge Commonwealth, European and International Trust
        - National Institutes of Health (NIH MIRA R35 GM122547 to Anne E Carpenter) 
        - Massachusetts Life Sciences Center Bits to Bytes Capital Call program for funding the data analysis (to Shantanu Singh, Broad Institute of MIT and Harvard)
        - OASIS Consortium organised by HESI (OASIS to Shantanu Singh, Broad Institute of MIT and Harvard)
        - Boak Student Support Fund (Clare Hall)
        - Jawaharlal Nehru Memorial Fund
        - Allen, Meek and Read Fund
        - Trinity Henry Barlow (Trinity College)
         """
    )

right_info_col.markdown(
        """
        ### License
        Apache License 2.0
        """
    )