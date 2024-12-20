import pickle
import numpy as np
from rdkit import Chem
import streamlit as st
import pandas as pd
from pksmart import (check_if_in_training_data, 
                     standardize, 
                     calcdesc, 
                     predict_animal, 
                     predict_VDss, 
                     predict_CL, 
                     predict_fup, 
                     predict_MRT, 
                     predict_thalf, 
                     avg_kNN_similarity)


##########

st.set_page_config(
    page_title="PKSmart Batch Run",
    page_icon="logo_icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

left_col, right_col = st.columns(2)

right_col.write("# Welcome to PKSmart")
right_col.write("v3.1")
right_col.write("Created by Srijit Seal and Andreas Bender")
left_col.image("logo_front.png")

left_col, right_col = st.columns(2)
left_col.markdown(
"""
    PKSmart is an open-source app framework built specifically for
    predicting human and animal PK parameters.
    Select from the sidebar to predict single molecule PK properties or submit a bulk job!
    
    ### Want to learn more?
    - Check out our paper at [bioarxiv](https://www.biorxiv.org/content/10.1101/2024.02.02.578658v1)
    """
)
right_col.image("logo_bottom.png")

st.markdown("---")

##########
st.markdown("## Predict using PKSmart")

# Define the threshold and the alert message
threshold = 0.25

# Input area for SMILES
smiles_input = st.text_area(
    "Enter SMILES (single or comma-separated):",
    help="You can enter one or more SMILES strings, separated by commas."
)

uploaded_file = st.file_uploader("Or upload a newline-delimited SMILES file", type=["smi", "txt"])

smiles_list = []

# Process the file upload if available
if uploaded_file is not None:
    smiles_list = uploaded_file.getvalue().decode("utf-8").splitlines()
    smiles_list = [smile.strip() for smile in smiles_list if smile.strip()]
elif smiles_input:
    # Process the input to handle comma-separated SMILES
    smiles_list = [smile.strip() for smile in smiles_input.split(",") if smile.strip()]

smiles_list_valid = []
smiles_list_valid_input = []
for smile in smiles_list:
    if Chem.MolFromSmiles(smile):
        smiles_list_valid.append(Chem.MolToSmiles(Chem.MolFromSmiles(smile)))
        smiles_list_valid_input.append(smile)
    else:
        st.error(f"Invalid SMILES: {smile}. Skipping this molecule.")

@st.cache_data
def run_pksmart(smiles_list, smiles_list_valid_input):
    # Create an empty DataFrame to hold the SMILES and predictions
    data = pd.DataFrame(smiles_list, columns=['smiles_r'])

    # Standardize and calculate descriptors for the input molecules
    data['input_smiles'] = smiles_list_valid_input
    data['standardized_smiles'] = data['smiles_r'].apply(standardize)
    
    # Identify invalid SMILES
    invalid_smiles = data[data['standardized_smiles'] == "Cannot_do"]
    
    # Display invalid SMILES in the Streamlit app
    if not invalid_smiles.empty:

        invalid_smiles = invalid_smiles.rename(columns={"smiles_r":"input smiles"})
        st.write("One or more invalid SMILES:")
        st.write(invalid_smiles)
    
    data = data[data['standardized_smiles'] != "Cannot_do"].reset_index(drop = True)

    st.write("Predictions for valid SMILES below:")
    
    for smiles in data["standardized_smiles"].values:
        check_if_in_training_data(smiles)

    data_mordred = calcdesc(data)
    ts_data = avg_kNN_similarity(data)

    # Run predictions for animal models
    animal_predictions = predict_animal(data_mordred)

    # Filter out only the relevant animal PK columns
    animal_columns = [
        "dog_VDss_L_kg", "dog_CL_mL_min_kg", "dog_fup",
        "monkey_VDss_L_kg", "monkey_CL_mL_min_kg", "monkey_fup",
        "rat_VDss_L_kg", "rat_CL_mL_min_kg", "rat_fup"
    ]
    
    # Create a copy of animal_predictions to avoid modifying the original
    display_predictions = animal_predictions.copy()
    for key in animal_columns:
        if not key.endswith("_fup"):
            display_predictions[key] = 10**display_predictions[key]

    # Run predictions for human models
    st.subheader("Human Pharmacokinetics Predictions")
    # human_columns = ['VDss', 'CL', 'fup', 'MRT', 'thalf']
    human_predictions = pd.DataFrame()

    with open("../data/features_mfp_mordred_animal_artificial_human_modelcolumns.txt") as f:
        model_features = f.read().splitlines()

    human_predictions['smiles_r'] = data_mordred['smiles_r']

    human_predictions['VDss_L_kg'] = 10**predict_VDss(data_mordred, model_features)
    Vd_Tc = ts_data["human_VDss_L_kg"]
    with open(f"../data/folderror_human_VDss_L_kg_generator.sav", 'rb') as f:
        loaded = pickle.load(f)
        human_predictions['Vd_fe'] = loaded.predict(Vd_Tc.values.reshape(-1, 1)).round(2)
    human_predictions['Vd_min'] = human_predictions['VDss_L_kg'] / human_predictions['Vd_fe']
    human_predictions['Vd_max'] = human_predictions['VDss_L_kg'] * human_predictions['Vd_fe']
    alert_message = f"Alert: This Molecule May Be Out Of AD for VDss (<{threshold} Tc with Training data)"
    human_predictions['comments'] = ts_data['human_VDss_L_kg'].apply(
        lambda x: f"{alert_message}" if x < threshold else ""
    )

    human_predictions['CL_mL_min_kg'] = 10**predict_CL(data_mordred, model_features)
    CL_Tc =  ts_data["human_CL_mL_min_kg"]
    with open(f"../data/folderror_human_CL_mL_min_kg_generator.sav", 'rb') as f:    
        loaded = pickle.load(f)
        human_predictions["CL_fe"]= loaded.predict(CL_Tc.values.reshape(-1, 1)).round(2)
    human_predictions['CL_min'] = human_predictions['CL_mL_min_kg'] / human_predictions['CL_fe']
    human_predictions['CL_max'] = human_predictions['CL_mL_min_kg'] * human_predictions['CL_fe']
    alert_message = f"Alert: This Molecule May Be Out Of AD for CL (<{threshold} Tc with Training data)"
    human_predictions['comments'] = human_predictions['comments'] + ts_data['human_CL_mL_min_kg'].apply(
        lambda x: f"\n{alert_message}" if x < threshold else ""
    )

    human_predictions['fup'] = predict_fup(data_mordred, model_features)
    fup_Tc =  ts_data["human_fup"]
    with open(f"../data/folderror_human_fup_generator.sav", 'rb') as f:
        loaded = pickle.load(f)
        human_predictions["fup_fe"]= loaded.predict(fup_Tc.values.reshape(-1, 1)).round(2)
    human_predictions['fup_min'] = human_predictions['fup'] / human_predictions['fup_fe']
    human_predictions['fup_max'] = human_predictions['fup'] * human_predictions['fup_fe'].clip(upper=1)
    alert_message = f"Alert: This Molecule May Be Out Of AD for fup (<{threshold} Tc with Training data)"

    human_predictions['comments'] = human_predictions['comments'] + ts_data['human_fup'].apply(
        lambda x: f"\n{alert_message}" if x < threshold else ""
    )

    human_predictions['MRT_hr'] = 10**predict_MRT(data_mordred, model_features)
    MRT_Tc =  ts_data["human_mrt"]
    with open(f"../data/folderror_human_mrt_generator.sav", 'rb') as f:
        loaded = pickle.load(f)
        human_predictions["MRT_fe"]= loaded.predict(MRT_Tc.values.reshape(-1, 1)).round(2)
    human_predictions['MRT_min'] = human_predictions['MRT_hr'] / human_predictions['MRT_fe']
    human_predictions['MRT_max'] = human_predictions['MRT_hr'] * human_predictions['MRT_fe']
    alert_message = f"Alert: This Molecule May Be Out Of AD for MRT (<{threshold} Tc with Training data)"

    human_predictions['comments'] = human_predictions['comments'] + ts_data['human_mrt'].apply(
        lambda x: f"\n{alert_message}" if x < threshold else ""
    )

    human_predictions['thalf_hr'] = 10**predict_thalf(data_mordred, model_features)
    thalf_Tc =  ts_data["human_thalf"]
    with open(f"../data/folderror_human_thalf_generator.sav", 'rb') as f:
        loaded = pickle.load(f)
        human_predictions["thalf_fe"]= loaded.predict(thalf_Tc.values.reshape(-1, 1)).round(2)
    human_predictions['thalf_min'] = human_predictions['thalf_hr'] / human_predictions['thalf_fe']
    human_predictions['thalf_max'] = human_predictions['thalf_hr'] * human_predictions['thalf_fe']
    alert_message = f"Alert: This Molecule May Be Out Of AD for thalf (<{threshold} Tc with Training data)"

    human_predictions['comments'] = human_predictions['comments'] + ts_data['human_thalf'].apply(
        lambda x: f"\n{alert_message}" if x < threshold else ""
    )

    human_predictions = human_predictions[[col for col in human_predictions if col != 'comments'] + ['comments']]
    human_predictions['input_smiles'] = smiles_list_valid_input
    # place input smiles at the beginning
    human_predictions = human_predictions[['input_smiles'] + [col for col in human_predictions if col != 'input_smiles']]
    # Display the human predictions in a table
    st.dataframe(human_predictions.round(2).head(), hide_index=True)

    st.subheader("Animal Pharmacokinetics Predictions")
    st.dataframe(display_predictions[['input_smiles', 'smiles_r'] + animal_columns].round(2).head(), hide_index=True)

    combined_predictions = pd.merge(human_predictions, display_predictions[animal_columns + ['smiles_r']], on='smiles_r')

    return combined_predictions

# Check if the user has provided input
if st.button("Run PKSmart"):
    if smiles_list_valid:
        combined_predictions = run_pksmart(smiles_list_valid, smiles_list_valid_input)

        column_mapping = {
            "VDss": "Volume_of_distribution_(VDss)(L/kg)",
            "Vd_fe": "Volume_of_distribution_(VDss)_folderror",
            "Vd_min": "Volume_of_distribution_(VDss)_lowerbound",
            "Vd_max": "Volume_of_distribution_(VDss)_upperbound",
            "CL": "Clearance_(CL)_(mL/min/kg)",
            "CL_fe": "Clearance_(CL)_folderror",
            "CL_min": "Clearance_(CL)_lowerbound",
            "CL_max": "Clearance_(CL)_upperbound",
            "fup": "Fraction_unbound_in_plasma_(fup)",
            "fup_fe": "Fraction_unbound_in_plasma_(fup)_folderror",
            "fup_min": "Fraction_unbound_in_plasma_(fup)_lowerbound",
            "fup_max": "Fraction_unbound_in_plasma_(fup)_upperbound",
            "MRT": "Mean_Residence_Time_(MRT)(h)",
            "MRT_fe": "Mean_Residence_Time_(MRT)_folderror",
            "MRT_min": "Mean_Residence_Time_(MRT)_lowerbound",
            "MRT_max": "Mean_Residence_Time_(MRT)_upperbound",
            "thalf": "Half_life_(thalf)(h)",
            "thalf_fe": "Half_life_(thalf)_folderror",
            "thalf_min": "Half_life_(thalf)_lowerbound",
            "thalf_max": "Half_life_(thalf)_upperbound"
        }
        combined_predictions.rename(columns=column_mapping, inplace=True)
        combined_predictions['input_smiles'] = smiles_list_valid_input

        # Optionally, download results as CSV
        st.subheader("Download Results")

        csv = combined_predictions.to_csv(index=False)
        st.download_button("Download All Results as CSV", data=csv, file_name='pksmart_predictions.csv', mime='text/csv')


#################

st.markdown("---")


left_info_col, right_info_col = st.columns(2)

left_info_col.markdown(

        f"""
        ### Authors
        
        ##### Srijit Seal 
        - Website:  https://srijitseal.com
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
        | Dev: Manas Mahale
        """
    )
# Values and definitions
definitions = {
    # Animal Parameters
    "dog_VDss_L_kg": "Volume of distribution at steady state for dog (L/kg)",
    "dog_CL_mL_min_kg": "Clearance rate for dog (mL/min/kg)",
    "dog_fup": "Fraction unbound in plasma for dog",
    "monkey_VDss_L_kg": "Volume of distribution at steady state for monkey (L/kg)",
    "monkey_CL_mL_min_kg": "Clearance rate for monkey (mL/min/kg)",
    "monkey_fup": "Fraction unbound in plasma for monkey",
    "rat_VDss_L_kg": "Volume of distribution at steady state for rat (L/kg)",
    "rat_CL_mL_min_kg": "Clearance rate for rat (mL/min/kg)",
    "rat_fup": "Fraction unbound in plasma for rat",
    
    # Human Parameters
    "CL_fe": "Fold error for human clearance (CL)",
    "CL": "Predicted value for human clearance (CL, mL/min/kg)",
    "CL_min": "Minimum predicted value for human clearance (CL, mL/min/kg)",
    "CL_max": "Maximum predicted value for human clearance (CL, mL/min/kg)",
    "Vd_fe": "Fold error for human volume of distribution (VDss)",
    "VDss": "Predicted value for human volume of distribution (VDss, L/kg)",
    "Vd_min": "Minimum predicted value for human volume of distribution (VDss, L/kg)",
    "Vd_max": "Maximum predicted value for human volume of distribution (VDss, L/kg)",
    "MRT_fe": "Fold error for human mean residence time (MRT)",
    "MRT": "Predicted value for human mean residence time (MRT, min)",
    "MRT_min": "Minimum predicted value for human mean residence time (MRT, min)",
    "MRT_max": "Maximum predicted value for human mean residence time (MRT, min)",
    "thalf_fe": "Fold error for human half-life (t1/2)",
    "thalf": "Predicted value for human half-life (t1/2, min)",
    "thalf_min": "Minimum predicted value for human half-life (t1/2, min)",
    "thalf_max": "Maximum predicted value for human half-life (t1/2, min)",
    "fup_fe": "Fold error for human fraction unbound in plasma (fup)",
    "fup": "Predicted value for human fraction unbound in plasma (fup)",
    "fup_min": "Minimum predicted value for human fraction unbound in plasma (fup)",
    "fup_max": "Maximum predicted value for human fraction unbound in plasma (fup)"
}


# Display definitions
st.markdown("### Definitions")
for key, description in definitions.items():
    st.write(f"**{key}**: {description}")
