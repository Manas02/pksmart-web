import pickle
import numpy as np
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from pksmart import (standardize, 
                     calcdesc, 
                     predict_animal, 
                     predict_VDss, 
                     predict_CL, 
                     predict_fup, 
                     predict_MRT, 
                     predict_thalf, 
                     avg_kNN_similarity,
                     run_pca,
                     mol2svg,
                     plot_ad,
                     check_if_in_training_data)
from bokeh.plotting import ColumnDataSource, figure


##########

st.set_page_config(
    page_title="PKSmart",
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
    "Enter SMILES:",
    help="You can enter one SMILES"
)
try:
    input_smiles_in_df = smiles_input.strip()
    smiles_list = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles_input.strip()))]
except:
    st.error("Invalid SMILES entered. Please check the input and try again.")

# Check if the user has provided input
if st.button("Run PKSmart"):
    if smiles_list:
        with st.spinner('Running PKSmart ...'): 
            # Create an empty DataFrame to hold the SMILES and predictions
            data = pd.DataFrame(smiles_list, columns=['smiles_r'])

            # Standardize and calculate descriptors for the input molecules
            data['standardized_smiles'] = data['smiles_r'].apply(standardize)

            # Check if standardization failed
            if "Cannot_do" in data['standardized_smiles'].values:
                st.error("Invalid SMILES entered. Please check the input and try again.")
            else:
                data_mordred = calcdesc(data)
                ts_data = avg_kNN_similarity(data)

                check_if_in_training_data(data['standardized_smiles'].values[0])

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
                display_predictions['input_smiles'] = input_smiles_in_df
                for key in animal_columns:
                    if not key.endswith("_fup"):
                        display_predictions[key] = 10**display_predictions[key]

                # Run predictions for human models
                
                # human_columns = ['VDss', 'CL', 'fup', 'MRT', 'thalf']
                human_predictions = pd.DataFrame()

                with open("../data/features_mfp_mordred_animal_artificial_human_modelcolumns.txt") as f:
                    model_features = f.read().splitlines()

                data_mordred['input_smiles'] = input_smiles_in_df
                human_predictions['input_smiles'] = data_mordred['input_smiles']
                human_predictions['smiles_r'] = data_mordred['smiles_r']

                human_predictions['VDss'] = 10**predict_VDss(data_mordred, model_features)
                Vd_Tc = ts_data["human_VDss_L_kg"]
                with open(f"../data/folderror_human_VDss_L_kg_generator.sav", 'rb') as f:
                    loaded = pickle.load(f)
                    human_predictions['Vd_fe'] = loaded.predict(Vd_Tc.values.reshape(-1, 1)).round(2)
                human_predictions['Vd_min'] = human_predictions['VDss'] / human_predictions['Vd_fe']
                human_predictions['Vd_max'] = human_predictions['VDss'] * human_predictions['Vd_fe']
                alert_message = f"Alert: This Molecule May Be Out Of AD for VDss (<{threshold} Tc with Training data)"
                if ts_data['human_VDss_L_kg'].values[0] < threshold:
                    st.markdown(f"<span style='color:red;'>Alert: This Molecule May Be Out Of AD for VDss (<{threshold} Tc with Training data)</span>", unsafe_allow_html=True)
                human_predictions['comments'] = ts_data['human_VDss_L_kg'].apply(
                    lambda x: f"{alert_message}" if x < threshold else ""
                )

                human_predictions['CL'] = 10**predict_CL(data_mordred, model_features)
                CL_Tc =  ts_data["human_CL_mL_min_kg"]
                with open(f"../data/folderror_human_CL_mL_min_kg_generator.sav", 'rb') as f:    
                    loaded = pickle.load(f)
                    human_predictions["CL_fe"]= loaded.predict(CL_Tc.values.reshape(-1, 1)).round(2)
                human_predictions['CL_min'] = human_predictions['CL'] / human_predictions['CL_fe']
                human_predictions['CL_max'] = human_predictions['CL'] * human_predictions['CL_fe']
                alert_message = f"Alert: This Molecule May Be Out Of AD for CL (<{threshold} Tc with Training data)"
                if ts_data['human_CL_mL_min_kg'].values[0] < threshold:
                    st.markdown(f"<span style='color:red;'>Alert: This Molecule May Be Out Of AD for CL (<{threshold} Tc with Training data)</span>", unsafe_allow_html=True)
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
                if ts_data['human_fup'].values[0] < threshold:
                    st.markdown(f"<span style='color:red;'>Alert: This Molecule May Be Out Of AD for fup (<{threshold} Tc with Training data)</span>", unsafe_allow_html=True)
                human_predictions['comments'] = human_predictions['comments'] + ts_data['human_fup'].apply(
                    lambda x: f"\n{alert_message}" if x < threshold else ""
                )

                human_predictions['MRT'] = 10**predict_MRT(data_mordred, model_features)
                MRT_Tc =  ts_data["human_mrt"]
                with open(f"../data/folderror_human_mrt_generator.sav", 'rb') as f:
                    loaded = pickle.load(f)
                    human_predictions["MRT_fe"]= loaded.predict(MRT_Tc.values.reshape(-1, 1)).round(2)
                human_predictions['MRT_min'] = human_predictions['MRT'] / human_predictions['MRT_fe']
                human_predictions['MRT_max'] = human_predictions['MRT'] * human_predictions['MRT_fe']
                alert_message = f"Alert: This Molecule May Be Out Of AD for MRT (<{threshold} Tc with Training data)"
                if ts_data['human_mrt'].values[0] < threshold:
                    st.markdown(f"<span style='color:red;'>Alert: This Molecule May Be Out Of AD for MRT (<{threshold} Tc with Training data)</span>", unsafe_allow_html=True)
                human_predictions['comments'] = human_predictions['comments'] + ts_data['human_mrt'].apply(
                    lambda x: f"\n{alert_message}" if x < threshold else ""
                )

                human_predictions['thalf'] = 10**predict_thalf(data_mordred, model_features)
                thalf_Tc =  ts_data["human_thalf"]
                with open(f"../data/folderror_human_thalf_generator.sav", 'rb') as f:
                    loaded = pickle.load(f)
                    human_predictions["thalf_fe"]= loaded.predict(thalf_Tc.values.reshape(-1, 1)).round(2)
                human_predictions['thalf_min'] = human_predictions['thalf'] / human_predictions['thalf_fe']
                human_predictions['thalf_max'] = human_predictions['thalf'] * human_predictions['thalf_fe']
                alert_message = f"Alert: This Molecule May Be Out Of AD for thalf (<{threshold} Tc with Training data)"
                if ts_data['human_thalf'].values[0] < threshold:
                    st.markdown(f"<span style='color:red;'>Alert: This Molecule May Be Out Of AD for thalf (<{threshold} Tc with Training data)</span>", unsafe_allow_html=True)
                human_predictions['comments'] = human_predictions['comments'] + ts_data['human_thalf'].apply(
                    lambda x: f"\n{alert_message}" if x < threshold else ""
                )

                human_predictions = human_predictions[[col for col in human_predictions if col != 'comments'] + ['comments']]

                combined_predictions = pd.merge(human_predictions, display_predictions[animal_columns + ['smiles_r']], on='smiles_r')
                
        #########################################################################
                
                pca_res= []
                data_train=dict()
                human_predictions = human_predictions.round(2)
                smiles_r = human_predictions['smiles_r'].values[0]
                CL_fe = human_predictions["CL_fe"].values[0]
                CL = human_predictions['CL'].values[0]
                Vd_fe = human_predictions['Vd_fe'].values[0]
                Vd = human_predictions['VDss'].values[0]
                MRT_fe = human_predictions['MRT_fe'].values[0]
                MRT = human_predictions['MRT'].values[0]
                thalf_fe = human_predictions['thalf_fe'].values[0]
                thalf = human_predictions['thalf'].values[0]
                fup_fe = human_predictions['fup_fe'].values[0]
                fup = human_predictions['fup'].values[0]
                pca_res, pcv_1, pcv_2 = run_pca(data_mordred)
                Vd_range = [human_predictions['Vd_min'].values[0], human_predictions['Vd_max'].values[0]]
                CL_range =  [human_predictions['CL_min'].values[0], human_predictions['CL_max'].values[0]]
                fup_range = [human_predictions['fup_min'].values[0], human_predictions['fup_max'].values[0]]
                MRT_range = [human_predictions['MRT_min'].values[0], human_predictions['MRT_max'].values[0]]
                thalf_range = [human_predictions['thalf_min'].values[0], human_predictions['thalf_max'].values[0]]

                molecule = Chem.MolFromSmiles(smiles_r)
                st.image(Draw.MolToImage(molecule), width=200)
                st.write(f"Standardised Query compound: {smiles_r}")

        ########################### copy pasta

                preds_dict = {'Endpoint Predicted':['Clearance (CL)', 
                'Volume of distribution (VDss)', 
                "Fraction unbound in plasma (fup)", 
                "Mean Residence Time (MRT)", 
                "Half-life (thalf)"],

                'Predicted value':[f'{CL} mL/min/kg', f'{Vd} L/kg', f'{fup}', f'{MRT} h', f'{thalf} h'],
                'Predicted Fold error':[CL_fe, Vd_fe, fup_fe, MRT_fe, thalf_fe],
                'Predicted Range':[ f'{np.round(CL/CL_fe,2)} to {np.round(CL*CL_fe, 2)} mL/min/kg', 
                f'{np.round(Vd/Vd_fe,2)} to {np.round(Vd*Vd_fe, 2)} L/kg', 
                f'{np.round(fup/fup_fe,2)} to {np.round(fup*fup_fe, 2)}', 
                f'{np.round(MRT/MRT_fe,2)} to {np.round(MRT*MRT_fe, 2)} h', 
                f'{np.round(thalf/thalf_fe,2)} to {np.round(thalf*thalf_fe, 2)} h', ]
                }

                preds_dict_download = {

                'smiles_r':[smiles_r],

                'Clearance_(CL)':[CL],
                'Clearance_(CL)_units': ["mL/min/kg"],
                'Clearance_(CL)_folderror': [CL_fe],
                'Clearance_(CL)_upperbound': np.max(CL_range),
                'Clearance_(CL)_lowerbound': np.min(CL_range),

                'Volume_of_distribution_(VDss)':[Vd],
                'Volume_of_distribution_(VDss)_units': ["L/kg"],
                'Volume_of_distribution_(VDss)_folderror': [Vd_fe],
                'Volume_of_distribution_(VDss)_upperbound': np.max(Vd_range),
                'Volume_of_distribution_(VDss)_lowerbound': np.min(Vd_range),

                'Fraction_unbound_in_plasma_(fup)':[fup],
                'Fraction_unbound_in_plasma_(fup)_units': ["dimensionless"],
                'Fraction_unbound_in_plasma_(fup)_folderror': [fup_fe],
                'Fraction_unbound_in_plasma_(fup)_upperbound': np.max(fup_range),
                'Fraction_unbound_in_plasma_(fup)_lowerbound': np.min(fup_range),

                'Mean_Residence_Time_(MRT)':[MRT],
                'Mean_Residence_Time_(MRT)_units': ["h"],
                'Mean_Residence_Time_(MRT)_folderror': [MRT_fe],
                'Mean_Residence_Time_(MRT)_upperbound': np.max(MRT_range),
                'Mean_Residence_Time_(MRT)_lowerbound': np.min(MRT_range),

                'Half_life_(thalf)':[thalf],
                'Half_life_(thalf)_units': ["h"],
                'Half_life_(thalf)_folderror': [thalf_fe],
                'Half_life_(thalf)_upperbound': np.max(thalf_range),
                'Half_life_(thalf)_lowerbound': np.min(thalf_range),

                }

                preds = pd.DataFrame(preds_dict)
                st.table(preds)
                
                preds_dict_download = pd.DataFrame(preds_dict_download)

                #interactive plot
                
                molsvgs_train = pickle.load(open("../data/molsvgs_train.sav", 'rb'))
                pca_res = np.array(pca_res)

                train_data_features= pd.read_csv("../data/Train_data_features.csv") 
                train_data_features["Data"] = "Train"

                human_VDss_L_kg=10**train_data_features[:]['human_VDss_L_kg']
                human_CL_mL_min_kg=10**train_data_features[:]['human_CL_mL_min_kg']
                human_fup=10**train_data_features[:]['human_fup']
                human_mrt=10**train_data_features[:]['human_mrt']
                human_thalf=10**train_data_features[:]['human_thalf']

                file = open("../data/features_mfp_mordred_columns_human.txt", "r")
                file_lines = file.read()
                features_mfp_mordred_columns = file_lines.split("\n")
                features_mfp_mordred_columns = features_mfp_mordred_columns[:-1]
            
                mols_test = Chem.MolFromSmiles(smiles_r)
                molsvgs_test = [mol2svg(mols_test)]

                data_train = dict(
                    x= pca_res[:-1][:,0],
                    y=pca_res[:-1][:,1],
                    img = molsvgs_train,
                    human_VDss_L_kg=human_VDss_L_kg,
                    human_CL_mL_min_kg=human_CL_mL_min_kg,
                    human_fup=human_fup,
                    human_mrt=human_mrt,
                    human_thalf=human_thalf
                    )

                data_test = dict(
                
                    x= pca_res[-1:][:,0],
                    y=pca_res[-1:][:,1],
                    img = molsvgs_test[-1:],
                    human_CL_mL_min_kg=[np.round(CL,2)],
                )


                source_train = ColumnDataSource(data_train)
                source_test= ColumnDataSource(data_test)

                TOOLTIPS = """
                <div>
                human_VDss_L_kg: @human_VDss_L_kg<br>
                human_CL_mL_min_kg: @human_CL_mL_min_kg<br>
                human_fup: @human_fup<br>
                human_mrt: @human_mrt<br>
                human_thalf: @human_thalf<br>
                @img{safe}
                </div>
                """

                st.write("Projecting query compound in the structural-physicochemical space of the training data:")

                p = figure(width=600, height=600, tooltips=TOOLTIPS,
                        title=f"Principal Component Analysis using selected Mordred descriptors ({np.round(pcv_1+pcv_2, 2)}% variance explained)",
                        x_axis_label=f"Principal Component 1 ({np.round(pcv_1, 2)}% explained variance)",
                        y_axis_label=f"Principal Component 2 ({np.round(pcv_2, 2)}% explained variance)")
                
                p.circle('x', 'y', size=10, source=source_train, color="red", legend_label="Training data")
                p.circle('x', 'y', size=10, source=source_test, color="blue", legend_label="Query Compound")
                
                p.legend.location = "top_left"

                # change border and background of legend
                p.legend.border_line_width = 3
                p.legend.border_line_color = "black"


                st.bokeh_chart(p, use_container_width=True)

                plot_ad(Vd_Tc, CL_Tc, fup_Tc, MRT_Tc ,thalf_Tc)

        #########################################################################
                # Optionally, download results as CSV
                st.subheader("Download Results")

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
                csv = combined_predictions.to_csv(index=False)
                st.download_button("Download Results as CSV", data=csv, file_name='pksmart_predictions.csv', mime='text/csv')

                # Display the human predictions in a table
                st.subheader("Human Pharmacokinetics Predictions")
                st.dataframe(human_predictions.round(2).head(), hide_index=True)

                st.subheader("Animal Pharmacokinetics Predictions")
                st.dataframe(display_predictions[['input_smiles', 'smiles_r'] + animal_columns].round(2).head(), hide_index=True)

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
        - Boak Student Support Fund (Clare Hall)
        - National Institutes of Health (NIH MIRA R35 GM122547 to Anne E Carpenter) 
        - Massachusetts Life Sciences Center Bits to Bytes Capital Call program for funding the data analysis (to Shantanu Singh, Broad Institute of MIT and Harvard)
        - OASIS Consortium organised by HESI (OASIS to Shantanu Singh, Broad Institute of MIT and Harvard)
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
