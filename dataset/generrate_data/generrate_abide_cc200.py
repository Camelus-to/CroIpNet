import os
import numpy as np
import pandas as pd
from scipy.io import loadmat


# FIQ: Full-scale IQ, the total IQ score calculated through psychological testing.
#
# VIQ: Verbal IQ, the IQ score measuring the subject's language ability.
#
# PIQ: Performance IQ, the IQ score measuring the subject's non-verbal ability.
# Process dynamic connectivity matrix files
def process_dynamic_connectivity_files(input_dir, dynamic_matrix_dir, output_dir, df):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    signals_path_yuan = r"E:\Technolgy_learning\Learning_code\New_Stage\shujuji\ABIDE_I\cc200\signals_quan"
    # Initialize lists to store the final output content
    FC_list = []
    FC_dynamic_list = []
    Label_list = []
    Site_list = []
    Filename_list = []
    sex_list = []
    Signals_list = []
    age_list = []
    FIQ_list = []
    VIQ_list = []
    PIQ_list = []
    # Iterate over each sample file in the dynamic connectivity matrix folder
    site_onehot = pd.get_dummies(df['SITE_ID']).astype(int)  # Create one-hot encoding
    for filename in os.listdir(dynamic_matrix_dir):
        if filename.endswith('.mat'):
            # Construct file path and load data
            static_file_path = os.path.join(input_dir, filename)
            static_matrix_dir = loadmat(static_file_path)
            static_matrix_dir = static_matrix_dir['fc_matrix']
            signals_path = os.path.join(signals_path_yuan, 'ROISignals_00' + filename)
            signals = loadmat(signals_path)['ROISignals'][:145, 228:428].T  # The format of the signal should be region*signal length
            # Sample name (remove file extension)
            sample_id = filename[:-4]
            df['SUB_ID'] = df['SUB_ID'].astype(str).str.strip()  # Ensure SUB_ID is of string type and remove spaces
            # Find the classification label and site information of the sample
            match = df[df['SUB_ID'] == str(sample_id)]
            if len(match) == 0:
                print(f"No match found for {sample_id}")
                continue

            # Get classification label. The above is randomly generated between 0 and 1, the following is obtained from the table
            # y = np.random.randint(0, 2)  # 0 或 1
            y = match.iloc[0]['DX_GROUP']
            site_id = match.iloc[0]['SITE_ID']  # Get site ID

            # Convert site information to one-hot encoding
            site_onehot_encoding = site_onehot.loc[match.index].values.flatten()
            sex = match.iloc[0]['SEX']
            age = match.iloc[0]['AGE_AT_SCAN']

            # Add dynamic connectivity matrix, label, and file name to the list
            FC_list.append(static_matrix_dir)
            Label_list.append(y)
            Site_list.append(site_onehot_encoding)
            Filename_list.append(sample_id)
            sex_list.append(sex)
            age_list.append(age)
            Signals_list.append(signals)

            print(f"Processed {sample_id}, loaded dynamic matrices.")

    # Convert lists to arrays
    FC_list = np.array(FC_list, dtype=object)
    FC_dynamic = np.array(FC_dynamic_list, dtype=object)
    Label_array = np.array(Label_list, dtype=np.int64)
    Site_array = np.array(Site_list, dtype=object)
    Signals_list = np.array(Signals_list, dtype=object)
    Sex_array = np.array(sex_list, dtype=np.int64)
    Age_array = np.array(age_list, dtype=np.int64)
    Filename_array = np.array(Filename_list, dtype=object)

    # Save the dynamic connectivity matrix, label, site information, and file name into a dictionary
    data_dict = {
        'corr': FC_list,  # Static functional connectivity matrix
        'dcorr': FC_dynamic,  # Dynamic functional connectivity matrix
        'label': Label_array,  # Label
        'site': Site_array,  # Site
        'sub': Filename_array,  # File name
        'age': Age_array,
        'sex': Sex_array,
        'signals': Signals_list
    }

    # Save the dictionary as a .npy file
    output_file = os.path.join(output_dir, 'connectivity_data_abide_cc200.npy')
    np.save(output_file, data_dict)

    print(f"All data saved to {output_file}")


# Main function
if __name__ == "__main__":
    # Path to the dynamic connectivity matrix folder
    static_matrix_dir = r"E:\Technolgy_learning\Learning_code\New_Stage\shujuji\ABIDE_I\cc200\static"
    dynamic_matrix_dir = r"E:\Technolgy_learning\Learning_code\New_Stage\shujuji\ABIDE_I\cc200\static"
    output_dir = r'E:\Technolgy_learning\Learning_code\New_Stage\KMGCN-main\KMGCN-yanxu\dataset\generrate_abide\processed'  # Final data output folder

    # Read CSV file
    df = pd.read_csv(
        r"E:\Technolgy_learning\Learning_code\New_Stage\shujuji\ABIDE_I\cc200\Phenotypic_V1_0b_preprocessed1.csv",
    )
    # df['Label'] = df['ID'].apply(lambda x: x.split('-')[1])  # Extract label
    process_dynamic_connectivity_files(static_matrix_dir, dynamic_matrix_dir, output_dir, df)
