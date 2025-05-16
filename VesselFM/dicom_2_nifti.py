import os
import shutil
import dicom2nifti # to convert DICOM files to the NIftI format

def convert_2_nifti(input_directory, output_directory):
    """
    input_directory: directory of dicom study
    output_directory: directory of NIfTI files
    """
    list_studies = os.listdir(input_directory)
    for study in list_studies:
        input_path = os.path.join(input_directory, study)
        output_path =os.path.join(output_directory, study)
        # The dicom2nifti does not correctly find the name of the study, so a directory with the name of dicom study
        # is created to avoid issues caused by creating files with the same name.
        os.makedirs(output_path, exist_ok=True)
        dicom2nifti.convert_dir.convert_directory(input_path, output_path, compression=True, reorient=True)

def create_input_path_vesselfm(output_directory, input_path_dir):
    """
    :param output_directory: directory of NIfTI files the outputs of dicom2nifit
    :param input_path_dir:  directory of renamed NIfTI files according to the name of the study.
    """
    list_studies = os.listdir(output_directory)
    os.makedirs(input_path_dir, exist_ok=True)
    for study in list_studies:
        # The dicom2nifti does not correctly find the name of the study, so manually changed the name of the study.
        nifti_file_name = os.listdir(os.path.join(output_directory, study))
        old_path = os.path.join(output_directory, study, nifti_file_name[0])
        new_path = os.path.join(output_directory, study, f"{study}.nii.gz")
        os.rename(old_path, new_path)
        shutil.copy(new_path, input_path_dir)

def main():
    input_directory = "./dicom_original"  # Replace with your DICOM directory
    output_directory = "./output_dicom2nifit"  # Replace with the NIfTI files output path
    os.makedirs(output_directory, exist_ok=True)
    convert_2_nifti(input_directory, output_directory)
    create_input_path_vesselfm(output_directory, input_path_dir='./input_nifti')


if __name__ == "__main__":
    main()
