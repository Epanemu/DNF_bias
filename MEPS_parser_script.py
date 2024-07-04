import numpy as np
import pandas as pd

xlsx_path = "data/h233.xlsx"
csv_path = "data/h233.csv"
selected_cols_path = "data/selected_cols.xlsx"
filtered_xlsx_path = "data/filtered_data.xlsx"
interpreted_xlsx_path = "data/interpreted_data.xlsx"
important_attributes = [
    "SEX",
    "RACEV1X",
    "HISPANX",
    "TTLP21X",
    "FAMINC21",
    "HRWG53H",
    "POVCAT21",
    "AGE21X",
    "YRSINUS",
    "BORNUSA",
    "WHTLGSPK",
    "INTVLANG",
    "ADLANG42",
    "HWELLSPK",
    "MARRY21X",
    "EDUCYR",
    "PERWT21F",
    "INSCOPE",
    "REGION21",
    "HAVEUS42",
    "SDPROBNONE",
    "SDSTRESS",
    "SDGETTGT",
    "SDDSCRMDR",
    "SDDSCRMWRK",
    "SDDSCRMJOB",
    "SDDSCRMHS",
    "SDDSCRMPOL",
    "SDDSCRMPUB",
    "SDDSCRMSTR",
    "SDPUBTRANS",
    "ANYLMI21",
    "INSCOV21",
]
# -----------------


def get_selected_cols_file():
    print("Reading the xlsx file.")
    df = pd.read_csv(csv_path)
    # df = pd.read_excel(xlsx_path, engine='openpyxl')

    print("Filtering the table.")
    filtered_df = df[important_attributes]

    filtered_df.to_excel(selected_cols_path, index=False)
    print("Filtered Excel file has been created successfully.")


def get_filtered_data():
    print("Filtering the data.")
    df = pd.read_excel(selected_cols_path, engine="openpyxl")

    df = df[df["PERWT21F"] != 0]
    df = df[df["INSCOPE"] == 1]

    conditions = [
        (df["SDDSCRMDR"] == 1)
        | (df["SDDSCRMWRK"] == 1)
        | (df["SDDSCRMJOB"] == 1)
        | (df["SDDSCRMHS"] == 1)
        | (df["SDDSCRMPOL"] == 1)
        | (df["SDDSCRMPUB"] == 1)
        | (df["SDDSCRMSTR"] == 1),
        (df["SDDSCRMDR"] == 2)
        | (df["SDDSCRMWRK"] == 2)
        | (df["SDDSCRMJOB"] == 2)
        | (df["SDDSCRMHS"] == 2)
        | (df["SDDSCRMPOL"] == 2)
        | (df["SDDSCRMPUB"] == 2)
        | (df["SDDSCRMSTR"] == 2),
        (df["SDDSCRMDR"] == -15)
        | (df["SDDSCRMWRK"] == -15)
        | (df["SDDSCRMJOB"] == -15)
        | (df["SDDSCRMHS"] == -15)
        | (df["SDDSCRMPOL"] == -15)
        | (df["SDDSCRMPUB"] == -15)
        | (df["SDDSCRMSTR"] == -15),
    ]
    choices = [1, 2, -15]
    df["SDDCOMB"] = np.select(conditions, choices, default=-1)
    # OR logical function for all 7 columns

    df.drop(
        columns=[
            "SDDSCRMDR",
            "SDDSCRMWRK",
            "SDDSCRMJOB",
            "SDDSCRMHS",
            "SDDSCRMPOL",
            "SDDSCRMPUB",
            "SDDSCRMSTR",
        ],
        inplace=True,
    )
    df.to_excel(filtered_xlsx_path, index=False)


def interpret_data():
    print("Reading the filtered xlsx file.")
    df = pd.read_excel(filtered_xlsx_path, engine="openpyxl")

    sex_mapping = {1: "MALE", 2: "FEMALE"}
    df["SEX"] = df["SEX"].map(sex_mapping)

    race_mapping = {
        1: "WHITE",
        2: "BLACK",
        3: "AMER INDIAN/ALASKA NATIVE",
        4: "ASIAN/NATV HAWAIIAN/PACFC ISL",
        6: "MULTIPLE RACES REPORTED",
    }
    df["RACEV1X"] = df["RACEV1X"].map(race_mapping)

    ethnicity_mapping = {1: "HISPANIC", 2: "NOT HISPANIC"}
    df["HISPANX"] = df["HISPANX"].map(ethnicity_mapping)

    hrwg_mapping = {-10: "HOURLY WAGE >= $105.77", -1: "INAPPLICABLE"}
    df["HRWG53H"] = df["HRWG53H"].map(hrwg_mapping).fillna(df["HRWG53H"])

    poverty_mapping = {
        1: "POOR/NEGATIVE",
        2: "NEAR POOR",
        3: "LOW INCOME",
        4: "MIDDLE INCOME",
        5: "HIGH INCOME",
    }
    df["POVCAT21"] = df["POVCAT21"].map(poverty_mapping)

    yrsinus_mapping = {
        -8: "DK",
        -7: "REFUSED",
        -1: "INAPPLICABLE",
        1: "LESS THAN 1 YEAR",
        2: "1 YEAR, LESS THAN 5 YEARS",
        3: "5 YEARS, LESS THAN 10 YEARS",
        4: "10 YEARS, LESS THAN 15 YEARS",
        5: "15 YEARS OR MORE",
    }
    df["YRSINUS"] = df["YRSINUS"].map(yrsinus_mapping)

    bornusa_mapping = {
        -15: "CANNOT BE COMPUTED",
        -8: "DK",
        -7: "REFUSED",
        -1: "INAPPLICABLE",
        1: "YES",
        2: "NO",
    }
    df["BORNUSA"] = df["BORNUSA"].map(bornusa_mapping)

    hwellspk_mapping = {
        -1: "INAPPLICABLE",
        1: "VERY WELL",
        2: "WELL",
        3: "NOT WELL",
        4: "NOT AT ALL",
        5: "UNDER AGE 5 IN ROUND 1 AND OTHLANG=1,INAPP",
    }
    df["HWELLSPK"] = df["HWELLSPK"].map(hwellspk_mapping)

    lang_mapping = {
        -7: "REFUSED",
        -1: "INAPPLICABLE",
        1: "SPANISH",
        2: "ANOTHER NON-ENGLISH",
        5: "UNDER 5 YEARS OLD - INAPPLICABLE",
    }
    df["WHTLGSPK"] = df["WHTLGSPK"].map(lang_mapping)

    interv_mapping = {
        1: "ENGLISH",
        2: "SPANISH",
        5: "BOTH ENGLISH AND SPANISH",
        91: "OTHER LANGUAGE",
    }
    df["INTVLANG"] = df["INTVLANG"].map(interv_mapping)

    saq_mapping = {
        -1: "INAPPLICABLE",
        1: "ENGLISH VERSION SAQ WAS ADMINISTERED",
        2: "SPANISH VERSION SAQ WAS ADMINISTERED",
    }
    df["ADLANG42"] = df["ADLANG42"].map(saq_mapping)

    marry_mapping = {
        -8: "DK",
        -7: "REFUSED",
        1: "MARRIED",
        2: "WIDOWED",
        3: "DIVORCED",
        4: "SEPARATED",
        5: "NEVER MARRIED",
        6: "UNDER AGE 16 - INAPPLICABLE",
    }
    df["MARRY21X"] = df["MARRY21X"].map(marry_mapping)

    edu_mapping = {
        -15: "CANNOT BE COMPUTED",
        -8: "DK",
        -7: "REFUSED",
        -1: "INAPPLICABLE",
        0: "NO SCHOOL/KINDERGARTEN ONLY",
        1: "ELEMENTARY GRADES 1",
        2: "ELEMENTARY GRADES 2",
        3: "ELEMENTARY GRADES 3",
        4: "ELEMENTARY GRADES 4",
        5: "ELEMENTARY GRADES 5",
        6: "ELEMENTARY GRADES 6",
        7: "ELEMENTARY GRADES 7",
        8: "ELEMENTARY GRADES 8",
        9: "HIGH SCHOOL GRADES 9",
        10: "HIGH SCHOOL GRADES 10",
        11: "HIGH SCHOOL GRADES 11",
        12: "GRADE 12",
        13: "1 YEAR COLLEGE",
        14: "2 YEARS COLLEGE",
        15: "3 YEARS COLLEGE",
        16: "4 YEARS COLLEGE",
        17: "5+ YEARS COLLEGE",
    }
    df["EDUCYR"] = df["EDUCYR"].map(edu_mapping)

    inscope_mapping = {
        1: "INSCOPE AT SOME TIME DURING 2021",
        2: "OUT-OF-SCOPE FOR ALL OF 2021",
    }
    df["INSCOPE"] = df["INSCOPE"].map(inscope_mapping)

    region_mapping = {
        -1: "INAPPLICABLE",
        1: "NORTHEAST",
        2: "MIDWEST",
        3: "SOUTH",
        4: "WEST",
    }
    df["REGION21"] = df["REGION21"].map(region_mapping)

    haveus_mapping = {-8: "DK", -7: "REFUSED", -1: "INAPPLICABLE", 1: "YES", 2: "NO"}
    df["HAVEUS42"] = df["HAVEUS42"].map(haveus_mapping)

    home_mapping = {-15: "CANNOT BE COMPUTED", -1: "INAPPLICABLE", 1: "YES", 2: "NO"}
    df["SDPROBNONE"] = df["SDPROBNONE"].map(home_mapping)

    stress_mapping = {
        -15: "CANNOT BE COMPUTED",
        -1: "INAPPLICABLE",
        1: "NOT AT ALL",
        2: "A LITTLE BIT",
        3: "SOMEWHAT",
        4: "QUITE A BIT",
        5: "VERY MUCH",
    }
    df["SDSTRESS"] = df["SDSTRESS"].map(stress_mapping)

    sdge_mapping = {
        -15: "CANNOT BE COMPUTED",
        -1: "INAPPLICABLE",
        0: "NEVER",
        1: "1 TIME",
        2: "2 TIMES",
        3: "3 TIMES",
        4: "4 TIMES",
        5: "5 TIMES",
        6: "6 OR MORE TIMES",
    }
    df["SDGETTGT"] = df["SDGETTGT"].map(sdge_mapping)

    sdoh_mapping = {
        -15: "CANNOT BE COMPUTED",
        -1: "INAPPLICABLE",
        1: "EXCELLENT",
        2: "VERY GOOD",
        3: "GOOD",
        4: "FAIR",
        5: "POOR",
        6: "EXCELLENT OR VERY GOOD",
        7: "VERY GOOD OR GOOD",
        8: "GOOD OR FAIR",
        9: "FAIR OR POOR",
    }
    df["SDPUBTRANS"] = df["SDPUBTRANS"].map(sdoh_mapping)

    anylmi_mapping = {-15: "CANNOT BE COMPUTED", -1: "INAPPLICABLE", 1: "YES", 2: "NO"}
    df["ANYLMI21"] = df["ANYLMI21"].map(anylmi_mapping)

    inscov_mapping = {1: "ANY PRIVATE", 2: "PUBLIC ONLY", 3: "UNINSURED"}
    df["INSCOV21"] = df["INSCOV21"].map(inscov_mapping)

    comb_mapping = {-15: "CANNOT BE COMPUTED", -1: "INAPPLICABLE", 1: "YES", 2: "NO"}
    df["SDDCOMB"] = df["SDDCOMB"].map(comb_mapping)

    print("Saving interpreted data to new file.")
    df.to_excel(interpreted_xlsx_path, index=False, sheet_name="Sheet1")
    print("Interpreted Excel file has been created successfully.")


def convert_to_csv():
    print("Reading the final xlsx file.")
    df = pd.read_excel(interpreted_xlsx_path, engine="openpyxl")
    print("Converting to the csv.")
    df.to_csv("data.csv", index=False, header=True)
    print("Converted!")


if __name__ == "__main__":
    get_selected_cols_file()
    get_filtered_data()
    interpret_data()
    convert_to_csv()
