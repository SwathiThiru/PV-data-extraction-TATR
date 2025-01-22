import re
import numpy as np
from statistics import mode, StatisticsError

def _determine_table_axis(
        table: list
) -> str:

    threshold = 0.5
    table_as_list = table

    # Create an empty list that will contain the raw values on which
    # the decision of either vertical or horizontal will be made.
    table_of_extracted_values = []

    # Iterate over the rows in the table
    for row in table_as_list:

        # This empty list will hold all the values that are extracted
        # from the current row
        extracted_row = []

        # Iterate over all columns in the current row
        for col in row:

            match = re.search(
                pattern= "[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                string=str(col)
            )

            if match is not None:

                col = match.group(0)

                # The following try-except clauses are used to determine if
                # the current element is a number or not.
                try:
                    curr_val = float(col)
                    extracted_row.append(curr_val)

                except ValueError as e:
                    pass

        # If something is extracted then add it to the main matrix
        if len(extracted_row) != 0:
            table_of_extracted_values.append(extracted_row)

    # If no values were extracted then return nothing
    if not table_of_extracted_values:
        return None

    # Calculate the mode of lengths which will be the most common length
    # of values found. This can then be used to remove misidentified values
    extracted_values_lens = [len(x) for x in table_of_extracted_values]

    try:
        m = mode(extracted_values_lens)

    except StatisticsError as e:
        return None

    mode_table_of_extracted_values = []

    for item in table_of_extracted_values:

        if len(item) == m:
            mode_table_of_extracted_values.append(item)

    table_of_extracted_values = mode_table_of_extracted_values

    # Determine if the extracted list of lists is a valid matrix, to do
    # that the lengths of all the rows are computed. If all the lengths
    # of the rows are equal then we have valid matrix otherwise not.
    is_valid_matrix = all([len(x) == len(table_of_extracted_values[0]) for x in table_of_extracted_values])

    # Exit with None if the matrix is not valid.
    if not is_valid_matrix:
        return None

    table_of_extracted_values = np.asarray(table_of_extracted_values)

    # print(table_of_extracted_values)

    # Calculate the variance for the extracted matrix in both the horizontal
    # and the vertical directions. The direction of least variation will be
    # the direction of the table
    vertical_var = np.sum(np.var(table_of_extracted_values, axis=1))
    horizontal_var = np.sum(np.var(table_of_extracted_values, axis=0))
    diff = abs(vertical_var - horizontal_var)

    if diff < threshold:
        return "dual-axis"
    elif horizontal_var > vertical_var:
        return 'horizontal'
    else:
        return 'vertical'

def evaluate_table_axis_function(test_data):
    correct = 0
    total = len(test_data)

    for test_case in test_data:
        table = test_case["table"]
        true_label = test_case["label"]

        # Call the function
        result = _determine_table_axis(table)

        # Compare the result with the true label
        if result == true_label:
            correct += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def main():
    test_data = [
        {"table": [['Type', '380', '385', '390', '395', '400', '405', '410'], ['Test Conditions', 'sTC', 'sTC', 'sTC', 'sTC', 'sTC', 'sTC', 'sTC'], ['Front Side Maximum Power (W)', '380', '385', '390', '395', '400', '405', '410'], ['. . Open ony Voltage', '47.80', '48.10', '48.30', '48.50', '48.80', '49.00', '49.29'], ['Shore Cireay Curent”', '998', '10.04', '10.11', '10.17', '10.24', '1031', '1038'], ['Working i Voltage', '49.0', '403', '40.6', '408', '41.0', '41.26', '4151'], ['Working oe Current', 'g 50', '958', '9.62', '9.69', '976', '981', '9.88'], ['PV Module Total Efficiency (%)', '2161', '2189', '22.18', '22.46', '22.75', '23.03', '23.32'], ['Bifaciality Rate (%)', '40.7401', '40.7401', '40.7401', '40.7401', '40.7401', '40.7401', '40.7401']], "label": "vertical"},
        {"table": [['ELECTRICAL DATA (STC) Circuit Voltage-Voc (V) 49.0 49.1 49.2 49.3 49.4 Circuit Current-Isc (A) 11.80 12.36 12.93 13.49 14.05 Coefficient of Isc 0.040%/°C connect Fuse in Combiner Box with two or more strings in parallel connection)', 'ELECTRICAL DATA (STC) Circuit Voltage-Voc (V) 49.0 49.1 49.2 49.3 49.4 Circuit Current-Isc (A) 11.80 12.36 12.93 13.49 14.05 Coefficient of Isc 0.040%/°C connect Fuse in Combiner Box with two or more strings in parallel connection)', 'ELECTRICAL DATA (STC) Circuit Voltage-Voc (V) 49.0 49.1 49.2 49.3 49.4 Circuit Current-Isc (A) 11.80 12.36 12.93 13.49 14.05 Coefficient of Isc 0.040%/°C connect Fuse in Combiner Box with two or more strings in parallel connection)', 'ELECTRICAL DATA (STC) Circuit Voltage-Voc (V) 49.0 49.1 49.2 49.3 49.4 Circuit Current-Isc (A) 11.80 12.36 12.93 13.49 14.05 Coefficient of Isc 0.040%/°C connect Fuse in Combiner Box with two or more strings in parallel connection)', 'ELECTRICAL DATA (STC) Circuit Voltage-Voc (V) 49.0 49.1 49.2 49.3 49.4 Circuit Current-Isc (A) 11.80 12.36 12.93 13.49 14.05 Coefficient of Isc 0.040%/°C connect Fuse in Combiner Box with two or more strings in parallel connection)', 'ELECTRICAL DATA (STC) Circuit Voltage-Voc (V) 49.0 49.1 49.2 49.3 49.4 Circuit Current-Isc (A) 11.80 12.36 12.93 13.49 14.05 Coefficient of Isc 0.040%/°C connect Fuse in Combiner Box with two or more strings in parallel connection)'], ['Peak Power Watts-Pmax (Wp)* 5%', '430 10%', '435 15%', '440 20%', '445 25%', '450'], ['Maximum Power Voltage-Vmp (V} Power Bifaciality: 705%', '40.5', '40.8', '41.1', '41.4', '41.7'], ['Maximum Power Current-Ime (A)', '10.62', '10.67', '10.71', '10.75', '10.80'], ['Open Circuit Voltage-Voc (V)', '48.7', '48.9', '49.1', '49.3', '49.5'], ['Short Circuit Current-Isc (A)', '11.20', '11.29', '11.37', '11.45', '11.53'], ['Module Efficiency nm (%)', '19.7', '20.0', '20.2', '20.4', '20.6'], ['Power Tolerance-Pmax (W)', 'O~+5', 'O~+5', 'O~+5', 'O~+5', 'O~+5']], "label": "horizontal"},
        {"table": [['TEMPERATURE RATINGS Coefficient | -0.25%/°C Coefficient of Isc 0.040%/°C', 'TEMPERATURE RATINGS Coefficient | -0.25%/°C Coefficient of Isc 0.040%/°C'], ['NMOT (Nominal Module Operating Temperature) connect Fuse in Combiner Box with two or more strings in', '41°C (43°C) parallel connection)'], ['Temperature Coefficient of Pmax', '-0.34%/°C'], ['Temperature', ''], ['Temperature', '']], "label": "horizontal"},
        {"table": [['Operational Temperature', '-40~+85°C', 'Modules per box', '35 pieces'], ['Maximum System Voltage |', '1500V OC (IEC)', 'Modules per 40’container', '770 pieces'], ['Max', '', '', '']], "label": "vertical"},
        {"table": [['Electrical Characteristics', 'Electrical Characteristics', 'Electrical Characteristics', 'Electrical Characteristics', 'Electrical Characteristics', 'Electrical Characteristics'], ['Power level', '435', '440', '445', '450', '455'], ['Pmax(W)', '435', '440', '445', '450', '455'], ['Vmp (Vv)', '41.04', '41.24', '41.44', '41.63', '41.82'], ['Imp (A)', '10.60', '10.67', '10.74', '10.81', '10.88'], ['Voc (V)', '49.25', '49.44', '49.65', '49.85', '$0.06'], ['Isc (A)', '11.11', '11.17', '11.24', '11.31', '11.38'], ['Module efficiency (%)', '20.01', '20.24', '20.47', '20.70', '20.93'], ['Maximum system voltage (V)', '1500', '1500', '1500', '1500', '1500'], ['Fuse Rating (A)', '20', '20', '20', '20', '20'], ['Temperature coefficient Pmax (%°C)', '-0.350', '-0.350', '-0.350', '-0.350', '-0.350'], ['Temperature coefficient Isc (%°C)', '0.05', '0.05', '0.05', '0.05', '0.05'], ['Temperature coefficient Voc (%°C)', '-0.275', '-0.275', '-0.275', '-0.275', '-0.275']], "label": "horizontal"},
        {"table": [['Working acteristics', 'Working acteristics', 'Working acteristics', 'Working acteristics', 'Working acteristics'], ['Power level 435', '440', '445', '450', '455'], ['Pmax (W) 323', '327', '330', '334', '337'], ['Vp (Vv) 37.83', '38.03', '38.11', '38.35', '38.43'], ['Imp (A) 8.54', '8.60', '8.66', '8.71', '8.77'], ['Voc (V) 45.60', '45.81', '45.99', '46.19', '46.01'], ['Isc (A) 8.96', '9.05', '9.10', '9.16', '9.20'], ['Power tolerance (%)', '0~+3', '0~+3', '0~+3', '0~+3'], ['NOCT (°C)', '4442', '4442', '4442', '4442'], ['NOCT:Conditions:Irradiance 800W/m? ambient', 'temperature', '20°C,', 'wind speed', '1m/s']], "label": "vertical"},
        {"table": [['| Packing Configuration', '| Packing Configuration'], ['Pieces per pallet', '36'], ['Size of packing (mm)', '2130*1140*1190'], ['Weight of packing (kg)', '1040'], ['Pi rcontainer', '792'], ['Size of container', '40°HC']], "label": "horizontal"},
        {"table": [['Electrical Data (STC) Maximum Power (Pmax/W) Circuit Voltage (Voc/V) 35.97 Circuit Current (Isc/A) 11.06 Orientation 108(6 x 18) Module Dimensions = 1722 1134K30mm Temperature Coefficient (Voc) Temperature Coefficient (Isc) container 936', '410 36.12 11.15 pcs', '415 36.27 11.25', '420 36.42 11.34 -0.250%/°C 0.045%/°C 36 pcs', '425 36.57 11.44 +36 pes', '430'], ['Open Circuit Voltage (Voc/V) Voltage at Maximum Power (Vmp/V) 29.50 Weight 25.0kg (Nominal Moudule Operating Temperature)', '37.90 29.65', '38.05 29.80', '38.20 29.95 41+3°C', '38.35 30.10', '38.50'], ['Short Circuit Current (Isc/A) Current at Maximum Power (Imp/A) 10.45 2.0mm high transmittance,', '13.52 10.53', '13.63 10.61 reinforced glass', '13.74 10.69', '13.85 10.77', '13.96'], ['Voltage at Maximum Power (Vmp/V) (Nominal Moudule Operating Temperature): Irradiance Backsheet 2.0mm part of the', '31.35 800W/m’, structure is', '31.50 Ambient Temperature grid-like white', '31.65 20°C , AM1.5, ceramic', '31.80 Wind Speed glass', '31.95 1m/s.'], ['Current at Maximum Power (Imp/A) Material Anodized aluminum', '13.08 alloy', '13.18', '13.28', '13.37', '13.46'], ['Module Efficiency (%) Junction Box Protection class IP68', '21.00', '21.25', '21.51', '21.76', '22.02'], ['Operating Temperature 4.0 mm’ positive pole: wire length can be', '-40° C~+85° C 200mm _ negative pole: 250 mm customized', '-40° C~+85° C 200mm _ negative pole: 250 mm customized', '-40° C~+85° C 200mm _ negative pole: 250 mm customized', '-40° C~+85° C 200mm _ negative pole: 250 mm customized', '-40° C~+85° C 200mm _ negative pole: 250 mm customized'], ['Maximum System Voltage Connector MC4 compatible', '1000/1500V connector', '1000/1500V connector', '1000/1500V connector', '1000/1500V connector', '1000/1500V connector'], ['STC (Standard Testing Conditions): lrradiance', '1000W/m’, Cell', 'Temperature', '25°C , AML.5', '', '']], "label": "horizontal"},
        {"table": [['Electrical Data (NMOT) Circuit Voltage (Voc/V) 35.97', '36.12', '36.27', '36.42', '36.57', ''], ['Maximum Power (Pmax/W) Circuit Current (Isc/A) 11.06', '308 11.15', '312 11.25', '316 11.34', '320 11.44', '324'], ['Voltage at Maximum Power (Vmp/V) 29.50', '29.65', '29.80', '29.95', '30.10', ''], ['Current at Maximum Power (Imp/A) 10.45', '10.53', '10.61', '10.69', '10.77', ''], ['(Nominal Moudule Operating Temperature): Irradiance', '800W/m’,', 'Ambient Temperature', '20°C , AM1.5,', 'Wind Speed', '1m/s.'], ['NMOT', '', '', '', '', '']], "label": "horizontal"},
        {"table": [['Temperature Coefficient (Pm) Temperature Coefficient (Isc)', '-0.300%/°C 0.045%/°C'], ['NMOT (Nominal Moudule Operating Temperature)', '41+3°C']], "label": "horizontal"},
        {"table": [['Packaging Configuration', '', ''], ['Number of Modules per Pallet', '', '25'], ['Number of Modules per 40ft HQ Container |', '[EA] |', '650'], ['Packaging Box Dimensions (Lx W x H)', '', '1,750 x 1,120 x 1,221'], ['Packaging Box Gross Weight', '', '464']], "label": "horizontal"},
        {"table": [['Temperature Characteristics Temperature Coe’cient -0.27 %/C', 'Temperature Characteristics Temperature Coe’cient -0.27 %/C'], ['Pmax Temperature Coe’cient Temperature Coe°cient +0.05', '-0.34 %/PC %/°C'], ['Voc . ~ Operating Temperature -40°+85', '° °C'], ['Isc Nominal Operating Cell Temperature (NOCT) 4542', '°C']], "label": "dual-axis"},
        {"table": [['Packing Configuration', '17'], ['Container per Container', '40’HQ'], ['Pieces per Container', '527 [A]'], ['Pallets', 'Current'], ['Pieces', '']], "label": "horizontal"},
        {"table": [['SW 350 Maximalleistung Pras 350 Wp', 'SW 350 Maximalleistung Pras 350 Wp', 'SW 350 Maximalleistung Pras 350 Wp'], ['Leerlaufspannung', 'Us', '48V'], ['Spannung bei Maximalleistung', 'Unos', '38,4V'], ['Kurzschlussstrom', 'ly', '982A'], ['Strom bei Maximalleistung', 'Vinop', '917A'], ['Modulwirkungsgrad', 'Nn', '17,54 %']], "label": "horizontal"},
        {"table": [['(Reference to S65W Front)', '(Reference to S65W Front)', '(Reference to S65W Front)', '(Reference to S65W Front)', '(Reference to S65W Front)', '(Reference to S65W Front)'], ['Backside Power Gain', '10%', '15%', '20%', '25%', '30%'], ['Maximum Power At STC(Pmax)', '621.5', '650.0', '678.0', '7060', '734.5'], ['Short Circuit Current(|sc)', '1548', '1618', '1687', '17.55', '18.25'], ['Open Circuit Voltage(Voc)', '$1.07', '51.27', '5147', '5167', '51.87'], ['Maximum Power Current(Impp)', '14-61', '15.27,', '15.91', '16.56', 0], ['Maximum Power Voltage(Vmpp)', '42:55', '42.58', '42.61', '42.64', '42.67']], "label": "vertical"},
        {"table": [['Container', '40° HQ'], ['Pieces Per Pallet', '36'], ['Pallets Per Container', '20'], ['Pieces Per Container', '720']], "label": "horizontal"},
        {"table": [['ELECTRICAL DATA (STC) 48.27 48.51 48.75 16.00 1733 18.66 20.00 Glass -0.35% / °C -40°C~+85C 20', 'ELECTRICAL DATA (STC) 48.27 48.51 48.75 16.00 1733 18.66 20.00 Glass -0.35% / °C -40°C~+85C 20', 'ELECTRICAL DATA (STC) 48.27 48.51 48.75 16.00 1733 18.66 20.00 Glass -0.35% / °C -40°C~+85C 20', 'ELECTRICAL DATA (STC) 48.27 48.51 48.75 16.00 1733 18.66 20.00 Glass -0.35% / °C -40°C~+85C 20', 'ELECTRICAL DATA (STC) 48.27 48.51 48.75 16.00 1733 18.66 20.00 Glass -0.35% / °C -40°C~+85C 20'], ['Rated Power In Watts-Pmax (Wp) 11.21 11.27 11.33 ISOOVDC', '560', '565', '570', '575'], ['Maximum Power Voltage-Vmpp (V) Alloy, Silver 25A', '42.68', '42.84', '4299', '43.14'], ['Maximum Power Current-Impp (A)', '13.12', '13.19', '13.26', '13.33'], ['Open Circuit Voltage-Voc (V} 380mm or customized length', '50.47', '50.72', '5097', '51.23'], ['Short Circuit Current-Isc (A)', '13.92', '14.00', '14.08', '14.16'], ['Module Efficiency (%)', '21.67%', '21.86%', '22.06%', '22.25%'], ["STC Irradiation 1000 W.'m”, Cell Temperature 25, Air Mass AM15 accor", 'ding to EN', '60904-3', '', '']], "label": "vertical"},
        {"table": [['ELECTRICAL DATA (NMOT} 10.62 10.70 10.78 48.27 48.51 48.75', 'ELECTRICAL DATA (NMOT} 10.62 10.70 10.78 48.27 48.51 48.75', 'ELECTRICAL DATA (NMOT} 10.62 10.70 10.78 48.27 48.51 48.75', 'ELECTRICAL DATA (NMOT} 10.62 10.70 10.78 48.27 48.51 48.75', 'ELECTRICAL DATA (NMOT} 10.62 10.70 10.78 48.27 48.51 48.75'], ['Maximum Power-Pmax (Wp} 11.21 11.27 11.33', '4g', '424', '429', '435'], ['Maximum Power Voltage-Vmpp (V)', '3973', '3993', '40.15', '40.32'], ['Maximum Power Current-Impp (A)', '10.54', '', '', ''], ['Open Circuit Voltage-Voc (V}', '48.03', '', '', ''], ['Short Circuit Current-Isc (A)', '1.15', '', '', ''], ["NOCT Irradiation 800 W.'m?, ambient temperature 20, air mass 15,", 'wind speed 1 m/s', '', '', '']], "label": "horizontal"},
        {"table": [['TEMPERATURE & MAXIMUM RATINGS -0.35% / °C', 'TEMPERATURE & MAXIMUM RATINGS -0.35% / °C'], ['Nominal Module Operating Temperature (NMOT} -40°C~+85C', '4442°C'], ['Temperature Coefficient of VOC ISOOVDC', '-0.275% / °C'], ['Temperature Coefficient of ISC 25A', '0.045% / °C'], ['Temperature Coefficient of PMAX', ''], ['Operational Temperature', ''], ['Maximum System Voltage', ''], ['Max Series Fuse Rating', '']], "label": "vertical"},
        {"table": [['', '40 FT (HQ)'], ['Number of Modules Per Container', '620'], ['Number of Modules Per Pallet', '3]'], ['Number of Pallets Per Container', '']], "label": "vertical"},
        {"table": [['Module type', 'TM -650 M-132 HC)', 'TM -650 M-132 HC)', '§=96TM-655M-132 HC]', '§=96TM-655M-132 HC]', '§=TM-660 M-132 HC', '§=TM-660 M-132 HC', 'TM-665M-132 HC', 'TM-665M-132 HC', 'TM -670 M-132 HC', 'TM -670 M-132 HC'], ['Testing condition', 'STC', 'NOTC', 'STC', 'NOTC', 'STC', 'NOTC', 'STC', 'NOTC', 'STC', 'NOTC'], ['Maximum Power (Pmax/w)', '650', '484', '655', '487', '660', '491', '665', '495', '670', '498'], ['Open Circuit Voltage (Voc/V)', '45', '42,6', '45,2', '42,8', '45,4', '43,0', '45.6', '43,1', '45,8', '43,3'], ['Short Circuit Current {(Isc/A)', '18,39', '14,41', '18,43', '14,43', '18,47', '14,48', '18,51', '14,57', '18,55', '14,58'], ['Voltage at Maximum Power (Vmp/V)', '37,6', '35,7', '37,8', '35,9', '38', '36,05', '38,2', '36,1', '38,4', '36,3'], ['Current at maximum Power {Imp/A)', '17,29', '13,56', '17,33', '13,57', '17,37', '13,62', '17,41', '13,71', '17,45', '13,72'], ['Module Efficiency (%)', '20,90%', '15,58%', '21,10%', '15.68%', '21,30%', '15,81%', '21.40%', '15,94%', '21,60%', '16,03%']], "label": "horizontal"},
        {"table": [['Temperatura ratings (STC)', ''], ['Temperature Coefficient of Isc', '+0.05%/°C'], ['Temperature Coefficient of Voc', '-0.27%/°C'], ['Temperature Coefficient of Pmax', '-0.35%/°C']], "label": "dual-axis"},
        {"table": [['Packaging Configuration', ''], ['Packing Type', '40°HQ'], ['Piece/Pallet', '31']], "label": "horizontal"},
        {"table": [['Temperature Coefficient (Pm)', '-0.300%/℃'], ['Temperature Coefficient (Voc)', '-0.250%/℃'], ['Temperature Coefficient (Isc)', '0.045%/℃'], ['NMOT (Nominal Moudule Operating Temperature)', '41±3℃']], "label": "horizontal"},
        {"table": [['Transportation methods', 'Number of modules per cabinet', 'Number of modules per pallet'], ['40HQ container', '936 pcs', '36 pcs +36 pcs']], "label": "vertical"}
    ]
    evaluate_table_axis_function(test_data)
    print("evaluation complete")

if __name__ == "__main__":
    main()




"""import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import pickle
import re
import pandas as pd
import numpy as np
from statistics import mode, StatisticsError

def _determine_table_axis(
        table: list
) -> str:

    table_as_list = table

    # Create an empty list that will contain the raw values on which
    # the decision of either vertical or horizontal will be made.
    table_of_extracted_values = []

    # Iterate over the rows in the table
    for row in table_as_list:

        # This empty list will hold all the values that are extracted
        # from the current row
        extracted_row = []

        # Iterate over all columns in the current row
        for col in row:

            match = re.search(
                pattern="\d+.\d+|\d+",
                string=str(col)
            )

            if match is not None:

                col = match.group(0)

                # The following try-except clauses are used to determine if
                # the current element is a number or not.
                try:
                    curr_val = float(col)
                    extracted_row.append(curr_val)

                except ValueError as e:
                    pass

        # If something is extracted then add it to the main matrix
        if len(extracted_row) != 0:
            table_of_extracted_values.append(extracted_row)

    # If no values were extracted then return nothing
    if not table_of_extracted_values:
        return None

    # Calculate the mode of lengths which will be the most common length
    # of values found. This can then be used to remove misidentified values
    extracted_values_lens = [len(x) for x in table_of_extracted_values]

    try:
        m = mode(extracted_values_lens)

    except StatisticsError as e:
        return None

    mode_table_of_extracted_values = []

    for item in table_of_extracted_values:

        if len(item) == m:
            mode_table_of_extracted_values.append(item)

    table_of_extracted_values = mode_table_of_extracted_values

    # Determine if the extracted list of lists is a valid matrix, to do
    # that the lengths of all the rows are computed. If all the lengths
    # of the rows are equal then we have valid matrix otherwise not.
    is_valid_matrix = all([len(x) == len(table_of_extracted_values[0]) for x in table_of_extracted_values])

    # Exit with None if the matrix is not valid.
    if not is_valid_matrix:
        return None

    table_of_extracted_values = np.asarray(table_of_extracted_values)

    # print(table_of_extracted_values)

    # Calculate the variance for the extracted matrix in both the horizontal
    # and the vertical directions. The direction of least variation will be
    # the direction of the table
    vertical_var = np.sum(np.var(table_of_extracted_values, axis=1))
    horizontal_var = np.sum(np.var(table_of_extracted_values, axis=0))

    if horizontal_var > vertical_var:
        return 'horizontal'
    else:
        return 'vertical'

def evaluate_table_axis_function(test_data):
    correct = 0
    total = len(test_data)

    for test_case in test_data:
        table = test_case["table"]
        true_label = test_case["label"]

        # Call the function
        result = _determine_table_axis(table)

        # Compare the result with the true label
        if result == true_label:
            correct += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def main():
    test_data = [
        {"table": [['Type', '380', '385', '390', '395', '400', '405', '410'], ['Test Conditions', 'sTC', 'sTC', 'sTC', 'sTC', 'sTC', 'sTC', 'sTC'], ['Front Side Maximum Power (W)', '380', '385', '390', '395', '400', '405', '410'], ['. . Open ony Voltage', '47.80', '48.10', '48.30', '48.50', '48.80', '49.00', '49.29'], ['Shore Cireay Curent”', '998', '10.04', '10.11', '10.17', '10.24', '1031', '1038'], ['Working i Voltage', '49.0', '403', '40.6', '408', '41.0', '41.26', '4151'], ['Working oe Current', 'g 50', '958', '9.62', '9.69', '976', '981', '9.88'], ['PV Module Total Efficiency (%)', '2161', '2189', '22.18', '22.46', '22.75', '23.03', '23.32'], ['Bifaciality Rate (%)', '40.7401', '40.7401', '40.7401', '40.7401', '40.7401', '40.7401', '40.7401']], "label": "vertical"},
        {"table": [['ELECTRICAL DATA (STC) Circuit Voltage-Voc (V) 49.0 49.1 49.2 49.3 49.4 Circuit Current-Isc (A) 11.80 12.36 12.93 13.49 14.05 Coefficient of Isc 0.040%/°C connect Fuse in Combiner Box with two or more strings in parallel connection)', 'ELECTRICAL DATA (STC) Circuit Voltage-Voc (V) 49.0 49.1 49.2 49.3 49.4 Circuit Current-Isc (A) 11.80 12.36 12.93 13.49 14.05 Coefficient of Isc 0.040%/°C connect Fuse in Combiner Box with two or more strings in parallel connection)', 'ELECTRICAL DATA (STC) Circuit Voltage-Voc (V) 49.0 49.1 49.2 49.3 49.4 Circuit Current-Isc (A) 11.80 12.36 12.93 13.49 14.05 Coefficient of Isc 0.040%/°C connect Fuse in Combiner Box with two or more strings in parallel connection)', 'ELECTRICAL DATA (STC) Circuit Voltage-Voc (V) 49.0 49.1 49.2 49.3 49.4 Circuit Current-Isc (A) 11.80 12.36 12.93 13.49 14.05 Coefficient of Isc 0.040%/°C connect Fuse in Combiner Box with two or more strings in parallel connection)', 'ELECTRICAL DATA (STC) Circuit Voltage-Voc (V) 49.0 49.1 49.2 49.3 49.4 Circuit Current-Isc (A) 11.80 12.36 12.93 13.49 14.05 Coefficient of Isc 0.040%/°C connect Fuse in Combiner Box with two or more strings in parallel connection)', 'ELECTRICAL DATA (STC) Circuit Voltage-Voc (V) 49.0 49.1 49.2 49.3 49.4 Circuit Current-Isc (A) 11.80 12.36 12.93 13.49 14.05 Coefficient of Isc 0.040%/°C connect Fuse in Combiner Box with two or more strings in parallel connection)'], ['Peak Power Watts-Pmax (Wp)* 5%', '430 10%', '435 15%', '440 20%', '445 25%', '450'], ['Maximum Power Voltage-Vmp (V} Power Bifaciality: 705%', '40.5', '40.8', '41.1', '41.4', '41.7'], ['Maximum Power Current-Ime (A)', '10.62', '10.67', '10.71', '10.75', '10.80'], ['Open Circuit Voltage-Voc (V)', '48.7', '48.9', '49.1', '49.3', '49.5'], ['Short Circuit Current-Isc (A)', '11.20', '11.29', '11.37', '11.45', '11.53'], ['Module Efficiency nm (%)', '19.7', '20.0', '20.2', '20.4', '20.6'], ['Power Tolerance-Pmax (W)', 'O~+5', 'O~+5', 'O~+5', 'O~+5', 'O~+5']], "label": "horizontal"},
        {"table": [['TEMPERATURE RATINGS Coefficient | -0.25%/°C Coefficient of Isc 0.040%/°C', 'TEMPERATURE RATINGS Coefficient | -0.25%/°C Coefficient of Isc 0.040%/°C'], ['NMOT (Nominal Module Operating Temperature) connect Fuse in Combiner Box with two or more strings in', '41°C (43°C) parallel connection)'], ['Temperature Coefficient of Pmax', '-0.34%/°C'], ['Temperature', ''], ['Temperature', '']], "label": "horizontal"},
        {"table": [['Operational Temperature', '-40~+85°C', 'Modules per box', '35 pieces'], ['Maximum System Voltage |', '1500V OC (IEC)', 'Modules per 40’container', '770 pieces'], ['Max', '', '', '']], "label": "vertical"},
        {"table": [['Electrical Characteristics', 'Electrical Characteristics', 'Electrical Characteristics', 'Electrical Characteristics', 'Electrical Characteristics', 'Electrical Characteristics'], ['Power level', '435', '440', '445', '450', '455'], ['Pmax(W)', '435', '440', '445', '450', '455'], ['Vmp (Vv)', '41.04', '41.24', '41.44', '41.63', '41.82'], ['Imp (A)', '10.60', '10.67', '10.74', '10.81', '10.88'], ['Voc (V)', '49.25', '49.44', '49.65', '49.85', '$0.06'], ['Isc (A)', '11.11', '11.17', '11.24', '11.31', '11.38'], ['Module efficiency (%)', '20.01', '20.24', '20.47', '20.70', '20.93'], ['Maximum system voltage (V)', '1500', '1500', '1500', '1500', '1500'], ['Fuse Rating (A)', '20', '20', '20', '20', '20'], ['Temperature coefficient Pmax (%°C)', '-0.350', '-0.350', '-0.350', '-0.350', '-0.350'], ['Temperature coefficient Isc (%°C)', '0.05', '0.05', '0.05', '0.05', '0.05'], ['Temperature coefficient Voc (%°C)', '-0.275', '-0.275', '-0.275', '-0.275', '-0.275']], "label": "horizontal"},
        {"table": [['Working acteristics', 'Working acteristics', 'Working acteristics', 'Working acteristics', 'Working acteristics'], ['Power level 435', '440', '445', '450', '455'], ['Pmax (W) 323', '327', '330', '334', '337'], ['Vp (Vv) 37.83', '38.03', '38.11', '38.35', '38.43'], ['Imp (A) 8.54', '8.60', '8.66', '8.71', '8.77'], ['Voc (V) 45.60', '45.81', '45.99', '46.19', '46.01'], ['Isc (A) 8.96', '9.05', '9.10', '9.16', '9.20'], ['Power tolerance (%)', '0~+3', '0~+3', '0~+3', '0~+3'], ['NOCT (°C)', '4442', '4442', '4442', '4442'], ['NOCT:Conditions:Irradiance 800W/m? ambient', 'temperature', '20°C,', 'wind speed', '1m/s']], "label": "horizontal"},
        {"table": [['| Packing Configuration', '| Packing Configuration'], ['Pieces per pallet', '36'], ['Size of packing (mm)', '2130*1140*1190'], ['Weight of packing (kg)', '1040'], ['Pi rcontainer', '792'], ['Size of container', '40°HC']], "label": "horizontal"},
        {"table": [['Electrical Data (STC) Maximum Power (Pmax/W) Circuit Voltage (Voc/V) 35.97 Circuit Current (Isc/A) 11.06 Orientation 108(6 x 18) Module Dimensions = 1722 1134K30mm Temperature Coefficient (Voc) Temperature Coefficient (Isc) container 936', '410 36.12 11.15 pcs', '415 36.27 11.25', '420 36.42 11.34 -0.250%/°C 0.045%/°C 36 pcs', '425 36.57 11.44 +36 pes', '430'], ['Open Circuit Voltage (Voc/V) Voltage at Maximum Power (Vmp/V) 29.50 Weight 25.0kg (Nominal Moudule Operating Temperature)', '37.90 29.65', '38.05 29.80', '38.20 29.95 41+3°C', '38.35 30.10', '38.50'], ['Short Circuit Current (Isc/A) Current at Maximum Power (Imp/A) 10.45 2.0mm high transmittance,', '13.52 10.53', '13.63 10.61 reinforced glass', '13.74 10.69', '13.85 10.77', '13.96'], ['Voltage at Maximum Power (Vmp/V) (Nominal Moudule Operating Temperature): Irradiance Backsheet 2.0mm part of the', '31.35 800W/m’, structure is', '31.50 Ambient Temperature grid-like white', '31.65 20°C , AM1.5, ceramic', '31.80 Wind Speed glass', '31.95 1m/s.'], ['Current at Maximum Power (Imp/A) Material Anodized aluminum', '13.08 alloy', '13.18', '13.28', '13.37', '13.46'], ['Module Efficiency (%) Junction Box Protection class IP68', '21.00', '21.25', '21.51', '21.76', '22.02'], ['Operating Temperature 4.0 mm’ positive pole: wire length can be', '-40° C~+85° C 200mm _ negative pole: 250 mm customized', '-40° C~+85° C 200mm _ negative pole: 250 mm customized', '-40° C~+85° C 200mm _ negative pole: 250 mm customized', '-40° C~+85° C 200mm _ negative pole: 250 mm customized', '-40° C~+85° C 200mm _ negative pole: 250 mm customized'], ['Maximum System Voltage Connector MC4 compatible', '1000/1500V connector', '1000/1500V connector', '1000/1500V connector', '1000/1500V connector', '1000/1500V connector'], ['STC (Standard Testing Conditions): lrradiance', '1000W/m’, Cell', 'Temperature', '25°C , AML.5', '', '']], "label": "vertical"},
        {"table": [['Electrical Data (NMOT) Circuit Voltage (Voc/V) 35.97', '36.12', '36.27', '36.42', '36.57', ''], ['Maximum Power (Pmax/W) Circuit Current (Isc/A) 11.06', '308 11.15', '312 11.25', '316 11.34', '320 11.44', '324'], ['Voltage at Maximum Power (Vmp/V) 29.50', '29.65', '29.80', '29.95', '30.10', ''], ['Current at Maximum Power (Imp/A) 10.45', '10.53', '10.61', '10.69', '10.77', ''], ['(Nominal Moudule Operating Temperature): Irradiance', '800W/m’,', 'Ambient Temperature', '20°C , AM1.5,', 'Wind Speed', '1m/s.'], ['NMOT', '', '', '', '', '']], "label": "horizontal"},
        {"table": [['Temperature Coefficient (Pm) Temperature Coefficient (Isc)', '-0.300%/°C 0.045%/°C'], ['NMOT (Nominal Moudule Operating Temperature)', '41+3°C']], "label": "vertical"},
        {"table": [['Packaging Configuration', '', ''], ['Number of Modules per Pallet', '', '25'], ['Number of Modules per 40ft HQ Container |', '[EA] |', '650'], ['Packaging Box Dimensions (Lx W x H)', '', '1,750 x 1,120 x 1,221'], ['Packaging Box Gross Weight', '', '464']], "label": "horizontal"},
        {"table": [['Temperature Characteristics Temperature Coe’cient -0.27 %/C', 'Temperature Characteristics Temperature Coe’cient -0.27 %/C'], ['Pmax Temperature Coe’cient Temperature Coe°cient +0.05', '-0.34 %/PC %/°C'], ['Voc . ~ Operating Temperature -40°+85', '° °C'], ['Isc Nominal Operating Cell Temperature (NOCT) 4542', '°C']], "label": "horizontal"},
        {"table": [['Packing Configuration', '17'], ['Container per Container', '40’HQ'], ['Pieces per Container', '527 [A]'], ['Pallets', 'Current'], ['Pieces', '']], "label": "horizontal"},
        {"table": [['SW 350 Maximalleistung Pras 350 Wp', 'SW 350 Maximalleistung Pras 350 Wp', 'SW 350 Maximalleistung Pras 350 Wp'], ['Leerlaufspannung', 'Us', '48V'], ['Spannung bei Maximalleistung', 'Unos', '38,4V'], ['Kurzschlussstrom', 'ly', '982A'], ['Strom bei Maximalleistung', 'Vinop', '917A'], ['Modulwirkungsgrad', 'Nn', '17,54 %']], "label": "horizontal"},
        {"table": [['(Reference to S65W Front)', '(Reference to S65W Front)', '(Reference to S65W Front)', '(Reference to S65W Front)', '(Reference to S65W Front)', '(Reference to S65W Front)'], ['Backside Power Gain', '10%', '15%', '20%', '25%', '30%'], ['Maximum Power At STC(Pmax)', '621.5', '650.0', '678.0', '7060', '734.5'], ['Short Circuit Current(|sc)', '1548', '1618', '1687', '17.55', '18.25'], ['Open Circuit Voltage(Voc)', '$1.07', '51.27', '5147', '5167', '51.87'], ['Maximum Power Current(Impp)', '14-61', '15.27,', '15.91', '16.56', 0], ['Maximum Power Voltage(Vmpp)', '42:55', '42.58', '42.61', '42.64', '42.67']], "label": "horizontal"},
        {"table": [['Container', '40° HQ'], ['Pieces Per Pallet', '36'], ['Pallets Per Container', '20'], ['Pieces Per Container', '720']], "label": "horizontal"},
        {"table": [['ELECTRICAL DATA (STC) 48.27 48.51 48.75 16.00 1733 18.66 20.00 Glass -0.35% / °C -40°C~+85C 20', 'ELECTRICAL DATA (STC) 48.27 48.51 48.75 16.00 1733 18.66 20.00 Glass -0.35% / °C -40°C~+85C 20', 'ELECTRICAL DATA (STC) 48.27 48.51 48.75 16.00 1733 18.66 20.00 Glass -0.35% / °C -40°C~+85C 20', 'ELECTRICAL DATA (STC) 48.27 48.51 48.75 16.00 1733 18.66 20.00 Glass -0.35% / °C -40°C~+85C 20', 'ELECTRICAL DATA (STC) 48.27 48.51 48.75 16.00 1733 18.66 20.00 Glass -0.35% / °C -40°C~+85C 20'], ['Rated Power In Watts-Pmax (Wp) 11.21 11.27 11.33 ISOOVDC', '560', '565', '570', '575'], ['Maximum Power Voltage-Vmpp (V) Alloy, Silver 25A', '42.68', '42.84', '4299', '43.14'], ['Maximum Power Current-Impp (A)', '13.12', '13.19', '13.26', '13.33'], ['Open Circuit Voltage-Voc (V} 380mm or customized length', '50.47', '50.72', '5097', '51.23'], ['Short Circuit Current-Isc (A)', '13.92', '14.00', '14.08', '14.16'], ['Module Efficiency (%)', '21.67%', '21.86%', '22.06%', '22.25%'], ["STC Irradiation 1000 W.'m”, Cell Temperature 25, Air Mass AM15 accor", 'ding to EN', '60904-3', '', '']], "label": "horizontal"},
        {"table": [['ELECTRICAL DATA (NMOT} 10.62 10.70 10.78 48.27 48.51 48.75', 'ELECTRICAL DATA (NMOT} 10.62 10.70 10.78 48.27 48.51 48.75', 'ELECTRICAL DATA (NMOT} 10.62 10.70 10.78 48.27 48.51 48.75', 'ELECTRICAL DATA (NMOT} 10.62 10.70 10.78 48.27 48.51 48.75', 'ELECTRICAL DATA (NMOT} 10.62 10.70 10.78 48.27 48.51 48.75'], ['Maximum Power-Pmax (Wp} 11.21 11.27 11.33', '4g', '424', '429', '435'], ['Maximum Power Voltage-Vmpp (V)', '3973', '3993', '40.15', '40.32'], ['Maximum Power Current-Impp (A)', '10.54', '', '', ''], ['Open Circuit Voltage-Voc (V}', '48.03', '', '', ''], ['Short Circuit Current-Isc (A)', '1.15', '', '', ''], ["NOCT Irradiation 800 W.'m?, ambient temperature 20, air mass 15,", 'wind speed 1 m/s', '', '', '']], "label": "horizontal"},
        {"table": [['TEMPERATURE & MAXIMUM RATINGS -0.35% / °C', 'TEMPERATURE & MAXIMUM RATINGS -0.35% / °C'], ['Nominal Module Operating Temperature (NMOT} -40°C~+85C', '4442°C'], ['Temperature Coefficient of VOC ISOOVDC', '-0.275% / °C'], ['Temperature Coefficient of ISC 25A', '0.045% / °C'], ['Temperature Coefficient of PMAX', ''], ['Operational Temperature', ''], ['Maximum System Voltage', ''], ['Max Series Fuse Rating', '']], "label": "vertical"},
        {"table": [['', '40 FT (HQ)'], ['Number of Modules Per Container', '620'], ['Number of Modules Per Pallet', '3]'], ['Number of Pallets Per Container', '']], "label": "horizontal"},
        {"table": [['Module type', 'TM -650 M-132 HC)', 'TM -650 M-132 HC)', '§=96TM-655M-132 HC]', '§=96TM-655M-132 HC]', '§=TM-660 M-132 HC', '§=TM-660 M-132 HC', 'TM-665M-132 HC', 'TM-665M-132 HC', 'TM -670 M-132 HC', 'TM -670 M-132 HC'], ['Testing condition', 'STC', 'NOTC', 'STC', 'NOTC', 'STC', 'NOTC', 'STC', 'NOTC', 'STC', 'NOTC'], ['Maximum Power (Pmax/w)', '650', '484', '655', '487', '660', '491', '665', '495', '670', '498'], ['Open Circuit Voltage (Voc/V)', '45', '42,6', '45,2', '42,8', '45,4', '43,0', '45.6', '43,1', '45,8', '43,3'], ['Short Circuit Current {(Isc/A)', '18,39', '14,41', '18,43', '14,43', '18,47', '14,48', '18,51', '14,57', '18,55', '14,58'], ['Voltage at Maximum Power (Vmp/V)', '37,6', '35,7', '37,8', '35,9', '38', '36,05', '38,2', '36,1', '38,4', '36,3'], ['Current at maximum Power {Imp/A)', '17,29', '13,56', '17,33', '13,57', '17,37', '13,62', '17,41', '13,71', '17,45', '13,72'], ['Module Efficiency (%)', '20,90%', '15,58%', '21,10%', '15.68%', '21,30%', '15,81%', '21.40%', '15,94%', '21,60%', '16,03%']], "label": "horizontal"},
        {"table": [['Temperatura ratings (STC)', ''], ['Temperature Coefficient of Isc', '+0.05%/°C'], ['Temperature Coefficient of Voc', '-0.27%/°C'], ['Temperature Coefficient of Pmax', '-0.35%/°C']], "label": "horizontal"},
        {"table": [['Packaging Configuration', ''], ['Packing Type', '40°HQ'], ['Piece/Pallet', '31']], "label": "horizontal"},
        {"table": [['Temperature Coefficient (Pm)', '-0.300%/℃'], ['Temperature Coefficient (Voc)', '-0.250%/℃'], ['Temperature Coefficient (Isc)', '0.045%/℃'], ['NMOT (Nominal Moudule Operating Temperature)', '41±3℃']], "label": "horizontal"},
        {"table": [['Transportation methods', 'Number of modules per cabinet', 'Number of modules per pallet'], ['40HQ container', '936 pcs', '36 pcs +36 pcs']], "label": "vertical"}
    ]
    evaluate_table_axis_function(test_data)
    print("evaluation complete")

if __name__ == "__main__":
    main()"""