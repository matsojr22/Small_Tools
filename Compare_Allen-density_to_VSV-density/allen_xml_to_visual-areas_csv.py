import os
import csv
import argparse
import xml.etree.ElementTree as ET

def extract_namespace(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    if '}' in root.tag:
        return {'ns': root.tag.split('}')[0].strip('{')}
    return {}

def infer_injection_hemisphere(hemisphere_data):
    scores = {}
    for h_id in ["1", "2"]:
        try:
            total_volume = sum(float(v) for v in hemisphere_data[h_id]["projection-volume"].values())
            total_intensity = sum(float(v) for v in hemisphere_data[h_id]["projection-intensity"].values())
            scores[h_id] = total_volume + 1e-9 * total_intensity
        except Exception as e:
            print(f"Error inferring hemisphere: {e}")
            scores[h_id] = 0

    if scores["1"] > scores["2"]:
        return "1"
    elif scores["2"] > scores["1"]:
        return "2"
    else:
        return "unknown"

def extract_projection_data(xml_file, areas_of_interest, additional_fields):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    filename = os.path.splitext(os.path.basename(xml_file))[0]

    hemisphere_data = {h_id: {field: {area: "0" for area in areas_of_interest.keys()} for field in ["projection-density"] + additional_fields} for h_id in ["1", "2", "3"]}
    injection_sites = []

    for projection in root.iter("projection-structure-unionize"):
        hemisphere_id = projection.find("hemisphere-id")
        structure_id = projection.find("structure-id")

        if hemisphere_id is None or structure_id is None:
            continue

        hemisphere_id = hemisphere_id.text
        structure_id = structure_id.text

        if hemisphere_id not in hemisphere_data:
            continue

        for area, s_id in areas_of_interest.items():
            if str(s_id) == structure_id:
                for field in hemisphere_data[hemisphere_id].keys():
                    field_value = projection.find(field)
                    hemisphere_data[hemisphere_id][field][area] = format(float(field_value.text), ".2E") if field_value is not None else "0"

    visp_ids = {"VISp": areas_of_interest["VISp"]}
    found_injection = False

    for injection in root.iter("projection-structure-unionize"):
        is_injection_elem = injection.find("is-injection")
        structure_id_elem = injection.find("structure-id")
        hemisphere_id_elem = injection.find("hemisphere-id")

        if is_injection_elem is None or structure_id_elem is None or hemisphere_id_elem is None:
            continue

        is_injection = is_injection_elem.text.strip().lower()
        structure_id = structure_id_elem.text.strip()
        hemisphere_id = hemisphere_id_elem.text.strip()

        if is_injection in ["true", "1", "yes"] and structure_id in visp_ids.values() and hemisphere_id in ["1", "2"]:
            print(f"Injection site found: File={filename}, Hemisphere={hemisphere_id}, Structure ID={structure_id}")
            injection_sites.append((filename, hemisphere_id, structure_id))
            found_injection = True

    if not found_injection:
        inferred_hemisphere = infer_injection_hemisphere(hemisphere_data)
        print(f"Inferred injection hemisphere for {filename}: {inferred_hemisphere}")
        injection_sites.append((filename, inferred_hemisphere, "inferred"))

    return filename, hemisphere_data, injection_sites

def process_xml_files(directory, output_directory, areas_of_interest, additional_fields):
    if not os.path.exists(directory):
        print(f"Error: Directory does not exist: {directory}")
        return

    xml_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".xml")]
    hemisphere_output_paths = {
        h_id: {field: os.path.join(output_directory, f"output_hemisphere_{h_id}_{field}.csv") for field in ["projection-density"] + additional_fields}
        for h_id in ["1", "2", "3"]
    }
    injection_output_path = os.path.join(output_directory, "output_injection_sites.csv")

    for h_id in hemisphere_output_paths:
        for field, path in hemisphere_output_paths[h_id].items():
            with open(path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["filename"] + list(areas_of_interest.keys()))

    with open(injection_output_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "hemisphere_id", "structure_id"])

    for xml_file in xml_files:
        filename, hemisphere_data, injection_sites = extract_projection_data(xml_file, areas_of_interest, additional_fields)

        for h_id in hemisphere_data:
            for field, path in hemisphere_output_paths[h_id].items():
                with open(path, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([filename] + [hemisphere_data[h_id][field][area] for area in areas_of_interest.keys()])

        if injection_sites:
            with open(injection_output_path, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(injection_sites)

    print("CSV files created:")
    for h_id in hemisphere_output_paths:
        for path in hemisphere_output_paths[h_id].values():
            print(path)
    print(injection_output_path)

areas_of_interest = {
    "VISal": 402, "VISam": 394, "VISl": 409, "VISp": 385, "VISpl": 425,
    "VISpm": 533, "VISli": 312782574, "VISpor": 312782628, "RSPagl": 894, "RSPd": 879,
    "RSPv": 886, "VISa": 312782546, "VISrl": 417
}

additional_fields = [
    "normalized-projection-volume", "projection-intensity", "projection-volume",
    "sum-pixel-intensity", "sum-pixels", "sum-projection-pixel-intensity",
    "sum-projection-pixels", "volume"
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse Allen XML files to CSV and infer injection hemisphere")
    parser.add_argument("--input_dir", required=True, help="Directory containing XML files")
    args = parser.parse_args()

    input_directory = args.input_dir
    output_directory = input_directory

    process_xml_files(input_directory, output_directory, areas_of_interest, additional_fields)
