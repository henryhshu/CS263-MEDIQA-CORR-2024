import csv
import json
from pathlib import Path

input_path = "/home/heewon/workspaces/courses/cs263nlp-26w/MEDEC/MEDEC-MS/MEDEC-MS-TestSet-with-GroundTruth-and-ErrorType.csv"
output_path = "MEDEC-MS-TestSet-50samples-with-GroundTruth-and-ErrorType.csv"

# indices you want to keep
#target_indices = {0, 2}  # example

def load_indices(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))["indices"]

target_indices = set(load_indices("sampled_test_indices.json"))

with open(input_path, newline="", encoding="utf-8") as fin, \
     open(output_path, "w", newline="", encoding="utf-8") as fout:

    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)

    writer.writeheader()

    for row in reader:
        text_id = row["Text ID"]  # e.g., ms-test-2
        if text_id is None or len(text_id) == 0:
            continue
        print(text_id)
        idx = int(text_id.split("-")[-1])

        if idx in target_indices:
            writer.writerow(row)
