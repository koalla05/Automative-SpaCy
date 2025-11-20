from normalization import equipment_normalization, manufacturer_normalization
import csv

all_equipment_variants = {v for variants in equipment_normalization.values() for v in variants}
all_manufacturer_variants = {v for variants in manufacturer_normalization.values() for v in variants}

new_eq = set()
new_man = set()

def clean_string(s):
    return s.strip(' "')

with open("../data/atmosfera_files.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)  # handles quoted commas correctly
    for line in reader:
        if len(line) < 4:
            continue  # skip incomplete rows

        eq = clean_string(line[2])
        man = clean_string(line[3])

        if eq.lower() not in all_equipment_variants and eq != "":
            new_eq.add(eq)
        if man.lower() not in all_manufacturer_variants and man != "":
            new_man.add(man)

new_eq.discard("")
new_man.discard("")

print("New equipment variants:", new_eq)
print("New manufacturer variants:", new_man)

def generate_suggestions(cleaned_set, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for item in sorted(cleaned_set):
            f.write(f'"{item}",\n')

generate_suggestions(new_eq, "equipment_suggestions.txt")
generate_suggestions(new_man, "manufacturer_suggestions.txt")