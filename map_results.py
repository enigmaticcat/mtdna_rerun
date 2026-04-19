import os, csv
from collections import defaultdict

pipeline_dir = "/home/minhtq/mtDNA_proj/mtdna_rerun/pipeline_results"
sample_to_runs = defaultdict(list)

print("Scanning pipeline directories...")
runs = os.listdir(pipeline_dir)
for run in runs:
    run_path = os.path.join(pipeline_dir, run)
    if os.path.isdir(run_path):
        for sample in os.listdir(run_path):
            sample_to_runs[sample].append(run)

tsv_path = "/home/minhtq/mtDNA_proj/mtdna_rerun/metadata_rerun.tsv"
out_path = "/home/minhtq/mtDNA_proj/mtdna_rerun/metadata_rerun_mapped.tsv"

with open(tsv_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8", newline="") as fout:
    reader = csv.reader(fin, delimiter="\t")
    writer = csv.writer(fout, delimiter="\t")
    
    header = next(reader)
    header.append("Pipeline_Results")
    writer.writerow(header)
    
    for row in reader:
        if len(row) > 1:
            samp_id1 = row[0].strip()
            samp_id2 = row[1].strip()
            
            # Check 2nd column first, then 1st
            matches = sample_to_runs.get(samp_id2)
            if not matches and samp_id1 != samp_id2:
                matches = sample_to_runs.get(samp_id1)
                
            if matches:
                # Store relative paths
                paths = [f"{run}/{samp_id2}" for run in matches]
                row.append("; ".join(paths))
            else:
                row.append("NOT FOUND")
            writer.writerow(row)
        else:
            row.append("")
            writer.writerow(row)

print("Created metadata_rerun_mapped.tsv")
