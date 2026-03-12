import metapredict as meta
import protfasta
import csv

metapredict_version = "v3"

file = "uniprotkb_AND_reviewed_true_AND_model_o_2025_12_17.fasta"
input_seqs = protfasta.read_fasta(file, expect_unique_header=False, return_list=True, invalid_sequence_action='convert')

clean_seqs = {}
idx2name = {}

for idx, s in enumerate(input_seqs):
    name = s[0]
    seq = s[1]
    name = name.replace(',', ';')
    idx2name[idx] = name
    clean_seqs[idx] = seq

batch_out = meta.predict_disorder_batch(clean_seqs, version=metapredict_version, return_domains=True)

# Save comprehensive data
outstring = f'disorder_complete_metapredict{metapredict_version}_Dec17.csv'

with open(outstring, 'w', newline='') as fh:
    writer = csv.writer(fh)
    writer.writerow(['idx', 'name', 'sequence_length', 'num_folded_domains', 'num_disordered_domains',
                     'folded_boundaries', 'disordered_boundaries', 'folded_domains', 'disordered_domains'])
    
    for idx in batch_out:
        disorder_obj = batch_out[idx]
        name = idx2name[idx]
        
        writer.writerow([
            idx,
            name,
            len(disorder_obj.sequence),
            len(disorder_obj.folded_domains),
            len(disorder_obj.disordered_domains),
            disorder_obj.folded_domain_boundaries,
            disorder_obj.disordered_domain_boundaries,
            disorder_obj.folded_domains,
            disorder_obj.disordered_domains
        ])
    
    print(f"Saved complete data to {outstring}")