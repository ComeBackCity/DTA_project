mkdir protT5 # root directory for storing checkpoints, results etc
mkdir protT5/protT5_checkpoint # directory holding the ProtT5 checkpoint
mkdir protT5/sec_struct_checkpoint # directory storing the supervised classifier's checkpoint
mkdir protT5/output # directory for storing your embeddings & predictions
wget -nc -P protT5/ https://rostlab.org/~deepppi/example_seqs.fasta
wget -nc -P protT5/protT5_checkpoint https://rostlab.org/~deepppi/protT5_xl_u50_encOnly_fp16_checkpoint/pytorch_model.bin
wget -nc -P protT5/protT5_checkpoint https://rostlab.org/~deepppi/protT5_xl_u50_encOnly_fp16_checkpoint/config.json
# Huge kudos to the bio_embeddings team here! We will integrate the new encoder, half-prec ProtT5 checkpoint soon
wget -nc -P protT5/sec_struct_checkpoint http://data.bioembeddings.com/public/embeddings/feature_models/t5/secstruct_checkpoint.pt