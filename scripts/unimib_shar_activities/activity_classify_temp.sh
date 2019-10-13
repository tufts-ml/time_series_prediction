TSPATH="../../datasets/unimib_shar_activities/v20190307"
python ../../src/eval_classifier.py logistic \
    --ts_dir $TSPATH \
    --data_dict $TSPATH/activity_dict__collapsed.json \
    --static_files $TSPATH/metadata_per_seq.csv $TSPATH/metadata_per_subj.csv \
    --validation_size 0.1 \
    --scoring balanced_accuracy \
    --grid_C 0.001 0.01 0.1 1 10 100 1000 10000 100000 1000000