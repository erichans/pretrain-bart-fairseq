for SPLIT in train valid; do \
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json gpt2_bpe/encoder.json \
        --vocab-bpe gpt2_bpe/vocab.bpe \
        --inputs tcu_jur_full/${SPLIT}.txt \
        --outputs tcu_jur_full/${SPLIT}.bpe \
        --keep-empty \
        --workers 60; \
done
