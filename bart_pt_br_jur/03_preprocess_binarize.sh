fairseq-preprocess \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --trainpref tcu_jur_full/train.bpe \
    --validpref tcu_jur_full/valid.bpe \
    --destdir data-bin/tcu_jur_full \
    --workers 60
