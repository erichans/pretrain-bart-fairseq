fairseq-preprocess \
    --only-source \
    --trainpref brwac-16384_1024/train.bpe \
    --validpref brwac-16384_1024/valid.bpe \
    --testpref brwac-16384_1024/test.bpe \
    --destdir data-bin/brwac-16384_1024 \
    --workers 60
