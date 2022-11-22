fairseq-preprocess \
    --only-source \
    --trainpref brwac/train.bpe \
    --validpref brwac/valid.bpe \
    --testpref brwac/test.bpe \
    --destdir data-bin/brwac \
    --workers 60
