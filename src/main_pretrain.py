import sentencepiece as spm

def main():
    spm.SentencePieceTrainer.train(input='bart/brwac-tmp/*.txt', model_prefix='m', vocab_size=1000, user_defined_symbols=['foo', 'bar'])
    

if __name__ == '__main__':
    run()