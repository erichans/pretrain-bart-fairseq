from fairseq.models.bart import BARTModel
from fairseq.models.roberta import RobertaModel
import torch

def main():
    checkpoint_dir = 'bart/outputs/2022-11-12/12-23-46/checkpoints/'
    checkpoint = 'checkpoint1.pt'
    bart = BARTModel.from_pretrained(checkpoint_dir, checkpoint, bpe=None)
    # bart = BARTModel.from_pretrained('bart.large')
    # roberta = RobertaModel.from_pretrained('roberta.large')
    # roberta.eval()
    assert isinstance(bart.model, torch.nn.Module)
    bart.eval()
    # bart.cuda()

    # tokens = bart.encode('Eu quero que esse modelo funcione!')
    # assert tokens.tolist() == [0, 619, 1414, 10, 140, 630, 22777, 74, 2]
    # print(bart.decode(tokens)) 

    exemplo = 'Este é um teste aleatório'
    print(exemplo)
    tokens = bart.encode(exemplo)
    print(tokens)
    assert bart.decode(tokens) == exemplo
    # print(roberta.fill_mask('BART is a <mask> model.'))
    # print(bart.fill_mask(['The first Star wars movie came out in <mask>.']))
    [print(item) for item in bart.fill_mask(['Brasília é a <mask> do Brasil.'], match_source_len=False)[0]]
    print()
    [print(item) for item in bart.fill_mask(['O Íbis é o pior time do <mask>.'], match_source_len=False)[0]]
    print()
    [print(item) for item in bart.fill_mask(['<mask> é o marido da rainha.'], match_source_len=False)[0]]
    print()
    
    [print(item) for item in bart.fill_mask(['The cat <mask> on the <mask>.'], topk=3, beam=10)[0]]
    
    

if __name__ == '__main__':
    main()