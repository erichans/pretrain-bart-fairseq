from fairseq.models.bart import BARTModel
from fairseq.models.roberta import RobertaModel
import torch

def main():
    checkpoint_dir = 'bart/multirun/2022-11-02/17-00-54/0/checkpoints'
    checkpoint = 'checkpoint_best.pt'
    bart = BARTModel.from_pretrained(checkpoint_dir, checkpoint, bpe=None)
    # bart = BARTModel.from_pretrained('bart.large')
    # roberta = RobertaModel.from_pretrained('roberta.large')
    # roberta.eval()
    assert isinstance(bart.model, torch.nn.Module)
    bart.eval()

    # tokens = bart.encode('Eu quero que esse modelo funcione!')
    # assert tokens.tolist() == [0, 619, 1414, 10, 140, 630, 22777, 74, 2]
    # print(bart.decode(tokens)) 

    # tokens = bart.encode('BART is a <mask> model.', 'BART is not <mask>.')
    # print(roberta.fill_mask('BART is a <mask> model.'))
    # print(bart.fill_mask(['The first Star wars movie came out in <mask>.']))
    [print(item) for item in bart.fill_mask(['Brasília é a <mask> do Brasil.'])[0]]
    # print(bart.fill_mask(['The first Star wars movie came out in <mask>. It was <mask> movie!']))
    # print(roberta.fill_mask('BART is a <mask> model. BART is not <mask>.'))

if __name__ == '__main__':
    main()