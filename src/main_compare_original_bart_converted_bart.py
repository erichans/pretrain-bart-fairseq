# based on https://github.com/allenai/longformer/blob/master/scripts/convert_bart_to_longformerencoderdecoder.py

import torch

from transformers import BartForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right
from tqdm import tqdm


def create_long_model(
    base_model
):
    bart = BartForConditionalGeneration.from_pretrained(base_model)
    bart_original = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

    compare_weights(bart, bart_original)

def compare_weights(bart_converted, bart_original):
    bart_converted_parameters = dict(bart_converted.named_parameters())
    bart_original_parameters = dict(bart_original.named_parameters())

    parameters_missing_comparison = list(bart_original_parameters.keys())

    assert len(bart_converted_parameters.keys()) == len(bart_original_parameters.keys())
    assert sorted(bart_converted_parameters.keys()) == sorted(bart_original_parameters.keys())

    for bart_converted_parameter, bart_original_parameter in zip(bart_converted_parameters, bart_original_parameters):
        assert bart_converted_parameter == bart_original_parameter
        assert torch.all(bart_converted_parameters[bart_converted_parameter] == bart_original_parameters[bart_original_parameter])
        parameters_missing_comparison.remove(bart_converted_parameter)

    assert len(parameters_missing_comparison) == 0
    print('Modelos iguais!', parameters_missing_comparison)

def main():
    # checkpoint = 29618
    # base_model = f'saved_models/checkpoint-{checkpoint}'
    # base_model = 'facebook/bart-large'
    # base_model = './tmp'

    create_long_model(
        base_model=base_model
    )

if __name__ == '__main__':
    main()