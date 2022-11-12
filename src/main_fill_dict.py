

def main():
    # with open('roberta/data-bin/wikitext-103_original/dict.txt', 'r') as f:
    with open('bart/data-bin/brwac/dict.txt', 'r') as f:
        input_lines = f.readlines() #50220

    bpes = sorted([int(input_line.split(' ')[0]) for input_line in input_lines if input_line.split(' ')[0].isnumeric()]) #50213

    # 5 special tokens: <pad>, <unk>, <s>, </s>, <mask>
    # original dict size: 50260 (171 zero occurences) - generates 50265 tokens (5 more)
    # generated dict zie: 50220 (7 zero occurences) 
    # adding more 50260 - 50220 = 40 zero occurences to fill the gap - generates 50225 (5 more)

    missing = []
    index_start = 5
    for index, bpe in enumerate(bpes):
        while index + index_start != bpe: 
            print(f'Missing: {index + index_start}. Actual BPE: {bpe}')
            missing.append(index + index_start)
            index_start += 1
            
    [print(f'{bpe} 0') for bpe in missing[:40]]



if __name__ == '__main__':
    main()