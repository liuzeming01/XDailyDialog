import os
import json
from transformers import MBartTokenizerFast, T5TokenizerFast
from transformers import MT5ForConditionalGeneration

# tokenizer = MBartTokenizerFast.from_pretrained('checkpoints/mbart-25')
tokenizer = T5TokenizerFast.from_pretrained('google/mt5-base')
# mo = MT5ForConditionalGeneration.from_pretrained('checkpoints/mt5-base')

dtypes = ['train', 'dev', 'test']
data_dir = './data/raw/'
save_dir = './data'
# datas = ['monolingual/En', 'monolingual/Zh', "monolingual/De", "monolingual/It"]
# datas = ['multilingual',]
# datas = ["crosslingual/De_En", "crosslingual/En_De", "crosslingual/Es_It", "crosslingual/It_Es", 
#          "crosslingual/It_Ro", "crosslingual/Ro_It", "crosslingual/De_Nl", "crosslingual/En_Zh",
#          "crosslingual/Fr_It", "crosslingual/It_Fr", "crosslingual/Nl_De", "crosslingual/Zh_En"]
import os
datas = []
for setting in os.listdir(data_dir):
    if "multilingual" not in setting:
        for task in os.listdir(data_dir + setting):
            datas.append(setting + "/" + task)
    else:
        datas.append(setting)
print("setttings: {}".format(datas))

for data in datas:
    print(data)
    for dtype in dtypes:
        src_lens, tgt_lens = [], []
        src_file, tgt_file = os.path.join(data_dir, data, dtype + '.src'), os.path.join(data_dir, data, dtype + '.tgt')
        src_lines, tgt_lines = open(src_file).readlines(), open(tgt_file).readlines()

        examples = []
        for src_line, tgt_line in zip(src_lines, tgt_lines):
            src, tgt = src_line.strip(), tgt_line.strip()
            # dic = {'src': src_line.split('</s>')[0], 'tgt': tgt_line.split('</s>')[0]}
            src_lens.append(len(tokenizer.tokenize(src)))
            tgt_lens.append(len(tokenizer.tokenize(tgt)))

            dic = {'src': src, 'tgt': tgt}
            examples.append(json.dumps(dic, ensure_ascii=False))

        out_dir = os.path.join(save_dir, data)
        os.makedirs(out_dir, exist_ok=True)

        out_file = os.path.join(out_dir, dtype + '.jsonl')
        with open(out_file, 'w') as f:
            f.write('\n'.join(examples))

        print(len(examples), dtype)

        src_lens.sort(reverse=True)
        tgt_lens.sort(reverse=True)
        print(src_lens[:100])
        print(tgt_lens[:100])