import sys
import struct
import json
import torch
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer

if len(sys.argv) < 3:
    print("Usage: convert-h5-to-ggml.py dir-model [use-f32]\n")
    print("  ftype == 0 -> float32")
    print("  ftype == 1 -> float16")
    sys.exit(1)
    
# ftype
ftype = 1
if len(sys.argv) > 2:
    ftype = int(sys.argv[2])
    
# possible data types
#   ftype == 0 -> float32
#   ftype == 1 -> float16
#
# map from ftype to string
ftype_str = ["f32", "f16"]

# h5 model is the directory that contains the tokenizer, config, weights, etc
dir_model = sys.argv[1]

with open(dir_model + "/tokenizer.json", "r", encoding="utf-8") as f:
    encoder = json.load(f)

with open(dir_model + "/config.json", "r", encoding="utf-8") as f:
    hparams = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(dir_model)
model = AutoModelForCausalLM.from_pretrained(dir_model, low_cpu_mem_usage=True)
#print (model)

#print(tokenizer.encode('I believe the meaning of life is'))

list_vars = model.state_dict()
for name in list_vars.keys():
    print(name, list_vars[name].shape, list_vars[name].dtype)
    
print(hparams)

# create new model in same directory as model, not within the old model
# it is reasonable to think many people will remove the h5 model once completed so no move necessary
# TODO: Create arg for in model (e.g. dir_model/ggml-model-f16.bin) or parent dir (e.g. ggml-{model_name}-f16.bin}
fn_bin = f"ggml-{dir_model.split('/')[-1]}-{ftype_str[ftype]}.bin"
fn_out = dir_model + "/../" + fn_bin
fout = open(fn_out, "wb")

ggml_file_magic = 0x67676d66 # 0x67676d6c is unversioned
ggml_file_version = 0x00000001 # v1

hparams["multiple_of"] = 1
fout.write(struct.pack("i", ggml_file_magic)) # magic: ggmf in hex
fout.write(struct.pack("i", ggml_file_version))
fout.write(struct.pack("i", hparams["vocab_size"]))
fout.write(struct.pack("i", hparams["max_position_embeddings"]))
fout.write(struct.pack("i", hparams["hidden_size"]))
fout.write(struct.pack("i", hparams["num_attention_heads"]))
fout.write(struct.pack("i", hparams["num_hidden_layers"]))
fout.write(struct.pack("i", int(hparams["rotary_pct"]*(hparams["hidden_size"]//hparams["num_attention_heads"]))))
fout.write(struct.pack("i", int(hparams["use_parallel_residual"])))
fout.write(struct.pack("i", ftype))

# TODO: temporary hack to not deal with implementing the tokenizer
dot_token = tokenizer.encode('.')[0]
for i in range(hparams["vocab_size"]):
    text = tokenizer.decode([dot_token, i]).encode('utf-8')
    # remove the first byte (it's always '.')
    text = text[1:]
    fout.write(struct.pack("i", len(text)))
    fout.write(text)

for name in list_vars.keys():
    data = list_vars[name].squeeze().numpy()
    print("Processing variable: " + name + " with shape: ", data.shape)

    # we don't need these
    if name.endswith(".attention.masked_bias") or     \
       name.endswith(".attention.bias") or \
       name.endswith(".attention.rotary_emb.inv_freq"):
        print("  Skipping variable: " + name)
        continue

    n_dims = len(data.shape);

    # ftype == 0 -> float32, ftype == 1 -> float16
    ftype_cur = 0;
    if ftype != 0:
        if name[-7:] == ".weight" and n_dims == 2:
            print("  Converting to float16")
            data = data.astype(np.float16)
            ftype_cur = 1
        else:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0
    else:
        if data.dtype != np.float32:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0

    # header
    str = name.encode('utf-8')
    fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    fout.write(str);

    # data
    data.tofile(fout)

fout.close()

print("Done. Output file: " + fn_out)
print("")
