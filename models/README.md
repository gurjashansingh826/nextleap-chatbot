# Model assets — committed, not downloaded

This directory holds everything a deployed instance needs to embed text, so the app is fully
offline at runtime (and at build time). There is no HuggingFace download anywhere in the
deployed path.

## Files

- `encoder_model_int8.onnx` — the BERT encoder of `sentence-transformers/all-MiniLM-L6-v2`,
  exported to ONNX (`opset 14`, dynamic batch/sequence) and dynamic-quantised (int8 weights,
  fp32 activations). 23 MB instead of the ~91 MB fp32/safetensors original.
- `tokenizer.json`, `vocab.txt`, `tokenizer_config.json`, `special_tokens_map.json`,
  `config.json` — the tokenizer assets, read by `AutoTokenizer.from_pretrained(...)`.
- `1_Pooling/config.json`, `sentence_bert_config.json`, `modules.json` — the sentence-
  transformers pooling recipe, asserted by `embedder._verify_recipe()` so mean-pooling +
  L2-normalise is *verified from the model's own declaration*, never assumed.

## Why ONNX

The pre-ONNX implementation loaded the model through `transformers` from the HuggingFace Hub
on first use. On a server with no model cache, that first load was a blocking ~91 MB download
that happened again on every Render free-tier cold start — blanking the first paint of the
app. The committed ONNX encoder removes the network from the path entirely.

## Regenerating (only needed if the model ever changes)

```powershell
# 1. export the encoder (torch)
python - <<'PY'
import torch, pathlib
from transformers import AutoModel
src = pathlib.Path("<hf model snapshot dir>")
model = AutoModel.from_pretrained(str(src)); model.eval()
torch.onnx.export(model,
    (torch.randint(2,1000,(1,8)), torch.ones(1,8,dtype=torch.long), torch.zeros(1,8,dtype=torch.long)),
    "encoder_model.onnx",
    input_names=["input_ids","attention_mask","token_type_ids"], output_names=["last_hidden_state"],
    dynamic_axes={"input_ids":{0:"batch",1:"seq"},"attention_mask":{0:"batch",1:"seq"},
                  "token_type_ids":{0:"batch",1:"seq"},"last_hidden_state":{0:"batch",1:"seq"}},
    opset_version=14)
PY
# 2. quantise
python - <<'PY'
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("encoder_model.onnx", "models/all-MiniLM-L6-v2/encoder_model_int8.onnx",
                 weight_type=QuantType.QUInt8)
PY
# 3. copy the tokenizer + recipe JSONs from the same snapshot, then rebuild the index:
python -m mf_rag.cli embed --rebuild
python tools/answer_eval.py   # gate: fact/both must not regress vs the previous backend
```

The int8 export embeds >0.98 cosine-similar to the fp32 torch model; still, an index built
with one backend is not comparable vector-by-vector to the other — always rebuild (`--rebuild`)
after changing `MF_RAG_EMBED_BACKEND` or regenerating the ONNX file.

## Parity path

`MF_RAG_EMBED_BACKEND=torch` switches embedding to the original Hub-backed transformers path,
kept only for parity checks. Production default is `onnx`.