model_name = 'bert-base-cased'
cache_dir = '/home/wangye/Transformer/tf/onnxruntime/python/tools/transformers/cache_models'
from transformers import AutoConfig
config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
from transformers import TFBertForQuestionAnswering
model = TFBertForQuestionAnswering.from_pretrained(model_name, cache_dir=cache_dir)
model._saved_model_inputs_spec = None
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
example_inputs = tokenizer.encode_plus("This is a sample input", return_tensors="tf", max_length=512, pad_to_max_length=True, truncation=True)
example_outputs = model(example_inputs, training=False)
import keras2onnx
onnx_model = keras2onnx.convert_keras(model, model.name, target_opset=11)
onnx_model_path = '/home/wangye/Transformer/tf/onnxruntime/python/tools/transformers/onnx_models/mymodel.onnx'
keras2onnx.save_model(onnx_model, onnx_model_path)

from optimizer import optimize_model
opt_model = optimize_model(onnx_model_path,
                           'bert_keras',
                           num_heads=12,
                           hidden_size=768,
                           opt_level=0)
opt_model.use_dynamic_axes()
optimized_model_path = '/home/wangye/Transformer/tf/onnxruntime/python/tools/transformers/onnx_models/myoptimizedmodel.onnx'
opt_model.save_model_to_file(optimized_model_path)
from onnxruntime import SessionOptions, InferenceSession, GraphOptimizationLevel, __version__ as onnxruntime_version
sess_options = SessionOptions()
sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
execution_providers = ['CPUExecutionProvider']
session = InferenceSession(optimized_model_path, sess_options, providers=execution_providers)

def create_onnxruntime_input_tf(vocab_size, batch_size, sequence_length):
    import numpy
    input_ids = numpy.random.randint(low=0, high=vocab_size - 1, size=(batch_size, sequence_length), dtype=numpy.int32)
    inputs = {'input_ids': input_ids}
    attention_mask = numpy.ones([batch_size, sequence_length], dtype=numpy.int32)
    inputs['attention_mask'] = attention_mask
    segment_ids = numpy.zeros([batch_size, sequence_length], dtype=numpy.int32)
    inputs['token_type_ids'] = segment_ids
    return inputs

ort_inputs = create_onnxruntime_input_tf(28996, 1, 32)
session.run(None, ort_inputs)

