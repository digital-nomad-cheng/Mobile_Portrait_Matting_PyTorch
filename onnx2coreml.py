import onnx
import coremltools
from onnx_coreml import convert
from coremltools.models import MLModel
# test
onnx_model = onnx.load('models/SegNet_portrait_epoch-0099_sim.onnx')
onnx.checker.check_model(onnx_model)
coreml_model = convert(onnx_model,
    image_input_names={'input.1'},
    preprocessing_args={"image_scale":1/255.})
spec = coreml_model.get_spec()
coremltools.utils.rename_feature(spec, 'input.1', 'input_1')
spec.neuralNetwork.preprocessing[0].featureName='input_1'
coreml_model = MLModel(spec)
coreml_model.save('SegNet.mlmodel')
