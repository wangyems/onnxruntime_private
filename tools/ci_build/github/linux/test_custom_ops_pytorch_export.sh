
pip3 install --user --upgrade pip

pip3 install --user numpy==1.19.0 torch pytest
pip3 install --user /build/Release/dist/*.whl

export PYTHONPATH=/onnxruntime_src/tools:/usr/local/lib/python3.6/site-packages:$PYTHONPATH

python3 -m pytest -v /onnxruntime_src/tools/test/test_custom_ops_pytorch_exporter.py || exit 1

cd /build/Release && python3 -m pytest -v /onnxruntime_src/onnxruntime/test/python/onnx_test_trilu.py

cd /build/Release && ./onnxruntime_test_all --gtest_filter=ShapeInferenceTests.*
