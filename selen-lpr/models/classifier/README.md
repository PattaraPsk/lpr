# Plate Design & Vehicle Type Classifier Weights

Place trained classifier model weights in this directory:

- `design_classifier.onnx` (Lightweight MobileNetV4 / ResNet18 palette classifier)
- `category_labels.json` (Indexed output class definitions)

When no neural classifier weights are present, `selen-lpr` uses the high-precision LAB-space pixel color cluster extractor + DLT consonant rule engine in `src/utils/`.
