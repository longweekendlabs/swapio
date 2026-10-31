"""Pinned, separately downloaded source-guided rendering models."""

SOURCE_COMPONENT = "Source-guided face rendering"
SOURCE_MODEL_RELEASE = "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/"
SOURCE_MODELS = {
    "appearance": ("live_portrait_feature_extractor.onnx", 3354251,
                   "980d26afc9af6d1b6c946329df08b89a4a8582c9bb668d9475f5254e69434d15"),
    "motion": ("live_portrait_motion_extractor.onnx", 112636642,
               "ef4a00ff50d5d864994eeecb226dc3b298e62afacec8fa4fc8b912835ec4b0ab"),
    "generator": ("live_portrait_generator.onnx", 222263347,
                  "db9274114fed504226c5921fc736b9558505e68682e7448631aea7b62cb3048a"),
    "eyes": ("live_portrait_eye_retargeter.onnx", 581421,
             "13cf8b06f0a314e6e65b4a47fea5ce270edb79cd54359c4d273f2c139820276b"),
    "lips": ("live_portrait_lip_retargeter.onnx", 151022,
             "0f5b76344d0bfbb44c8a0b1ec57a4a7c36fd38db530338777a48c4a33a02d71f"),
}
