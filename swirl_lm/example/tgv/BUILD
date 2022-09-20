load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary")

pytype_strict_binary(
    name = "main",
    srcs = ["main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/matplotlib",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/base:driver",
        "//third_party/py/swirl_lm/base:driver_tpu",
        "//third_party/py/swirl_lm/base:initializer",
        "//third_party/py/swirl_lm/base:parameters",
        "//third_party/py/swirl_lm/utility:tpu_util",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow:tensorflow_google",  # build_cleaner: keep
    ],
)
