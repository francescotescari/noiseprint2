# noiseprint2

This module is lightweight, simple to use, porting of the original [noiseprint](https://github.com/grip-unina/noiseprint) repo using keras and tensorflow 2.

All the credits for noiseprint goes to the original authors at [GRIP UNINA](http://www.grip.unina.it/).

This porting can be useful to enable eager execution in tensorflow and compute the noiseprint at the same time.

Original repo: [https://github.com/grip-unina/noiseprint](https://github.com/grip-unina/noiseprint)

Original paper: [http://doi.org/10.1109/TIFS.2019.2916364](http://doi.org/10.1109/TIFS.2019.2916364)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install noiseprint2.

```bash
git clone https://github.com/francescotescari/noiseprint2.git
cd noiseprint2
pip install .
```

## Usage

If you want to run the sample script to generate the noiseprint for a given image (at IMAGE_PATH)
```bash
python sample.py [-h] [-q QUALITY] [--show] [-o OUTPUT_PATH] IMAGE_PATH
```

Access the module APIs:

```python
from noiseprint2 import NoiseprintEngine, gen_noiseprint, normalize_noiseprint

# How to compute noiseprint of a single image
noiseprint = gen_noiseprint(image path or image data, quality_level)
# Util function to normalize the noiseprint
noiseprint = normalize_noiseprint(noiseprint)

# How to compute the noiseprint of batches of images without reloading the weights each time:
engine = NoiseprintEngine()
engine.load_quality(56)
noiseprint1 = engine.predict(image1)
noiseprint2 = engine.predict(image2)
...
engine.load_quality(76)
noiseprint23 = engine.predict(image23)
noiseprint24 = engine.predict(image24)
...

```

## License
Please consider the original license at [https://github.com/grip-unina/noiseprint/blob/master/LICENSE.txt](https://github.com/grip-unina/noiseprint/blob/master/LICENSE.txt)
