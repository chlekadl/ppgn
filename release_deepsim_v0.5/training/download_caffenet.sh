URL="http://lmb.informatik.uni-freiburg.de/resources/binaries/arxiv2016_alexnet_inversion_with_gans/trained_models_deepsim/caffenet.zip"
if [ ! -d "../trained_models/caffenet" ]; then
  echo "Downloading ${URL}"
  wget "${URL}"
  unzip "caffenet.zip"
  rm "caffenet.zip"
  mv caffenet ../trained_models
else
  echo "caffenet already downloaded"
fi
