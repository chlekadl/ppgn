ROOT_URL="http://lmb.informatik.uni-freiburg.de/resources/binaries/arxiv2016_alexnet_inversion_with_gans/trained_models_deepsim"
for NET in caffenet norm1 norm2 conv3 conv4 pool5 fc6 fc6_eucl fc7 fc8; 
do
  if [ ! -d "${NET}" ]; then
    CURR_URL="${ROOT_URL}/${NET}.zip"
    echo "Downloading ${CURR_URL}"
    wget "${CURR_URL}"
    unzip "${NET}.zip"
    rm "${NET}.zip"
  else
    echo "${NET} already downloaded"
  fi
done
