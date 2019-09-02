git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet
cd mxnet
cp make/config.mk .
echo "USE_CUDA=1" >>config.mk
echo "USE_CUDNN=1" >>config.mk
echo "USE_BLAS=openblas" >>config.mk
#echo "ADD_CFLAGS += -I/usr/include/openblas" >>config.mk   #此步不用，因为我是装在/usr/local,环境变量中已配
echo "ADD_LDFLAGS += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs" >>config.mk

make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1