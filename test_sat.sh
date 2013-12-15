cd /root/cuda-convnet/

python convnet.py --data-path=/root/tmp/batches --save-path=/root/tmp --test-range=1 --train-range=0 --layer-def=./example-layers/layers-conv-local-binary.cfg --layer-params=./example-layers/layer-params-conv-local-binary.cfg --data-provider=sat --test-freq=1 --epochs=2000

