#!/bin/bash


# ROUND MIX
# nothing is frozne.  a mix of cold to warm.  base_lr: 0.001
#caffe train --solver=solver.prototxt --weights=bvlc_reference_caffenet.caffemodel
#caffe train --solver=solver.prototxt --weights=net1_MIX_iter_1000.caffemodel

# ROUND A
# all but (myfc8) were frozen.  base_lr: 0.001
# accuracy = 0.942708, loss = 0.177642
#caffe train --solver=solver.prototxt --weights=bvlc_reference_caffenet.caffemodel

# ROUND B
# all but (myfc8, fc7) were frozen.  base_lr: 0.001
caffe train --solver=solver.prototxt --weights=net1_A_iter_1000.caffemodel

# ROUND C
# all but (myfc8, fc7, fc6) were frozen.
# 20k iters, [40k to 60k], val loss ~1.40
#caffe train --solver=solver.prototxt --weights=net1_iter_30000.caffemodel

# ROUND D
# nothing is frozen.
# 40k iters, [60k to 100k], 
# and i doubled the train batch size from 64 to 128.
# val loss got down to ~1.13 halfway through, 
# then rising to ~1.16.
#caffe train --solver=solver.prototxt --weights=net1_iter_20000_60ktotal.caffemodel
