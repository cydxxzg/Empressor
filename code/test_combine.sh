python Compress.py  \
 -name CMV \
 -predictor ../model_zoo/DenseLSTM2_HepG2.pth \
 -generator ../model_zoo/WGAN_HepG2.pth \
 -enhancer_path /home/zjli/Empressor/data/enhancer/cmv_enhancer.txt \
 -motif_split /home/zjli/Empressor/data/enhancer/cmv_enhancer_motif.txt \
 -tag HepG2

