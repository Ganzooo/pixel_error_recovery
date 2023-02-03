#Local PC
python train_recover_pixel.py dataset=train_grb_all train_config.description=rec_ep5000 train_config.epochs=5000 train_config.gpu_id=1 train_config.batch_size=32

#Server PC
#python train_recover_pixel.py dataset=server_train_rgb_all train_config.description=rec_server_ep5000 train_config.epochs=5000 train_config.gpu_id=1 train_config.batch_size=256
