mkdir -p /root/.cache/torch/hub/checkpoints/

# ln -s  /external_da/tf_efficientnet_lite4-741542c3.pth /root/.cache/torch/hub/checkpoints/
ln -s  /external_data/efficientnet-b7-dcc49843.pth /root/.cache/torch/hub/checkpoints/
# mkdir -p /home/ll/.cache/torch/hub/checkpoints/
# ln -s /home/ll/project/external_data/efficientnet-b7-dcc49843.pth /home/ll/.cache/torch/hub/checkpoints/

python make_dataset.py
python train.py
python train_2.py