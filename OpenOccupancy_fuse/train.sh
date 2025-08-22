mkdir -p /root/worker/data/nuscenes
#将hds的路径挂到开发者的数据集路径
ln -s /data-engine/haomo-algorithms/release/algorithms/manual_created_cards/628c5f667844160dda20fcc8/samples /root/worker/data/nuscenes/
ln -s /data-engine/haomo-algorithms/release/algorithms/manual_created_cards/628c5f667844160dda20fcc8/sweeps /root/worker/data/nuscenes/
ln -s /data-engine/haomo-algorithms/nuscenes/lidarseg /root/worker/data/nuscenes/
ln -s /data-engine/haomo-algorithms/cf53a9ee45edb078191fcb683dac2e41/nuScenes-Occupancy /root/worker/data/
cp /share/lisy/data/nuscenes_new/nuscenes_occ_infos_train.pkl /root/worker/data/nuscenes/
cp /share/lisy/data/nuscenes_new/nuscenes_occ_infos_val.pkl /root/worker/data/nuscenes/
ln -s /share/lisy/mmdetection3d-0.17.1 /root/worker/mmdetection3d
ln -s /share/lisy/OpenOccupancy/pretrained /root/worker/
ln -s /share/lisy/OpenOccupancy/data/depth_gt /root/worker/data/

python setup.py develop

pip install \
      PyMCubes \
      fvcore \
      shapely==1.8.5 \
      numba==0.48.0 \
      cachetools==5.3.0

cd /root/worker/mmdetection3d
python setup.py install
cd /root/worker

# CONFIG=./projects/configs/baselines/Multimodal-R50_img1600_128x128x10_df_mdla.py
CONFIG=./projects/configs/Cascade-Occupancy-Network/Multimodal-R50_img1600_cascade_x4_deepfuse.py
# CONFIG=./projects/configs/Cascade-Occupancy-Network/Multimodal-R50_img1600_cascade_x4.py
PORT=${PORT:-28509}

_MASTER_ADDR=$MASTER_ADDR
_MASTER_PORT=$MASTER_PORT
unset MASTER_ADDR
unset MASTER_PORT
unset RANK
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.run \
      --nproc_per_node=${gpus} \
      --nnodes=${WORLD_SIZE} \
      --rdzv_id="training" \
      --rdzv_backend="c10d" \
      --rdzv_endpoint="${_MASTER_ADDR}:${_MASTER_PORT}" \
        $(dirname "$0")/tools/train.py \
      ${CONFIG} \
      --launcher pytorch \
