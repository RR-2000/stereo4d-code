#!/bin/bash
echo "Running camera experiment script"

echo "Running Kubric"

# python track_kubric.py

# for i in 4 6 8 20; do
#     # python segmentation_kubric.py --depth=gt --num_views=$i

#     # python track_optimization_kubric.py --optimize=True --depth=gt --exp=gt_no_reg_$i --reg_type=none --num_views=$i
#     # python track_optimization_kubric.py --optimize=True --depth=gt --exp=gt_depth_reg_$i --reg_type=depth --num_views=$i
#     # python track_optimization_kubric.py --optimize=True --depth=gt --exp=gt_full_reg_$i --reg_type=full --num_views=$i

#     # python segmentation_kubric.py --depth=dust3r --num_views=$i

#     # python track_optimization_kubric.py --optimize=True --depth=dust3r --exp=dust3r_no_reg_$i --reg_type=none --num_views=$i
#     # python track_optimization_kubric.py --optimize=True --depth=dust3r --exp=dust3r_depth_reg_$i --reg_type=depth --num_views=$i
#     # python track_optimization_kubric.py --optimize=True --depth=dust3r --exp=dust3r_full_reg_$i --reg_type=full --num_views=$i

#     # python segmentation_kubric.py --depth=vggt --num_views=$i

#     # python track_optimization_kubric.py --optimize=True --depth=vggt --exp=vggt_no_reg_$i --reg_type=none --num_views=$i
#     # python track_optimization_kubric.py --optimize=True --depth=vggt --exp=vggt_depth_reg_$i --reg_type=depth --num_views=$i
#     # python track_optimization_kubric.py --optimize=True --depth=vggt --exp=vggt_full_reg_$i --reg_type=full --num_views=$i
# done

echo "Running DexYCB"

# python dexycb_scripts/segmentation_dexycb.py

for i in 6; do
    # python dexycb_scripts/segmentation_dexycb.py --depth=gt --num_views=$i

    # python dexycb_scripts/track_optimization_dexycb.py --optimize=True --depth=gt --exp=gt_no_reg_$i --reg_type=none --num_views=$i
    # python dexycb_scripts/track_optimization_dexycb.py --optimize=True --depth=gt --exp=gt_depth_reg_$i --reg_type=depth --num_views=$i
    # python dexycb_scripts/track_optimization_dexycb.py --optimize=True --depth=gt --exp=gt_full_reg_$i --reg_type=full --num_views=$i


    python dexycb_scripts/segmentation_dexycb.py --depth=dust3r --num_views=$i

    python dexycb_scripts/track_optimization_dexycb.py --optimize=True --depth=dust3r --exp=dust3r_no_reg_$i --reg_type=none --num_views=$i
    python dexycb_scripts/track_optimization_dexycb.py --optimize=True --depth=dust3r --exp=dust3r_depth_reg_$i --reg_type=depth --num_views=$i
    python dexycb_scripts/track_optimization_dexycb.py --optimize=True --depth=dust3r --exp=dust3r_full_reg_$i --reg_type=full --num_views=$i


    # python dexycb_scripts/segmentation_dexycb.py --depth=vggt --num_views=$i

    # python dexycb_scripts/track_optimization_dexycb.py --optimize=True --depth=vggt --exp=vggt_no_reg_$i --reg_type=none --num_views=$i
    # python dexycb_scripts/track_optimization_dexycb.py --optimize=True --depth=vggt --exp=vggt_depth_reg_$i --reg_type=depth --num_views=$i
    # python dexycb_scripts/track_optimization_dexycb.py --optimize=True --depth=vggt --exp=vggt_full_reg_$i --reg_type=full --num_views=$i
done