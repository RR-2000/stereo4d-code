#!/bin/bash
echo "Running camera experiment script"

# echo "Running Kubric"



# DIR=/project/Thesis/kubric-private/output/multiview_36_v3/train/

# for j in 2; do

#     python track_kubric.py --num_views=20 --dir=$DIR$j

#     for i in 20; do
#         echo "Running segmentation and optimization for $i views in directory $DIR$j"
#         python segmentation_kubric.py --depth=gt --num_views=$i --dir=$DIR$j

#         python track_optimization_kubric.py --optimize=True --depth=gt --exp=gt_no_reg_$i --reg_type=none --num_views=$i --dir=$DIR$j
#         python track_optimization_kubric.py --optimize=True --depth=gt --exp=gt_depth_reg_$i --reg_type=depth --num_views=$i --dir=$DIR$j
#         python track_optimization_kubric.py --optimize=True --depth=gt --exp=gt_full_reg_$i --reg_type=full --num_views=$i --dir=$DIR$j

#         python segmentation_kubric.py --depth=dust3r --num_views=$i --dir=$DIR$j

#         python track_optimization_kubric.py --optimize=True --depth=dust3r --exp=dust3r_no_reg_$i --reg_type=none --num_views=$i --dir=$DIR$j
#         python track_optimization_kubric.py --optimize=True --depth=dust3r --exp=dust3r_depth_reg_$i --reg_type=depth --num_views=$i --dir=$DIR$j
#         python track_optimization_kubric.py --optimize=True --depth=dust3r --exp=dust3r_full_reg_$i --reg_type=full --num_views=$i --dir=$DIR$j

#         python segmentation_kubric.py --depth=vggt --num_views=$i --dir=$DIR$j

#         python track_optimization_kubric.py --optimize=True --depth=vggt --exp=vggt_no_reg_$i --reg_type=none --num_views=$i --dir=$DIR$j
#         python track_optimization_kubric.py --optimize=True --depth=vggt --exp=vggt_depth_reg_$i --reg_type=depth --num_views=$i --dir=$DIR$j
#         python track_optimization_kubric.py --optimize=True --depth=vggt --exp=vggt_full_reg_$i --reg_type=full --num_views=$i --dir=$DIR$j
#     done

# done

DIR=/project/Thesis/data/dexycb/

echo "Running DexYCB"



for j in subject-02 subject-03 subject-04 subject-05 subject-06 subject-07 subject-08 subject-09 subject-10; do
    echo "Running segmentation and optimization for $j"

    # python dexycb_scripts/track_dexycb.py --dir=$DIR$j

    for i in 4 6 8; do
        # python dexycb_scripts/segmentation_dexycb.py --depth=gt --num_views=$i --dir=$DIR$j

        # python dexycb_scripts/track_optimization_dexycb.py --optimize=True --depth=gt --exp=gt_no_reg_$i --reg_type=none --num_views=$i --dir=$DIR$j
        # python dexycb_scripts/track_optimization_dexycb.py --optimize=True --depth=gt --exp=gt_depth_reg_$i --reg_type=depth --num_views=$i --dir=$DIR$j
        # python dexycb_scripts/track_optimization_dexycb.py --optimize=True --depth=gt --exp=gt_full_reg_$i --reg_type=full --num_views=$i --dir=$DIR$j


        # python dexycb_scripts/segmentation_dexycb.py --depth=dust3r --num_views=$i --dir=$DIR$j

        # python dexycb_scripts/track_optimization_dexycb.py --optimize=True --depth=dust3r --exp=dust3r_no_reg_$i --reg_type=none --num_views=$i --dir=$DIR$j
        # python dexycb_scripts/track_optimization_dexycb.py --optimize=True --depth=dust3r --exp=dust3r_depth_reg_$i --reg_type=depth --num_views=$i --dir=$DIR$j
        # python dexycb_scripts/track_optimization_dexycb.py --optimize=True --depth=dust3r --exp=dust3r_full_reg_$i --reg_type=full --num_views=$i --dir=$DIR$j


        python dexycb_scripts/segmentation_dexycb.py --depth=vggt --num_views=$i --dir=$DIR$j

        python dexycb_scripts/track_optimization_dexycb.py --optimize=True --depth=vggt --exp=vggt_no_reg_$i --reg_type=none --num_views=$i --dir=$DIR$j
        python dexycb_scripts/track_optimization_dexycb.py --optimize=True --depth=vggt --exp=vggt_depth_reg_$i --reg_type=depth --num_views=$i --dir=$DIR$j
        python dexycb_scripts/track_optimization_dexycb.py --optimize=True --depth=vggt --exp=vggt_full_reg_$i --reg_type=full --num_views=$i --dir=$DIR$j
    done
done