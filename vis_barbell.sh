#!/bin/bash


e=5
exp=nc_experiment

for dataset in barbell
do
     for dim in  2 
     do
            edgelist=data/barbell.edgelist
            embedding_dir=embeddings/${dataset}

            test_results=$(printf "test_results/${dataset}/${exp}/dim=%03d/" ${dim})
            embedding_f=$(printf "${embedding_dir}/%05d_embedding.csv.gz"  ${e})
            echo $embedding_f
            
            args=$(echo --edgelist ${edgelist} --embedding ${embedding_dir} --dim ${dim}  -e ${e}  )  
            python vis_barbell.py ${args}	
          
done
done