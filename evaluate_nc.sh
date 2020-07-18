#!/bin/bash


e=5
exp=nc_experiment

for dataset in  europe usa brazil
do
     for dim in  128 
     do
            edgelist=data/${dataset}-airports.edgelist
            labels=data/labels-${dataset}-airports.txt
            embedding_dir=embeddings/${dataset}

            test_results=$(printf "test_results/${dataset}/${exp}/dim=%03d/" ${dim})
            embedding_f=$(printf "${embedding_dir}/%05d_embedding.csv.gz"  ${e})
            echo $embedding_f
            
            args=$(echo --edgelist ${edgelist} --embedding ${embedding_dir} --dim ${dim}  -e ${e}  )  
            python main.py ${args}	
            
            args=$(echo --edgelist ${edgelist} --labels ${labels} \
                --dist_fn hyperboloid \
                --embedding ${embedding_f}  \
                --test-results-dir ${test_results})
            python evaluate_svm_nc.py ${args}

done
done