#! /usr/bin/env sh

# run `./timpute/figures/memUsage.sh > ./timpute/figures/memUsage.out`

if [ -e timpute/figures/revision_cache/dataUsage/memory.pickle ]
then
    rm timpute/figures/revision_cache/dataUsage/memory.pickle
fi

for a in {1..20}
do
    t="entry"
    for d in 0 0.05 0.1 0.2 0.3 0.4 0.5
    do
        for i in "zohar" "alter" "hms" "coh_response"
        do 
            for j in "DO" "ALS" "CLS"
            do
                poetry run python -m timpute.figures.memUsage --dataname $i --method $j --filename $d --dropType $t --dropPerc $d --seed $a
                echo $a ":" $i $j "for" $t "drop" $d "%"
            done
        done
    done
done
