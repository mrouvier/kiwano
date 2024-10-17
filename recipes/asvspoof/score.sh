
echo "ASVSpoof5"
t=$1/asvspoof.$2/
python utils/embedding_copy.py "pkl:cat $1/asvspoof.$2/xvector.*.pkl |" pkl,t:- | cut -f1,2 -d" " | awk '{print $1" "$1" "$2}' | sort  > $t/score.txt
python utils/compute_eer.py data/ASVSpoof5_dev/trials.sort.v2 $t/score.txt
python utils/compute_dcf.py --c-fa=10 --c-miss=1 --p-target=0.05 data/ASVSpoof5_dev/trials.sort.v2 $t/score.txt
python utils/compute_cllr.py data/ASVSpoof5_dev/trials.sort.v2 $t/score.txt
python utils/compute_auc.py data/ASVSpoof5_dev/trials.sort.v2 $t/score.txt

echo "-----"

echo "Latin American"
t=$1/latin_american.$2/
python utils/embedding_copy.py "pkl:cat $1/latin_american.$2/xvector.*.pkl |" pkl,t:- | cut -f1,2 -d" " | awk '{print $1" "$1" "$2}' | sort  > $t/score.txt
python utils/compute_eer.py data/latin_american/trials $t/score.txt
python utils/compute_dcf.py --c-fa=10 --c-miss=1 --p-target=0.05 data/latin_american/trials $t/score.txt
python utils/compute_cllr.py data/latin_american/trials $t/score.txt
python utils/compute_auc.py data/latin_american/trials $t/score.txt

echo "-----"

echo "ITW"
t=$1/itw.$2/
python utils/embedding_copy.py "pkl:cat $1/itw.$2/xvector.*.pkl |" pkl,t:- | cut -f1,2 -d" " | awk '{print $1" "$1" "$2}' | sort  > $t/score.txt
python utils/compute_eer.py data/itw/trials $t/score.txt
python utils/compute_dcf.py --c-fa=10 --c-miss=1 --p-target=0.05 data/itw/trials $t/score.txt
python utils/compute_cllr.py data/itw/trials $t/score.txt
python utils/compute_auc.py data/itw/trials $t/score.txt

echo "-----"


echo "CFAD Noisy"
t=$1/cfad_noisy.$2/
python utils/embedding_copy.py "pkl:cat $1/cfad_noisy.$2/xvector.*.pkl |" pkl,t:- | cut -f1,2 -d" " | awk '{print $1" "$1" "$2}' | sort  > $t/score.txt
python utils/compute_eer.py data/cfad_noisy_version/trials $t/score.txt
python utils/compute_dcf.py --c-fa=10 --c-miss=1 --p-target=0.05 data/cfad_noisy_version/trials $t/score.txt
python utils/compute_cllr.py data/cfad_noisy_version/trials $t/score.txt
python utils/compute_auc.py data/cfad_noisy_version/trials $t/score.txt

echo "-----"



echo "ALL"
cat $1/asvspoof.$2/score.txt $1/latin_american.$2/score.txt $1/itw.$2/score.txt $1/cfad_noisy.$2/score.txt | sort > $1/score_$2.txt

python utils/compute_eer.py data/all_airbus/trials $1/score_$2.txt
python utils/compute_dcf.py --c-fa=10 --c-miss=1 --p-target=0.05 data/all_airbus/trials $1/score_$2.txt
python utils/compute_cllr.py data/all_airbus/trials $1/score_$2.txt
python utils/compute_auc.py data/all_airbus/trials $1/score_$2.txt


#echo "-----"

#echo "DEFI"
#t=$1/defi.$2/
#python utils/embedding_copy.py "pkl:cat $1/defi.$2/xvector.*.pkl |" pkl,t:- | cut -f1,2 -d" " | awk '{print $1" "$1" "$2}' | sort  > $t/score.txt


