python utils/embedding_copy.py "pkl:cat $1/asvspoof.$2/xvector.*.pkl |" pkl,t:- | cut -f1,2 -d" " | awk '{print $1" "$1" "$2}' | sort  > score.txt
python utils/compute_eer.py data/ASVSpoof5_dev/trials.sort.v2 score.txt
python utils/compute_dcf.py --c-fa=10 --c-miss=1 --p-target=0.05 data/ASVSpoof5_dev/trials.sort.v2 score.txt
python utils/compute_cllr.py data/ASVSpoof5_dev/trials.sort.v2 score.txt
python utils/compute_auc.py data/ASVSpoof5_dev/trials.sort.v2 score.txt

echo "-----"
