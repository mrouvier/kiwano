dir=$1

echo "voxceleb1-o"
python utils/compute_cosine.py  data/voxceleb1/voxceleb1-o-cleaned.trials  "pkl:cat $dir/voxceleb1.${2}/xvector.*.pkl |"   "pkl:cat $dir/voxceleb1.${2}/xvector.*.pkl |" > $dir/voxceleb1.${2}/scores.txt
python utils/compute_eer.py data/voxceleb1/voxceleb1-o-cleaned.trials $dir/voxceleb1.${2}/scores.txt
python utils/compute_dcf.py data/voxceleb1/voxceleb1-o-cleaned.trials $dir/voxceleb1.${2}/scores.txt
python utils/compute_cllr.py data/voxceleb1/voxceleb1-o-cleaned.trials $dir/voxceleb1.${2}/scores.txt
echo -e "\n"


echo "voxceleb1-e"
python utils/compute_cosine.py  data/voxceleb1/voxceleb1-e-cleaned.trials  "pkl:cat $dir/voxceleb1.${2}/xvector.*.pkl |"   "pkl:cat $dir/voxceleb1.${2}/xvector.*.pkl |" > $dir/voxceleb1.${2}/scores.txt
python utils/compute_eer.py data/voxceleb1/voxceleb1-e-cleaned.trials $dir/voxceleb1.${2}/scores.txt
python utils/compute_dcf.py data/voxceleb1/voxceleb1-e-cleaned.trials $dir/voxceleb1.${2}/scores.txt
python utils/compute_cllr.py data/voxceleb1/voxceleb1-e-cleaned.trials $dir/voxceleb1.${2}/scores.txt
echo -e "\n"


echo "voxceleb1-h"
python utils/compute_cosine.py  data/voxceleb1/voxceleb1-h-cleaned.trials  "pkl:cat $dir/voxceleb1.${2}/xvector.*.pkl |"   "pkl:cat $dir/voxceleb1.${2}/xvector.*.pkl |" > $dir/voxceleb1.${2}/scores.txt
python utils/compute_eer.py data/voxceleb1/voxceleb1-h-cleaned.trials $dir/voxceleb1.${2}/scores.txt
python utils/compute_dcf.py data/voxceleb1/voxceleb1-h-cleaned.trials $dir/voxceleb1.${2}/scores.txt
python utils/compute_cllr.py data/voxceleb1/voxceleb1-h-cleaned.trials $dir/voxceleb1.${2}/scores.txt
echo -e "\n"
