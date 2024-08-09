#dir=exp/resnet_v2h_perturb_speed/
dir=$1
#dir=exp/resnet_v2h_jeffrey_subcenterv2/
#dir=exp/resnet_v2h_jeffrey_positional/
#dir=exp/resnet_v2h_crop200/
#dir=exp/resnet_v2h_perturb_speed_jeffrey/
#dir=exp/resnet_v2h_jeffrey_fbank96/
#exp/resnet_v2h/


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
