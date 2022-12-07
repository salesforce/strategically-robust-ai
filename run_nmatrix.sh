python batch_launcher.py \
--exp-config-path ./yaml_configs/coop_bimatrix_3/train_baselines.yaml \
--base-exp-config-path ./yaml_configs/coop_bimatrix_3/base.yaml \
--job-yaml-path   ./gcp_yaml_configs/base-intel-cpu-job.yaml

python batch_launcher.py \
--exp-config-path ./yaml_configs/coop_bimatrix_3/train_oracle.yaml \
--base-exp-config-path ./yaml_configs/coop_bimatrix_3/base.yaml \
--job-yaml-path   ./gcp_yaml_configs/base-intel-cpu-job.yaml

python batch_launcher.py \
--exp-config-path ./yaml_configs/coop_bimatrix_3/train_ermas.yaml \
--base-exp-config-path ./yaml_configs/coop_bimatrix_3/base.yaml \
--job-yaml-path   ./gcp_yaml_configs/base-intel-big-cpu-job.yaml

exit
python batch_launcher.py \
--exp-config-path ./yaml_configs/coop_bimatrix_3/eval_dr.yaml \
--base-exp-config-path ./yaml_configs/coop_bimatrix_3/base.yaml \
--job-yaml-path   ./gcp_yaml_configs/base-intel-cpu-job.yaml

python batch_launcher.py \
--exp-config-path ./yaml_configs/coop_bimatrix_3/eval_act_perturb.yaml \
--base-exp-config-path ./yaml_configs/coop_bimatrix_3/base.yaml \
--job-yaml-path   ./gcp_yaml_configs/base-intel-cpu-job.yaml

python batch_launcher.py \
--exp-config-path ./yaml_configs/coop_bimatrix_3/eval_concav.yaml \
--base-exp-config-path ./yaml_configs/coop_bimatrix_3/base.yaml \
--job-yaml-path   ./gcp_yaml_configs/base-intel-cpu-job.yaml

python batch_launcher.py \
--exp-config-path ./yaml_configs/coop_bimatrix_3/eval_perturb.yaml \
--base-exp-config-path ./yaml_configs/coop_bimatrix_3/base.yaml \
--job-yaml-path   ./gcp_yaml_configs/base-intel-cpu-job.yaml

python batch_launcher.py \
--exp-config-path ./yaml_configs/coop_bimatrix_3/val.yaml \
--base-exp-config-path ./yaml_configs/coop_bimatrix_3/base.yaml \
--job-yaml-path   ./gcp_yaml_configs/base-intel-cpu-job.yaml

exit
