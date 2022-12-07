python batch_launcher.py \
--exp-config-path ./yaml_configs/aie/train_baselines.yaml \
--base-exp-config-path ./yaml_configs/aie/base.yaml \
--job-yaml-path   ./gcp_yaml_configs/base-intel-cpu-job.yaml

exit

python batch_launcher.py \
--exp-config-path ./yaml_configs/aie/train_ermas.yaml \
--base-exp-config-path ./yaml_configs/aie/base.yaml \
--job-yaml-path   ./gcp_yaml_configs/base-intel-big-cpu-job.yaml

python batch_launcher.py \
--exp-config-path ./yaml_configs/aie/train_ermas_unused.yaml \
--base-exp-config-path ./yaml_configs/aie/base.yaml \
--job-yaml-path   ./gcp_yaml_configs/base-intel-big-cpu-job.yaml

exit

python batch_launcher.py \
--exp-config-path ./yaml_configs/aie/eval_entropy_3.yaml \
--base-exp-config-path ./yaml_configs/aie/base.yaml \
--job-yaml-path   ./gcp_yaml_configs/base-intel-cpu-job.yaml

python batch_launcher.py \
--exp-config-path ./yaml_configs/aie/eval_isoelastic_3.yaml \
--base-exp-config-path ./yaml_configs/aie/base.yaml \
--job-yaml-path   ./gcp_yaml_configs/base-intel-cpu-job.yaml

exit

