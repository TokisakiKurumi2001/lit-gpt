export WORK_DIR=llama13-4epoch
cd benchmark_output/runs/
mkdir ${WORK_DIR}
cp -rf baseline/eval_cache ${WORK_DIR}/
cd ../..
helm-run --conf-paths helm_conf/cnn.conf --suite ${WORK_DIR} -m 100 -n 1
# helm-run --conf-paths helm_conf/gsm.conf --suite ${WORK_DIR} -m 100 -n 1
# helm-run --conf-paths helm_conf/big_bench.conf --suite ${WORK_DIR} -m 100 -n 1
helm-run --conf-paths helm_conf/mcqa.conf --suite ${WORK_DIR} -m 100 -n 1
python summary_v1.py --version ${WORK_DIR}
