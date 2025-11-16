### quick run

```bash
pip install torch transformers datasets tqdm
python test_quantization.py
python setup_config.py quick_test|comprehensive_benchmark|extreme_compression|pot_grid_search
python benchmark_runner.py config.json
# saves restults in benchmark_results.json 
```

### quantization methods

the framework includes the following quantization methods:

- awq
- gptq
- pot
- apot
- smoothquant

### other useful

- list configs

```bash
python setup_config.py list
```


- change model

```bash
# edit config.json → "model_name": "your/model"
python benchmark_runner.py config.json
```

### transfer quantization experiment (new!)

test whether quantization parameters from a smaller model can be transferred to a larger model:

```bash
# run the transfer experiment (350m → 1b)
python transfer_quantization_experiment.py

# or with custom config
python transfer_quantization_experiment.py transfer_config.json

# or use the shell script
chmod +x run_transfer_experiment.sh
./run_transfer_experiment.sh
```

**what it does:**
- quantizes mobilellm-350m and extracts scaling factors
- transfers those factors to mobilellm-1b
- compares: normal quantization vs transferred quantization
- analyzes transfer effectiveness

**see:** `TRANSFER_EXPERIMENT_README.md` for detailed documentation

### notes

- results are saved to `benchmark_results.json`
- see `README_QUANTIZATION.md` for details if needed


