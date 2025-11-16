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
# quick test with minimal memory (~6gb ram)
./run_transfer_experiment.sh transfer_config_tiny.json

# moderate test (~12gb ram)
./run_transfer_experiment.sh transfer_config_small.json

# full experiment (~24gb ram)
./run_transfer_experiment.sh transfer_config.json
```

**what it does:**
- quantizes opt-350m (or gpt2) and extracts scaling factors
- transfers those factors to opt-1.3b (or gpt2-medium)
- compares: normal quantization vs transferred quantization
- analyzes transfer effectiveness

**configs available:**
- `transfer_config_tiny.json` - gpt2 → gpt2-medium (minimal memory)
- `transfer_config_small.json` - opt-350m → opt-1.3b (moderate memory)
- `transfer_config.json` - opt-350m → opt-1.3b (full accuracy)

**see:** 
- `TRANSFER_EXPERIMENT_README.md` for detailed documentation
- `MEMORY_OPTIMIZATION.md` if you encounter memory issues

### notes

- results are saved to `benchmark_results.json`
- see `README_QUANTIZATION.md` for details if needed


