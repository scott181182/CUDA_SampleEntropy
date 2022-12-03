


build: target/sampen

target/sampen: src/*.cu
	nvcc --output-file target/sampen src/main.cu
# target/%: src/%.cu
# 	nvcc --output-file $@ $<

run: target/sampen
	./target/sampen

run_naive: target/sampen.cpu
	./target/sampen.cpu data/sine.csv
